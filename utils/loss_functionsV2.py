
import numpy as np
import torch as th
import torch.nn.functional as F


def getMSEloss(recon, target):
    """

    Args:
        recon (torch.FloatTensor):
        target (torch.FloatTensor):

    """
    dims = list(target.size())
    bs = dims[0]
    loss = th.sum(th.square(recon - target)) / bs
    return loss


def getBCELoss(prediction, label):
    """

    Args:
        prediction (torch.FloatTensor):
        label (torch.FloatTensor):

    """
    dims = list(prediction.size())
    bs = dims[0]
    return F.binary_cross_entropy(prediction, label, reduction='sum') / bs


class JointLoss(th.nn.Module):
    """
    Modifed from: https://github.com/sthalles/SimCLR/blob/master/loss/nt_xent.py
    When computing loss, we are using a 2Nx2N similarity matrix, in which positve samples are on the diagonal of four
    quadrants while negatives are all the other samples as shown below in 8x8 array, where we assume batch_size=4.
                                        P . . . P . . .
                                        . P . . . P . .
                                        . . P . . . P .
                                        . . . P . . . P
                                        P . . . P . . .
                                        . P . . . P . .
                                        . . P . . . P .
                                        . . . P . . . P
    """

    def __init__(self, options):
        super(JointLoss, self).__init__()
        # Assign options to self
        self.options = options
        # Batch size
        self.batch_size = options["batch_size"]
        # Temperature to use scale logits
        self.temperature = options["tau"]
        # Device to use: GPU or CPU
        self.device = options["device"]
        # initialize softmax
        self.softmax = th.nn.Softmax(dim=-1)
        # Mask to use to get negative samples from similarity matrix
        self.mask_for_neg_samples = self._get_mask_for_neg_samples().type(th.bool)
        # Function to generate similarity matrix: Cosine, or Dot product
        self.similarity_fn = self._dot_simililarity
        # Loss function
        self.criterion = th.nn.CrossEntropyLoss(reduction="sum")
        self.mseLoss = th.nn.MSELoss()


    def _get_mask_for_neg_samples(self):
        # Diagonal 2Nx2N identity matrix, which consists of four (NxN) quadrants
        diagonal = np.eye( self.batch_size)
        # Diagonal 2Nx2N matrix with 1st quadrant being identity matrix
        q1 = np.eye((self.batch_size),  self.batch_size, k=self.batch_size)
        # Diagonal 2Nx2N matrix with 3rd quadrant being identity matrix
        q3 = np.eye(( self.batch_size),  self.batch_size, k=-self.batch_size)
        # Generate mask with diagonals of all four quadrants being 1.
        mask = th.from_numpy((diagonal + q1 + q3))
        # Reverse the mask: 1s become 0, 0s become 1. This mask will be used to select negative samples
        mask = (1 - mask).type(th.bool)
        # Transfer the mask to the device and return
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        # print(x.shape,y.shape)
        x = x[:int(x.shape[0]/2)]
        y = y[int(y.shape[0]/2):]
        # print(x.shape,y.shape)
        # # Reshape x: (2N, C) -> (2N, 1, C)
        x = x.unsqueeze(1)
        # # # Reshape y: (2N, C) -> (1, C, 2N)
        y = y.unsqueeze(1)
        # # Similarity shape: (2N, 2N)
        # print(x.shape,y.shape)
        similarity = th.tensordot(x, y, dims=2)
        # print(similarity.shape)
        # similarity = th.tensordot(x, y)
        return similarity


    def XNegloss(self, representation):
        # Compute similarity matrix
        similarity = self.similarity_fn(representation, representation)

        logits = similarity

        logits /= self.temperature
        
        if self.options['task_type'] == 'regression' :
            # labels = th.zeros( self.batch_size).to(self.device).float()
            labels = th.zeros( logits.shape).to(self.device).float()
            loss = self.mseLoss(logits, labels)
        else:
            labels = th.zeros( self.batch_size).to(self.device).long()
            # Compute total loss
            loss = self.criterion(logits, labels)
            
        # Loss per sample
        closs = loss / ( self.batch_size)
        # Return contrastive loss
        return closs

    def forward(self, representation, xrecon, xorig):
        """

        Args:
            representation (torch.FloatTensor):
            xrecon (torch.FloatTensor):
            xorig (torch.FloatTensor):

        """

        # recontruction loss
        recon_loss = getMSEloss(xrecon, xorig) if self.options["reconstruction"] else getBCELoss(xrecon, xorig)

        # Initialize contrastive and distance losses with recon_loss as placeholder
        closs, zrecon_loss = recon_loss, recon_loss

        # Start with default loss i.e. reconstruction loss
        loss = recon_loss

        if self.options["contrastive_loss"]:
            closs = self.XNegloss(representation)
            loss = loss + closs

        if self.options["distance_loss"]:
            # recontruction loss for z
            zi, zj = th.split(representation, self.batch_size)

            # print(representation.shape, zi,zj)

            zrecon_loss = getMSEloss(zi, zj)
            # print(zrecon_loss)
            loss = loss + zrecon_loss

        # Return
        return loss, closs, recon_loss, zrecon_loss
