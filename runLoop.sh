#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=CFL-covtype
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -o out_covtype.txt
#SBATCH -e error_covtype.txt
#SBATCH --partition=general 
#SBATCH --account=a_xue_li
#SBATCH --time=120:00:00

module load miniconda3

source activate /scratch/user/uqaginan/RQ3


for i in adult year covtype microsoft  epsilon aloi helena yahoo higgs_small california_housing jannis

do
  echo "Running training on dataset : $i"
  python train.py -d $i -e 100
done