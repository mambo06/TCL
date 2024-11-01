import torch
import time

def benchmark_matmul(device, size=2000):
    # Create random matrices
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # Warm up
    for _ in range(3):
        torch.matmul(a, b)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Measure time
    start_time = time.time()
    
    # Perform matrix multiplication multiple times
    num_iterations = 10
    for _ in range(num_iterations):
        c = torch.matmul(a, b)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_iterations
    
    return avg_time

# Test sizes
# sizes = [100, 2000, 3000]

print("Device Information:")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
print(f"PyTorch Version: {torch.__version__}")
print("\nBenchmarking Matrix Multiplication:")
print(f"{'Size':>10} {'CPU Time (s)':>15} {'GPU Time (s)':>15}")
print("-" * 40)

for size in range(1,3001,200):
    # CPU benchmark
    cpu_time = benchmark_matmul(torch.device('cpu'), size)
    
    # GPU benchmark (if available)
    if torch.cuda.is_available():
        gpu_time = benchmark_matmul(torch.device('cuda'), size)
    else:
        gpu_time = float('nan')
    
    print(f"{size:>10} {cpu_time:>15.4f} {gpu_time:>15.4f}")