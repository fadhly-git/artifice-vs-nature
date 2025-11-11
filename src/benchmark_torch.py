# Benchmark Torch Performance
import time
import torch

def benchmark_torch(device='cpu', size=4096, runs=10):
    print(f"Benchmarking torch.matmul on {device} with size {size}x{size}, {runs} runs...")
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    times = []
    for i in range(runs):
        start = time.time()
        c = torch.matmul(a, b)
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"Run {i+1}: {elapsed:.4f} sec")
    avg = sum(times) / len(times)
    print(f"\nAverage time: {avg:.4f} sec")
    return avg

if __name__ == "__main__":
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    benchmark_torch(device='cpu')
    if torch.cuda.is_available():
        benchmark_torch(device='cuda')
