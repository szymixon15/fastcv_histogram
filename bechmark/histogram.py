import time

import cv2
import torch
import fastcv
import numpy as np

def benchmark_histogram(sizes=[1024, 2048, 4096], runs=50):
    results = []
    
    for size in sizes:
        print(f"\n=== Benchmarking {size}x{size} image ===")
        
        img_np = np.random.randint(0, 2, (size, size), dtype=np.uint8) * 255
        img_torch = torch.from_numpy(img_np).cuda()

        print(img_torch)

        start = time.perf_counter()
        for _ in range(runs):
            _ = cv2.calcHist(img_np, [0], None, [256], [0, 256]) #obraz, [0] - kanal (pierwszy i jedyny), none - brak maski, [256] - l. kubelkow, [0, 256] - zakres wartosci pikseli
            
        end = time.perf_counter()
        cv_time = (end - start) / runs * 1000  # ms per run

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            _ = fastcv.histogram_cub(img_torch)
        torch.cuda.synchronize()
        end = time.perf_counter()
        fc_cub_time = (end - start) / runs * 1000  # ms per run

        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            _ = fastcv.histogram_thrust(img_torch)
        torch.cuda.synchronize()
        end = time.perf_counter()
        fc_thrust_time = (end - start) / runs * 1000  # ms per run

        results.append((size, cv_time, fc_cub_time, fc_thrust_time))
        print(f"OpenCV (CPU): {cv_time:.4f} ms | fastcv (CUB): {fc_cub_time:.4f} ms | fastcv (THRUST): {fc_thrust_time:.4f} ms")
    
    return results


if __name__ == "__main__":
    results = benchmark_histogram()
    print("\n=== Final Results ===")
    print("Size\t\tOpenCV (CPU)\tfastcv (CUB)\tfastcv (THRUST)")
    for size, cv_time, fc_cub_time, fc_thrust_time in results:
        print(f"{size}x{size}\t{cv_time:.4f} ms\t{fc_cub_time:.4f} ms\t{fc_thrust_time:.4f} ms")
