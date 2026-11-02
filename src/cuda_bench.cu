#include "reference.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(expr) do { \
  cudaError_t error = (expr); \
  if (error != cudaSuccess) throw std::runtime_error(std::string(#expr) + ": " + cudaGetErrorString(error)); \
} while (0)

__global__ void vector_add(const float* a, const float* b, float* out, size_t n) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

struct Event {
    cudaEvent_t handle{};
    Event() { CUDA_CHECK(cudaEventCreate(&handle)); }
    ~Event() { if (handle) cudaEventDestroy(handle); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};
struct DeviceBuffer {
    float* p = nullptr;
    explicit DeviceBuffer(size_t bytes) { CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p), bytes)); }
    ~DeviceBuffer() { if (p) cudaFree(p); }
    DeviceBuffer(const DeviceBuffer&) = delete;
};
struct PinnedBuffer {
    float* p = nullptr;
    explicit PinnedBuffer(size_t bytes) { CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&p), bytes)); }
    ~PinnedBuffer() { if (p) cudaFreeHost(p); }
    PinnedBuffer(const PinnedBuffer&) = delete;
};

float time_copy_ms(void* dest, const void* src, size_t bytes, cudaMemcpyKind kind, int iterations) {
    // Host-wall time includes API overhead and any synchronization; useful for pageable vs pinned.
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) CUDA_CHECK(cudaMemcpy(dest, src, bytes, kind));
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<float, std::milli>(end - start).count() / iterations;
}

float time_kernel_ms(const float* a, const float* b, float* out, size_t n,
                     int block, int iterations) {
    Event start, stop;
    const unsigned grid = static_cast<unsigned>((n + block - 1) / block);
    for (int i = 0; i < 5; ++i) vector_add<<<grid, block>>>(a, b, out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start.handle));
    for (int i = 0; i < iterations; ++i) vector_add<<<grid, block>>>(a, b, out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop.handle));
    CUDA_CHECK(cudaEventSynchronize(stop.handle));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start.handle, stop.handle));
    return ms / iterations;
}

int main(int argc, char** argv) {
    try {
        size_t n = (argc > 1) ? std::stoull(argv[1]) : (1u << 20);
        int iterations = (argc > 2) ? std::stoi(argv[2]) : 50;
        if (n == 0 || n > (1u << 24) || iterations < 1 || iterations > 10000)
            throw std::invalid_argument("usage: cuda_bench [N: 1..16777216] [iterations: 1..10000]");
        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (device_count == 0) throw std::runtime_error("No NVIDIA CUDA device detected");
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        const size_t bytes = n * sizeof(float);
        std::vector<float> pageable_a(n), pageable_b(n), pageable_out(n), expected(n);
        for (size_t i = 0; i < n; ++i) {
            pageable_a[i] = static_cast<float>((i % 127) * 0.125f);
            pageable_b[i] = static_cast<float>((i % 59) * -0.25f);
        }
        vector_add_cpu(pageable_a.data(), pageable_b.data(), expected.data(), n);
        DeviceBuffer d_a(bytes), d_b(bytes), d_out(bytes);
        PinnedBuffer pin_a(bytes), pin_b(bytes), pin_out(bytes);
        std::copy(pageable_a.begin(), pageable_a.end(), pin_a.p);
        std::copy(pageable_b.begin(), pageable_b.end(), pin_b.p);
        const float pageable_h2d = time_copy_ms(d_a.p, pageable_a.data(), bytes,
                                                 cudaMemcpyHostToDevice, iterations);
        const float pinned_h2d = time_copy_ms(d_a.p, pin_a.p, bytes,
                                              cudaMemcpyHostToDevice, iterations);
        const float pageable_d2h = time_copy_ms(pageable_out.data(), d_a.p, bytes,
                                                 cudaMemcpyDeviceToHost, iterations);
        const float pinned_d2h = time_copy_ms(pin_out.p, d_a.p, bytes,
                                              cudaMemcpyDeviceToHost, iterations);
        CUDA_CHECK(cudaMemcpy(d_a.p, pageable_a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b.p, pageable_b.data(), bytes, cudaMemcpyHostToDevice));
        std::cout << std::fixed << std::setprecision(5);
        std::cout << "GPU=" << prop.name << " N=" << n << " iterations=" << iterations << "\n";
        std::cout << "metric,value_ms\n";
        std::cout << "pageable_h2d," << pageable_h2d << "\n";
        std::cout << "pinned_h2d," << pinned_h2d << "\n";
        std::cout << "pageable_d2h," << pageable_d2h << "\n";
        std::cout << "pinned_d2h," << pinned_d2h << "\n";
        for (int block : {128, 256, 512}) {
            int max_threads = prop.maxThreadsPerBlock;
            if (block > max_threads) continue;
            const float kernel_ms = time_kernel_ms(d_a.p, d_b.p, d_out.p, n, block, iterations);
            CUDA_CHECK(cudaMemcpy(pageable_out.data(), d_out.p, bytes, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < n; ++i)
                if (!close_enough(pageable_out[i], expected[i]))
                    throw std::runtime_error("GPU correctness failed at index " + std::to_string(i));
            // Naive nominal effective bandwidth: two input reads + one output write.
            double gbps = (3.0 * bytes) / (kernel_ms / 1000.0) / 1e9;
            std::cout << "kernel_block" << block << "," << kernel_ms << "\n";
            std::cout << "kernel_block" << block << "_nominal_gbps," << gbps << "\n";
        }
        std::cout << "PASS: all CUDA outputs match CPU reference\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
}
