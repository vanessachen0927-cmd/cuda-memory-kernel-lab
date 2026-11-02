#include "reference.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do { \
    const cudaError_t err = (call); \
    if (err != cudaSuccess) \
        throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(err)); \
} while (0)

__global__ void vector_add_kernel(const float* a, const float* b,
                                  float* out, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

struct Pinned {
    float* data = nullptr;
    explicit Pinned(size_t bytes) {
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&data), bytes));
    }
    ~Pinned() { if (data) cudaFreeHost(data); }
    Pinned(const Pinned&) = delete;
    Pinned& operator=(const Pinned&) = delete;
};

struct Device {
    float* data = nullptr;
    explicit Device(size_t bytes) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), bytes));
    }
    ~Device() { if (data) cudaFree(data); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};

struct Streams {
    std::vector<cudaStream_t> handles;
    explicit Streams(int count) {
        try {
            for (int i = 0; i < count; ++i) {
                cudaStream_t s{};
                CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
                handles.push_back(s);
            }
        } catch (...) {
            for (auto s : handles) cudaStreamDestroy(s);
            throw;
        }
    }
    ~Streams() {
        for (auto s : handles) cudaStreamDestroy(s);
    }
    Streams(const Streams&) = delete;
    Streams& operator=(const Streams&) = delete;
};

void check_output(const float* a, const float* b, const float* out, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (!close_enough(out[i], a[i] + b[i]))
            throw std::runtime_error("GPU result mismatch at " +
                                     std::to_string(i));
}

// Asynchronously enqueue non-overlapping H2D -> kernel -> D2H operations.
// Each chunk stays ordered within its stream; different chunks may overlap.
// Uses NON-BLOCKING streams to avoid legacy default-stream synchronization.
void submit_pipeline(const float* h_a, const float* h_b, float* h_out,
                     float* d_a, float* d_b, float* d_out, size_t n,
                     int chunk_count, const Streams& streams) {
    const size_t chunk_elements = (n + static_cast<size_t>(chunk_count) - 1) /
                                  static_cast<size_t>(chunk_count);
    constexpr int block = 256;
    for (int chunk = 0; chunk < chunk_count; ++chunk) {
        const size_t start = static_cast<size_t>(chunk) * chunk_elements;
        if (start >= n) break;
        const size_t count = std::min(chunk_elements, n - start);
        const size_t bytes = count * sizeof(float);
        cudaStream_t stream = streams.handles[
            static_cast<size_t>(chunk) % streams.handles.size()];
        CUDA_CHECK(cudaMemcpyAsync(d_a + start, h_a + start, bytes,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_b + start, h_b + start, bytes,
                                   cudaMemcpyHostToDevice, stream));
        const unsigned grid = static_cast<unsigned>((count + block - 1) / block);
        vector_add_kernel<<<grid, block, 0, stream>>>(
            d_a + start, d_b + start, d_out + start, count);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(h_out + start, d_out + start, bytes,
                                   cudaMemcpyDeviceToHost, stream));
    }
    // Synchronize all work, including the final D2H, before using host output.
    CUDA_CHECK(cudaDeviceSynchronize());
}

struct Measurement {
    double mean_ms = 0;
    double min_ms = 0;
    double max_ms = 0;
};

// Wall-clock includes host enqueue overhead and synchronization, but excludes
// allocations and setup; this is end-to-end H2D+compute+D2H per iteration.
Measurement measure(const float* h_a, const float* h_b, float* h_out,
                    float* d_a, float* d_b, float* d_out, size_t n,
                    int chunks, int stream_count, int iterations) {
    Streams streams(stream_count);
    // Warm up kernels, memory paths and CUDA runtime.
    for (int i = 0; i < 3; ++i)
        submit_pipeline(h_a, h_b, h_out, d_a, d_b, d_out, n,
                        chunks, streams);
    check_output(h_a, h_b, h_out, n);
    Measurement m;
    double total = 0;
    for (int i = 0; i < iterations; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        submit_pipeline(h_a, h_b, h_out, d_a, d_b, d_out, n,
                        chunks, streams);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(
            t1 - t0).count();
        if (i == 0 || ms < m.min_ms) m.min_ms = ms;
        if (i == 0 || ms > m.max_ms) m.max_ms = ms;
        total += ms;
    }
    m.mean_ms = total / iterations;
    check_output(h_a, h_b, h_out, n);
    return m;
}

int main(int argc, char** argv) {
    try {
        const size_t n = argc > 1 ? std::stoull(argv[1]) : (1u << 23);
        const int iterations = argc > 2 ? std::stoi(argv[2]) : 20;
        const int chunks = argc > 3 ? std::stoi(argv[3]) : 16;
        if (n < 1024 || n > (1u << 24) || iterations < 1 ||
            iterations > 1000 || chunks < 2 || chunks > 256 ||
            static_cast<size_t>(chunks) > n)
            throw std::invalid_argument(
                "usage: stream_pipeline [N: 1024..16777216] "
                "[iterations: 1..1000] [chunks: 2..256]");

        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        const size_t bytes = n * sizeof(float);
        Pinned a(bytes), b(bytes), out(bytes);
        Device da(bytes), db(bytes), dout(bytes);
        for (size_t i = 0; i < n; ++i) {
            a.data[i] = static_cast<float>((i % 127) * 0.125f);
            b.data[i] = static_cast<float>((i % 59) * -0.25f);
        }

        // Both variants use identical pinned allocations and chunk sizes.
        // Only the number of nonblocking CUDA streams differs.
        const auto single = measure(a.data, b.data, out.data,
                                    da.data, db.data, dout.data,
                                    n, chunks, 1, iterations);
        const auto multi = measure(a.data, b.data, out.data,
                                   da.data, db.data, dout.data,
                                   n, chunks, 4, iterations);

        std::cout << std::fixed << std::setprecision(5);
        std::cout << "GPU=" << prop.name << " N=" << n
                  << " iterations=" << iterations
                  << " chunks=" << chunks << "\n";
        std::cout << "asyncEngineCount=" << prop.asyncEngineCount
                  << " deviceOverlap=" << prop.deviceOverlap << "\n";
        std::cout << "metric,value\n";
        std::cout << "single_stream_mean_ms," << single.mean_ms << "\n";
        std::cout << "single_stream_min_ms," << single.min_ms << "\n";
        std::cout << "single_stream_max_ms," << single.max_ms << "\n";
        std::cout << "four_stream_mean_ms," << multi.mean_ms << "\n";
        std::cout << "four_stream_min_ms," << multi.min_ms << "\n";
        std::cout << "four_stream_max_ms," << multi.max_ms << "\n";
        std::cout << "single_over_four_speedup,"
                  << single.mean_ms / multi.mean_ms << "\n";
        std::cout << "PASS: single-stream and four-stream GPU outputs "
                     "match CPU reference\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
}
