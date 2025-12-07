#pragma once

#ifdef ARTEMIS_CUDA_ENABLED
#include <cuda_runtime.h>
#endif

#include <stdexcept>
#include <string>
#include <memory>

namespace artemis {
namespace gpu {

#ifdef ARTEMIS_CUDA_ENABLED

// ============================================================================
// CUDA Error Checking
// ============================================================================

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t error = call;                                             \
        if (error != cudaSuccess) {                                           \
            throw ::artemis::gpu::CudaException(                              \
                error, __FILE__, __LINE__, #call);                            \
        }                                                                     \
    } while(0)

#define CUDA_CHECK_LAST_ERROR()                                               \
    do {                                                                      \
        cudaError_t error = cudaGetLastError();                               \
        if (error != cudaSuccess) {                                           \
            throw ::artemis::gpu::CudaException(                              \
                error, __FILE__, __LINE__, "cudaGetLastError()");             \
        }                                                                     \
    } while(0)

// ============================================================================
// CUDA Exception
// ============================================================================

class CudaException : public std::runtime_error {
public:
    CudaException(cudaError_t error, const char* file, int line, const char* call)
        : std::runtime_error(format_message(error, file, line, call))
        , error_code_(error) {}

    cudaError_t error_code() const { return error_code_; }

private:
    cudaError_t error_code_;

    static std::string format_message(
        cudaError_t error,
        const char* file,
        int line,
        const char* call) {
        return std::string("CUDA Error: ") + cudaGetErrorString(error) +
               "\n  File: " + file +
               "\n  Line: " + std::to_string(line) +
               "\n  Call: " + call;
    }
};

// ============================================================================
// GPU Memory Manager (RAII)
// ============================================================================

template<typename T>
class DeviceBuffer {
public:
    DeviceBuffer() : data_(nullptr), size_(0) {}

    explicit DeviceBuffer(size_t size) : data_(nullptr), size_(0) {
        allocate(size);
    }

    ~DeviceBuffer() {
        free();
    }

    // No copy (move only)
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // Move semantics
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            free();
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    void allocate(size_t size) {
        free();
        if (size > 0) {
            CUDA_CHECK(cudaMalloc(&data_, size * sizeof(T)));
            size_ = size;
        }
    }

    void free() {
        if (data_ != nullptr) {
            cudaFree(data_);
            data_ = nullptr;
            size_ = 0;
        }
    }

    void upload(const T* host_data, size_t count) {
        if (count > size_) {
            allocate(count);
        }
        CUDA_CHECK(cudaMemcpy(data_, host_data, count * sizeof(T),
                              cudaMemcpyHostToDevice));
    }

    void download(T* host_data, size_t count) const {
        if (count > size_) {
            throw std::runtime_error("Download count exceeds buffer size");
        }
        CUDA_CHECK(cudaMemcpy(host_data, data_, count * sizeof(T),
                              cudaMemcpyDeviceToHost));
    }

    void zero() {
        if (data_ != nullptr) {
            CUDA_CHECK(cudaMemset(data_, 0, size_ * sizeof(T)));
        }
    }

    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }

private:
    T* data_;
    size_t size_;
};

// ============================================================================
// GPU Device Information
// ============================================================================

struct DeviceInfo {
    int device_id;
    std::string name;
    size_t total_memory;
    size_t free_memory;
    int compute_capability_major;
    int compute_capability_minor;
    int multiprocessor_count;
    int max_threads_per_block;
    int max_threads_per_multiprocessor;

    static DeviceInfo get_current_device();
    static DeviceInfo get_device(int device_id);
    static int get_device_count();
};

// ============================================================================
// Kernel Launch Configuration
// ============================================================================

struct LaunchConfig {
    dim3 grid_size;
    dim3 block_size;
    size_t shared_mem_bytes;
    cudaStream_t stream;

    LaunchConfig()
        : grid_size(1), block_size(256), shared_mem_bytes(0), stream(0) {}

    static LaunchConfig compute_1d(size_t total_threads, int block_size = 256);
    static LaunchConfig compute_2d(size_t width, size_t height, dim3 block_size = dim3(16, 16));
};

// ============================================================================
// GPU Stream Manager (RAII)
// ============================================================================

class Stream {
public:
    Stream() : stream_(nullptr) {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }

    ~Stream() {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }

    // No copy (move only)
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    Stream(Stream&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }

    Stream& operator=(Stream&& other) noexcept {
        if (this != &other) {
            if (stream_ != nullptr) {
                cudaStreamDestroy(stream_);
            }
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }

    void synchronize() {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    cudaStream_t get() const { return stream_; }
    operator cudaStream_t() const { return stream_; }

private:
    cudaStream_t stream_;
};

// ============================================================================
// GPU Event for Timing (RAII)
// ============================================================================

class Event {
public:
    Event() : event_(nullptr) {
        CUDA_CHECK(cudaEventCreate(&event_));
    }

    ~Event() {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }

    // No copy (move only)
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    void record(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(event_, stream));
    }

    void synchronize() {
        CUDA_CHECK(cudaEventSynchronize(event_));
    }

    float elapsed_time(const Event& start) const {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_, event_));
        return ms;
    }

    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_;
};

#endif // ARTEMIS_CUDA_ENABLED

} // namespace gpu
} // namespace artemis
