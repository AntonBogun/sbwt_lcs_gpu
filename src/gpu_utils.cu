#include "gpu/gpu_utils.cuh"
// clang-format off
namespace sbwt_lcs_gpu {
u64 get_free_gpu_memory() {
    u64 free = 0;
    u64 total = 0;
    cudaMemGetInfo(&free, &total);
    return free;
}

u64 get_total_gpu_memory() {
    u64 free = 0;
    u64 total = 0;
    cudaMemGetInfo(&free, &total);
    return total;
}

u64 get_taken_gpu_memory() {
    u64 free = 0;
    u64 total = 0;
    cudaMemGetInfo(&free, &total);
    return total - free;
}

// GPU Stream

// GpuStream::GpuStream(bool default_stream) {
// if (default_stream) {
//   element = nullptr;
// } else {
//   element = static_cast<void *>(new cudaStream_t{});
//   cudaStreamCreate(static_cast<cudaStream_t *>(element));
// }
GpuStream::GpuStream() {
    element = static_cast<void *>(new cudaStream_t{});
    cudaStreamCreate(static_cast<cudaStream_t *>(element));
}
GpuStream::GpuStream(GpuStream &&other) {
    element = other.element;
    other.element = nullptr;
}

GpuStream::~GpuStream() {
    if (element != nullptr) { // only when got r-value moved
        cudaStreamDestroy(*static_cast<cudaStream_t *>(element));
        delete static_cast<cudaStream_t *>(element);
    }
    // cudaStreamDestroy(*static_cast<cudaStream_t *>(element));
    // delete static_cast<cudaStream_t *>(element);
}

void *GpuStream::data() const { return element; }

void GpuStream::sync() {
    CUDA_CHECK(
        cudaStreamSynchronize(*static_cast<cudaStream_t *>(element))
    );
}

cudaStream_t *getStream(GpuStream &stream) {
    return static_cast<cudaStream_t *>(stream.data());
}

// GPU Pointer

template <class T> GpuPointer<T>::GpuPointer(u64 size) : bytes(size * sizeof(T)), owning_pointer(true) {
    CUDA_CHECK(
        cudaMalloc((void **)(&ptr), bytes)
    );
}
template <class T> GpuPointer<T>::GpuPointer(const T *cpu_ptr, u64 size) : GpuPointer(size) {
    CUDA_CHECK(
        cudaMemcpy(ptr, cpu_ptr, bytes, cudaMemcpyHostToDevice)
    );
}
template <class T> GpuPointer<T>::GpuPointer(const std::vector<T> &v) : GpuPointer<T>(v.data(), v.size()) {}

template <class T>
GpuPointer<T>::GpuPointer(GpuPointer<T> &other, u64 offset, u64 amount)
    : ptr(other.ptr + offset), bytes(amount * sizeof(T)), owning_pointer(false) {}

template <class T>
GpuPointer<T>::GpuPointer(u64 size, GpuStream &gpu_stream) : bytes(size * sizeof(T)), owning_pointer(true) {
    CUDA_CHECK(
        cudaMallocAsync(
            // (void **)(&ptr), bytes, *reinterpret_cast<cudaStream_t *>(gpu_stream.data())
            (void **)(&ptr),
            bytes,
            *getStream(gpu_stream)
        )
    );
}
template <class T> GpuPointer<T>::GpuPointer(const T *cpu_ptr, u64 size, GpuStream &gpu_stream) : GpuPointer(size) {
    CUDA_CHECK(
        cudaMemcpyAsync(
            ptr,
            cpu_ptr,
            bytes,
            cudaMemcpyHostToDevice,
            // *reinterpret_cast<cudaStream_t *>(gpu_stream.data())
            *getStream(gpu_stream)
        )
    );
}
template <class T>
GpuPointer<T>::GpuPointer(const std::vector<T> &v, GpuStream &gpu_stream)
    : GpuPointer<T>(v.data(), v.size(), gpu_stream) {}

template <class T>
GpuPointer<T>::GpuPointer(GpuPointer<T> &&other)
    : ptr(other.ptr), bytes(other.bytes), owning_pointer(other.owning_pointer) {
    other.ptr = nullptr;
    other.bytes = 0;
    other.owning_pointer = false;
}

template <class T> T *GpuPointer<T>::data() const { return ptr; }

template <class T> void GpuPointer<T>::memset(u64 index, uint8_t value) {
    CUDA_CHECK(
        cudaMemset(ptr + index, value, bytes)
    );
}

template <class T> void GpuPointer<T>::memset(u64 index, u64 amount, uint8_t value) {
    CUDA_CHECK(
        cudaMemset(ptr + index, value, amount * sizeof(T))
    );
}

template <class T> void GpuPointer<T>::memset_async(u64 index, uint8_t value, GpuStream &gpu_stream) {
    CUDA_CHECK(
        cudaMemsetAsync(
            ptr + index,
            value,
            bytes,
            // *reinterpret_cast<cudaStream_t *>(gpu_stream.data())
            *getStream(gpu_stream)
        )
    );
}

template <class T> void GpuPointer<T>::memset_async(u64 index, u64 amount, uint8_t value, GpuStream &gpu_stream) {
    CUDA_CHECK(
        cudaMemsetAsync(
            ptr + index,
            value,
            amount * sizeof(T),
            // *reinterpret_cast<cudaStream_t *>(gpu_stream.data())
            *getStream(gpu_stream)
        )
    );
}

template <class T> void GpuPointer<T>::set(const T *source, u64 amount, u64 destination_index) {
    CUDA_CHECK(
        cudaMemcpy(
            ptr + destination_index,
            source,
            amount * sizeof(T),
            cudaMemcpyHostToDevice
        )
    );
}
template <class T> void GpuPointer<T>::set(const std::vector<T> &source, u64 amount, u64 destination_index) {
    set(source.data(), amount, destination_index);
}

template <class T>
void GpuPointer<T>::set_async(const T *source, u64 amount, GpuStream &gpu_stream, u64 destination_index) {
    CUDA_CHECK(
        cudaMemcpyAsync(
            ptr + destination_index,
            source,
            amount * sizeof(T),
            cudaMemcpyHostToDevice,
            // *reinterpret_cast<cudaStream_t *>(gpu_stream.data())
            *getStream(gpu_stream)
        )
    );
}
template <class T>
void GpuPointer<T>::set_async(const std::vector<T> &source, u64 amount, GpuStream &gpu_stream, u64 destination_index) {
    set_async(source.data(), amount, gpu_stream, destination_index);
}

template <class T> void GpuPointer<T>::copy_to(T *destination, u64 amount) const {
    CUDA_CHECK(
        cudaMemcpy(destination,
            ptr,
            amount * sizeof(T),
            cudaMemcpyDeviceToHost
        )
    );
}
template <class T> void GpuPointer<T>::copy_to(T *destination) const { copy_to(destination, bytes / sizeof(T)); }
template <class T> void GpuPointer<T>::copy_to(std::vector<T> &destination, u64 amount) const {
    copy_to(destination.data(), amount);
}
template <class T> void GpuPointer<T>::copy_to(std::vector<T> &destination) const {
    destination.resize(bytes / sizeof(T));
    copy_to(destination.data());
}

template <class T> void GpuPointer<T>::copy_to_async(T *destination, u64 amount, GpuStream &gpu_stream) const {
    CUDA_CHECK(
        cudaMemcpyAsync(
            destination,
            ptr,
            amount * sizeof(T),
            cudaMemcpyDeviceToHost,
            // *reinterpret_cast<cudaStream_t *>(gpu_stream.data())
            *getStream(gpu_stream)
        )
    );
}
template <class T> void GpuPointer<T>::copy_to_async(T *destination, GpuStream &gpu_stream) const {
    copy_to_async(destination, bytes / sizeof(T), gpu_stream);
}
template <class T>
void GpuPointer<T>::copy_to_async(std::vector<T> &destination, u64 amount, GpuStream &gpu_stream) const {
    copy_to_async(destination.data(), amount, gpu_stream);
}
template <class T> void GpuPointer<T>::copy_to_async(std::vector<T> &destination, GpuStream &gpu_stream) const {
    destination.resize(bytes / sizeof(T));
    copy_to_async(destination.data(), gpu_stream);
}

template <class T> GpuPointer<T>::~GpuPointer() {
    if (owning_pointer) {
        try {
            CUDA_CHECK(cudaFree(ptr));
        } catch (std::runtime_error &e) {
            std::cerr << e.what() << std::endl;
        }
    }
}

template class GpuPointer<char>;
template class GpuPointer<float>;
template class GpuPointer<double>;
template class GpuPointer<uint64_t>;
template class GpuPointer<int64_t>;
template class GpuPointer<uint32_t>;
template class GpuPointer<int32_t>;
template class GpuPointer<uint16_t>;
template class GpuPointer<uint8_t>;

template class GpuPointer<char *>;
template class GpuPointer<float *>;
template class GpuPointer<double *>;
template class GpuPointer<uint64_t *>;
template class GpuPointer<int64_t *>;
template class GpuPointer<uint32_t *>;
template class GpuPointer<int32_t *>;
template class GpuPointer<uint8_t *>;
template class GpuPointer<uint16_t *>;

// GPU Event

GpuEvent::GpuEvent() : element(static_cast<void *>(new cudaEvent_t{})) {
    cudaEventCreate(static_cast<cudaEvent_t *>(element));
}
GpuEvent::GpuEvent(GpuEvent &&other) : element(other.element) { other.element = nullptr; }

GpuEvent::~GpuEvent() {
    if (element == nullptr)
        return; // only when got r-value moved
    cudaEventDestroy(*static_cast<cudaEvent_t *>(element));
    delete static_cast<cudaEvent_t *>(element);
}
void GpuEvent::record(GpuStream *s) {
    CUDA_CHECK(
        cudaEventRecord(
            *static_cast<cudaEvent_t *>(element),
            s == nullptr ? nullptr : *getStream(*s)
        )
    );
}
void GpuEvent::record(GpuStream &s) { record(&s); }

void *GpuEvent::get() const { return element; }

float GpuEvent::time_elapsed_ms(const GpuEvent &e) {
    float millis = -1;
    CUDA_CHECK(
        cudaEventElapsedTime(
            &millis,
            *static_cast<cudaEvent_t *>(e.get()),
            *static_cast<cudaEvent_t *>(element)
        )
    );
    return millis;
}

void GpuEvent::sync() { CUDA_CHECK(
    cudaEventSynchronize(*static_cast<cudaEvent_t *>(element))
); }

cudaEvent_t *getEvent(GpuEvent &event) { return static_cast<cudaEvent_t *>(event.get()); }

} // namespace sbwt_lcs_gpu