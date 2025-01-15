#pragma once

#include "utils.h"
#include <sstream>
#include <stdexcept>

namespace sbwt_lcs_gpu {

// This is kept as a macro instead of converting it to a modern c++ constexpr
// because otherwise __FILE__ and __LINE__ will not work as intended, ie to
// report where the error is
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define CUDA_CHECK(code_block)                                                                                         \
    {                                                                                                                  \
        auto code = code_block;                                                                                        \
        if (code != cudaSuccess) {                                                                                     \
            std::stringstream ss;                                                                                      \
            ss << "GPUassert: " << cudaGetErrorString(code) << " at " << __FILE__ << ":" << __LINE__ << '\n';          \
            throw std::runtime_error(ss.str());                                                                        \
        }                                                                                                              \
    }

u64 get_free_gpu_memory();
u64 get_total_gpu_memory();
u64 get_taken_gpu_memory();

class GpuStream {
  private:
    void *element;

  public:
    // cannot use default stream since data() returns nullptr (cannot deref)
    //  GpuStream(bool default_stream = false);
    GpuStream();
    GpuStream(GpuStream &) = delete;
    auto operator=(GpuStream &) = delete;
    auto operator=(GpuStream &&) = delete;
    GpuStream(GpuStream &&);//support move constructor
    ~GpuStream();
    [[nodiscard]] void *data() const;
    void sync();
};

template <class T> class GpuPointer {
  private:
    T *ptr;
    u64 bytes = 0;
    bool owning_pointer;

  public:
    // owns pointer:
    // malloc size of sizeof(T) bytes of memory on the gpu
    explicit GpuPointer(u64 size);
    // malloc then copy the data from v/cpu_ptr to the gpu (host to device)
    explicit GpuPointer(const std::vector<T> &v);
    GpuPointer(const T *cpu_ptr, u64 size);

    // does not own pointer:
    // create a pointer that points to the memory of another pointer at offset and amount max size
    //(offset and amount in sizeof(T))
    GpuPointer(GpuPointer<T> &other, u64 offset, u64 amount);

    // like above but async and with a stream
    GpuPointer(u64 size, GpuStream &gpu_stream);
    GpuPointer(const std::vector<T> &v, GpuStream &gpu_stream);
    GpuPointer(const std::vector<T> &&v, GpuStream &gpu_stream);
    GpuPointer(const T *cpu_ptr, u64 size, GpuStream &gpu_stream);

    // delete copy and move constructors and operators
    GpuPointer(GpuPointer &) = delete;
    GpuPointer operator=(GpuPointer &) = delete;
    GpuPointer operator=(GpuPointer &&) = delete;
    GpuPointer(GpuPointer &&);//support move constructor

    // memset in sizeof(T) bytes
    void memset(u64 index, u64 amount, uint8_t value);
    void memset(u64 index, uint8_t value);
    // same but async
    void memset_async(u64 index, u64 amount, uint8_t value, GpuStream &gpu_stream);
    void memset_async(u64 index, uint8_t value, GpuStream &gpu_stream);

    // get ptr
    T *data() const;

    // memcpy in sizeof(T) bytes (like in constructor), host to device
    void set(const T *source, u64 amount, u64 destination_index = 0);
    void set(const std::vector<T> &source, u64 amount, u64 destination_index = 0);
    // same but async
    void set_async(const T *source, u64 amount, GpuStream &gpu_stream, u64 destination_index = 0);
    void set_async(const std::vector<T> &source, u64 amount, GpuStream &gpu_stream, u64 destination_index = 0);
    // memcpy in sizeof(T) bytes, device to host
    void copy_to(T *destination, u64 amount) const;
    void copy_to(T *destination) const;
    void copy_to(std::vector<T> &destination, u64 amount) const;
    //! this resizes the vector to the amount
    void copy_to(std::vector<T> &destination) const;

    // same but async
    void copy_to_async(T *destination, u64 amount, GpuStream &gpu_stream) const;
    void copy_to_async(T *destination, GpuStream &gpu_stream) const;
    void copy_to_async(std::vector<T> &destination, u64 amount, GpuStream &gpu_stream) const;
    void copy_to_async(std::vector<T> &destination, GpuStream &gpu_stream) const;
    // frees the ptr
    ~GpuPointer();
};

class GpuEvent {
    void *element;

  public:
    GpuEvent();
    GpuEvent(GpuEvent &) = delete;
    auto operator=(GpuEvent &) = delete;
    auto operator=(GpuEvent &&) = delete;
    GpuEvent(GpuEvent &&); //support move constructor
    ~GpuEvent();

    // if nullptr then record on the default stream
    void record(GpuStream *s = nullptr);
    void record(GpuStream &s);

    [[nodiscard]] void * get() const;
    // call this function from the end timer, give start as argument
    float time_elapsed_ms(const GpuEvent &e);
    void sync();
};

inline bool gpu_get_bitvector_bit(const u64 *const bitvector, u64 bit) {
    return (bitvector[bit / 64] >> (bit % 64)) & 1;
}
inline bool gpu_get_two_bit_char(const u64 *const bitvector, u64 indx) {
    return (bitvector[indx / 32] >> ((indx % 32) * 2)) & 0b11;
}
//extracts indx u5 within sector
//> least significant end | <40 bits of packed u5s><12 bits PSS offset><12 bits NSS offset> | most significant end
inline u32 gpu_get_lcs_u5(const u64 sector, i32 indx){
    return (sector >> (indx * 5)) & 0b11111;
}
} // namespace sbwt_lcs_gpu
