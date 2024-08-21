#pragma once
// clang-format off
#include <cstdint>
#include <vector>
#include <string>
#include <iostream>
#include <utility>
#include <tuple>
#include <functional>
#include <fstream>
#include <sstream>
#include <cmath>
#include <type_traits>
#include <cstdio>

namespace sbwt_lcs_gpu {
using std::min;
using std::max;
using std::ceil;
using i64 = int64_t;
using u64 = uint64_t;
using u32 = uint32_t;
using i32 = int32_t;
using u8 = uint8_t;
using i8 = int8_t;


inline i64 mod(i64 a, i64 b) {
    return (a % b + b) % b;
}
inline i64 ceil_div(i64 a, i64 b) {
    return (a + b - 1) / b;
}
inline i64 ceil_mult_of(i64 a, i64 b) {
    return ceil_div(a, b) * b;
}
constexpr i64 ceil_div_const(i64 a, i64 b) {
    return (a + b - 1) / b;
}
constexpr i64 ceil_mult_of_const(i64 a, i64 b) {
    return ceil_div_const(a, b) * b;
}
inline i64 ceil_double_div(double a, double b) {
    return std::ceil(a / b);
}
constexpr double ceil_const(double x) {
    i64 x_int = static_cast<i64>(x);
    return (x > x_int) ? x_int + 1 : x_int;
}
constexpr i64 ceil_double_div_const(double a, double b) {
    return ceil_const(a / b);
}



static_assert(sizeof(std::pair<i32, i32>)==2*sizeof(i32), "std::pair<i32, i32> is not packed");
static_assert(sizeof(std::pair<i32, i32>)==2*sizeof(i32), "std::pair<i32, i32> is not packed");


const u64 u64_bits = 64;
const u64 bits_in_byte = 8;

enum class alphabet { A = 0, C = 1, G = 2, T = 3 };
const u64 cmap_size = 5;
const u64 alphabet_size = 4;
const u64 bits_per_char = 2;
const u64 chars_per_u64 = u64_bits / bits_per_char;//32, //!code does not deal with u64_bits%bits_per_char!=0

const u64 bitvector_pad_u64s = 4;//gpu rank scans in batches of 4, so to avoid bounds checking, pad by 4
const u64 bitvector_pad_bits = u64_bits*4;

const u64 presearch_letters = 12; //do not presearch with left contractions
const u64 threads_per_block = 1024;
const u64 gpu_warp_size = 32;//NVIDIA, 64 for AMD


const u64 max_files_in_stream = 100;
//desired core data batch size (e.g. chars or 2 bit chars)
const u64 batch_buf_bytes = (1<<20);//1MB
const u64 batch_buf_u64s = ceil_div_const(batch_buf_bytes, sizeof(u64));
// static_assert((batch_buf_size % sizeof(u64) == 0), "batch_buf_size must be a multiple of sizeof(u64)");

const u64 max_read_chars = 100;
const u64 batches_in_stream = 16;
const u64 max_num_physical_streams = 10;
//how many normal sized batches are processed in the gpu at the same time
const u64 gpu_batch_mult = 16;
//these are gpu_mult sized batches
const u64 batches_in_gpu_stream = 8;
//these are divided by gpu_mult sized compared to gpu batches
const u64 batches_in_out_buf_stream = 32;
const u64 gpu_readers = 4;
constexpr double out_compression_ratio = 1.8;
extern u64 num_physical_streams;
extern i32 total_threads;
// const std::vector<u64> nothing_vector;//basically nullptr for offsetvector

// //=File buf section
// const u64 F_section_stream_u64s = batch_buf_u64s * batches_in_stream;
// u64 F_section_u64s;


// //=Parse section
// //forward declare
// inline u64 poppysmall_from_bitvector_u64s(u64 num_u64s);
// //section dedicated to 2 bit chars + section for seps + bit vector

// const u64 P_section_2bit_batch_u64s=batch_buf_u64s;

// const u64 P_section_seps_batch_u64s=ceil_div(batch_buf_u64s*u64_bits/bits_per_char,//num chars in batch
//         max_read_chars)+2;//max seps in batch + some padding

// const u64 P_section_seps_bitvector_batch_u64s=ceil_div(P_section_seps_batch_u64s, u64_bits)+4;//+4 for padding
// const u64 P_section_seps_rank_batch_u64s=poppysmall_from_bitvector_u64s(P_section_seps_bitvector_batch_u64s);
// const u64 P_section_batch_u64s = (P_section_2bit_batch_u64s
//     +P_section_seps_batch_u64s
//     +P_section_seps_bitvector_batch_u64s
//     +P_section_seps_rank_batch_u64s);
// u64 P_section_u64s;
// //=Multiplex section
// const u64 M_section_2bit_batch_section_u64s

// inline void set_section_sizes(u64 num_streams){
//     F_section_u64s = F_section_stream_u64s * num_streams;
//     P_section_u64s = P_section_batch_u64s * num_streams;
// }


// const u64 batch_size_reads = ceil_div(batch_buf_size,max_read_chars)/bits_per_char*bits_in_byte;//ignoring extra datastructures

const u64 desired_stream_size = (1<<20) * 20;//20MB



//todo: c++20 add conditional copy/move constructors based on T
template <typename T> 
class FCVector { // Fixed Capacity Vector
  public:
    using value_type = T;

    explicit FCVector(u64 size)
        : size_(size), count_(0), data_(static_cast<T *>(::operator new(size * sizeof(T)))) {}
    template <typename... Args>
    explicit FCVector(u64 size, Args&&... args)
        : size_(size), count_(0), data_(static_cast<T *>(::operator new(size * sizeof(T)))) {
        for (u64 i = 0; i < size; ++i) {
            emplace_back(std::forward<Args>(args)...);
        }
    }

    ~FCVector() {
        clear();
        ::operator delete(data_);
    }

    template <typename... Args>
    inline void emplace_back(Args &&...args) {
        if (count_ >= size_) {
            throw std::out_of_range("FCVector capacity exceeded: "+std::to_string(count_)+" >= "+std::to_string(size_));
        }
        new (&data_[count_]) T(std::forward<Args>(args)...);
        ++count_;
    }

    // inline T &operator[](i64 index) {
    //     if (index >= count_ || index < 0) {
    inline T &operator[](u64 index) {
        if (index >= count_) {//automatically checks if index is negative
            throw std::out_of_range("FCV Index [] out of range: "+std::to_string(index)+" >= "+std::to_string(count_));
        }
        return data_[index];
    }

    inline const T &operator[](u64 index) const {
        if (index >= count_) {//automatically checks if index is negative
            throw std::out_of_range("FCV Index [] out of range: "+std::to_string(index)+" >= "+std::to_string(count_));
        }
        return data_[index];
    }

    inline u64 size() const { return count_; }

    inline u64 capacity() const { return size_; }
    inline T &back() {
        if (count_ == 0) {
            throw std::out_of_range("FCVector: back() called on empty vector");
        }
        return data_[count_ - 1];
    }
    inline void pop_back() {
        if (count_ == 0) {
            throw std::out_of_range("FCVector: pop_back() called on empty vector");
        }
        data_[count_ - 1].~T();
        --count_;
    }

    inline void clear() {
        for (u64 i = 0; i < count_; ++i) {
            data_[i].~T();
        }
        count_ = 0;
    }
    inline T *data() { return data_; }
    inline const T *data() const { return data_; }

  private:
    u64 size_;
    u64 count_;
    T *data_;
};

template <typename V_type>
class ContinuousVector: public V_type {
    public:
    ContinuousVector(u64 size): V_type(size) {}
    inline void remove(u64 index) {
        if (index >= this->size()) {
            throw std::out_of_range("ContinuousVector: Remove out of range: "+std::to_string(index)+" >= "+std::to_string(this->size()));
        }
        if (index!=this->size()-1) {
            std::swap((*this)[index], this->back());
        }
        this->pop_back();
    }
};


//non-owning vector
template <typename T> class OffsetVector { // Fixed Capacity Vector
    static_assert(std::is_trivially_copyable_v<T>,"OffsetVector T must be trivially copyable");
    static_assert(std::is_standard_layout_v<T>, "OffsetVector T must be standard layout");
  private:
    T* ptr_;
    u64 max_cap_;
    u64 size_;
    const u64 in_v_off_;
  public:

    using value_type = T;


    // explicit OffsetVector(u64 cap, u64 off, V_type &v) : max_cap_(cap), offset_(off), size_(0), vec_(v) {
    //     if (offset_ >= vec_.size()) {
    //         throw std::out_of_range("OffsetVector: offset out of range");
    //     }
    //     if (offset_ + max_cap_ > vec_.size()){
    //         throw std::out_of_range("OffsetVector: offset + max_cap_ out of range");
    //     }
    // }
    template <typename V_type>
    constexpr void assert_compatible(){
        // static_assert(sizeof(T)%sizeof(V_type::value_type)==0,"OffsetVector size must be multiple of V_type::value_type for safe reinterpret_cast");
        static_assert(alignof(typename V_type::value_type)%alignof(T)==0,"OffsetVector must align with V_type::value_type");
        static_assert(std::is_trivially_copyable_v<typename V_type::value_type>,"OffsetVector V_type::value_type must be trivially copyable");
        static_assert(std::is_standard_layout_v<typename V_type::value_type>, "OffsetVector V_type::value_type must be standard layout");
    }
    template <typename V_type>
    explicit OffsetVector(V_type &v) :
    max_cap_(v.size()*sizeof(typename V_type::value_type)/sizeof(T)),
    size_(0),
    ptr_(reinterpret_cast<T*>(v.data())),
    in_v_off_(0) {
        assert_compatible<V_type>();
    }

    template <typename V_type>
    explicit OffsetVector(V_type &in_v, u64 in_v_off, u64 in_v_cap) :
    max_cap_(in_v_cap*sizeof(typename V_type::value_type)/sizeof(T)),
    size_(0),
    ptr_(reinterpret_cast<T*>(in_v.data()+in_v_off)),
    in_v_off_(in_v_off) {
        assert_compatible<V_type>();
        if (in_v_off + max_cap_ > in_v.size()){
            throw std::out_of_range("OffsetVector: offset + max_cap_ out of range: "+std::to_string(in_v_off)+" + "+std::to_string(max_cap_)+" > "+std::to_string(in_v.size()));
        }
    }
    // explicit OffsetVector(u64 cap, T* ptr) : max_cap_(cap), size_(0), ptr_(ptr) {
    // }
    OffsetVector(OffsetVector&& other) : 
    max_cap_(other.max_cap_), size_(other.size_), ptr_(other.ptr_), in_v_off_(other.in_v_off_) {
        other.size_ = 0;
        other.max_cap_ = 0;
    } 
    inline void resize(u64 new_size) {
        if (new_size > max_cap_) {
            throw std::out_of_range("OffsetVector resize capacity exceeded: "+std::to_string(new_size)+" > "+std::to_string(max_cap_));
        }
        size_ = new_size;
    }
    inline void reserve(u64 new_cap) {
        if (new_cap > max_cap_) {
            throw std::out_of_range("OffsetVector reserve capacity exceeded: "+std::to_string(new_cap)+" > "+std::to_string(max_cap_));
        }
    }
    inline T &operator[](u64 index) {
        if (index >= size_) {
            throw std::out_of_range("OV Index [] out of range: "+std::to_string(index)+" >= "+std::to_string(size_));
        }
        // return vec_[index + offset_];
        return ptr_[index];
    }

    inline const T &operator[](u64 index) const {
        if (index >= size_) { // automatically checks if index is negative
            throw std::out_of_range("OV Index [] out of range: "+std::to_string(index)+" >= "+std::to_string(size_));
        }
        // return vec_[index + offset_];
        return ptr_[index];
    }

    inline u64 size() const { return size_; }

    inline u64 capacity() const { return max_cap_; }
    inline u64 offset() const { return in_v_off_; }
    inline u64 in_v_offset() const { return in_v_off_; }
    inline T &back() {
        if (size_ == 0) {
            throw std::out_of_range("OffsetVector: back() called on empty vector");
        }
        return ptr_[size_ - 1];
    }
    inline void pop_back() {
        if (size_ == 0) {
            throw std::out_of_range("OffsetVector: pop_back() called on empty vector");
        }
        // vec[count_ - 1].~T();
        --size_;
    }

    inline void clear() {
        // for (u64 i = 0; i < count_; ++i) {
        //     vec[i].~T();
        // }
        size_ = 0;
    }
    inline void push_back(const T &value) {
        if (size_ >= max_cap_) {
            throw std::out_of_range("OffsetVector push_back capacity exceeded: "+std::to_string(size_)+" >= "+std::to_string(max_cap_));
        }
        ptr_[size_] = value;
        ++size_;
    }
    inline void push_back(T &&value) {
        if (size_ >= max_cap_) {
            throw std::out_of_range("OffsetVector push_back capacity exceeded: "+std::to_string(size_)+" >= "+std::to_string(max_cap_));
        }
        ptr_[size_] = std::move(value);
        ++size_;
    }
    inline T *data() { return ptr_; }
    inline const T *data() const { return ptr_; }
};

template <typename V>
inline i32 find_map(const V& map, i64 id){
    for(i32 i=0; i<map.size(); i++){
        if(map[i].first==id){
            return map[i].second;
        }
    }
    return -1;
}
template <typename V, typename Any_Func>
inline bool any_of(const V& vec, Any_Func func){
    for(i32 i=0; i<vec.size(); i++){
        if(func(vec, i)){
            return true;
        }
    }
    return false;
}
template <typename V, typename Any_Func>
inline i32 first_of(const V& vec, Any_Func func){
    for(i32 i=0; i<vec.size(); i++){
        if(func(vec, i)){
            return i;
        }
    }
    return -1;
}
template <typename V, typename All_Func>
inline bool all_of(const V& vec, All_Func func){
    for(i32 i=0; i<vec.size(); i++){
        if(!func(vec, i)){
            return false;
        }
    }
    return true;
}

template <typename V, typename T>
inline void fill_vector(V& vec, T val){
    for(i32 i=0; i<vec.size(); i++){
        vec[i]=val;
    }
}

template<typename T, typename... Args>
void prints(std::stringstream &ss, T arg, Args&&... args) {
    ss<<arg;
    ((ss << " " << args), ...); // C++17 fold expression to print all arguments with a space in between
}
template<typename T, typename... Args>
std::string prints_new(T arg, Args&&... args) {
    std::stringstream ss;
    ss<<arg;
    ((ss << " " << args), ...); // C++17 fold expression to print all arguments with a space in between
    return ss.str();
}
template <typename V>
std::string print_vec_new(V& vec){
    std::stringstream ss;
    ss<<"[";
    for(i32 i=0; i<vec.size(); i++){
        ss<<vec[i];
        if(i!=vec.size()-1){
            ss<<",";
        }
    }
    ss<<"]";
    return ss.str();
}


struct CompareFirst{
    inline bool operator()(std::pair<i64,i32> const& a, std::pair<i64,i32> const& b) const noexcept{
        return a.first>b.first;
    }
};


class ThrowingOfstream : public std::ofstream {
  public:

    static bool check_path_valid(const std::string &filepath) {
        // ThrowingOfstream(filepath, std::ios::out);
        std::ofstream file(filepath);
        return file.good();
    }


    ThrowingOfstream(const std::string &filepath, std::ios::openmode mode)
        : std::ofstream(filepath, mode) {
        if (this->fail()) {
            throw std::ios::failure(
                prints_new("The path", filepath,
                           "cannot be opened. Check that all the folders in the path is "
                           "correct and that you have permission to create files in this path folder"));
        }
    }

    void write_line(const std::string &s) {
        (*this) << s << '\n';
    }
    using std::ofstream::write;
    template <typename T> 
    void write(T t) {
        write(reinterpret_cast<char *>(&t), sizeof(t));
    }

    template <typename V_type>
    void write_v(const V_type &v) {
        write(reinterpret_cast<const char *>(v.data()), v.size() * sizeof(typename V_type::value_type));
    }


    void write_string_with_size(const std::string &s) {
        u64 size = reinterpret_cast<u64>(s.size());
        std::ofstream::write(reinterpret_cast<char *>(&size), sizeof(u64));
        (*this) << s;
    }
};

class ThrowingIfstream : public std::ifstream {
  public:
    ThrowingIfstream(const std::string &filename, std::ios::openmode mode)
        : std::ifstream(filename, mode) {
        if (this->fail()) {
            throw std::ios::failure(prints_new("The input file", filename, "cannot be opened"));
        }
    }

    static bool check_file_exists(const std::string &filename) {
        // ThrowingIfstream(filename, std::ios::in);
        std::ifstream file(filename);
        return file.good();
    }

    //bool for not end of file
    bool read_line(std::string &s) {
        std::getline(*this, s);
        return !(this->eof());
    }

    using std::ifstream::read;
    std::string read_string_with_size() {
        u64 size = 0;
        read(reinterpret_cast<char *>(&size), sizeof(u64));
        std::string s;
        s.resize(size);
        read(reinterpret_cast<char *>(s.data()), static_cast<std::streamsize>(size));
        return s;
    }

    template <typename T> 
    T read() {
        T t = 0;
        read(reinterpret_cast<char *>(&t), sizeof(T));
        return t;
    }

};

inline void skip_bits_vector(std::istream &stream) {
    u64 bits = 0;
    stream.read(reinterpret_cast<char *>(&bits), sizeof(u64));
    // u64 bytes = round_up<u64>(bits, u64_bits) / sizeof(u64);
    u64 bytes = ceil_div(bits, u64_bits)*sizeof(u64);//based on how sdsl stores bitvector
    stream.seekg(static_cast<std::ios::off_type>(bytes), std::ios::cur);
}

inline void skip_bytes_vector(std::istream &stream) {
    u64 bytes = 0;
    stream.read(reinterpret_cast<char *>(&bytes), sizeof(u64));
    stream.seekg(static_cast<std::ios::off_type>(bytes), std::ios::cur);
}


template <typename ptr>
ptr quicksort_partition(ptr begin, ptr end) {
    ptr pivot = end - 1;
    ptr i = begin - 1;
    for (ptr j = begin; j < pivot; ++j) {
        if (*j < *pivot) {
            ++i;
            swap(*i, *j);
        }
    }
    swap(*(i + 1), *pivot);
    return (i + 1);
}

template <typename ptr>
void quickSort(ptr begin, ptr end) {
    if (begin < end - 1) {
        ptr pivot = quicksort_partition(begin, end);
        quickSort(begin, pivot);
        quickSort(pivot + 1, end);
    }
}
template <typename ptr>
void reverse(ptr begin, ptr end) {
    while (begin < end) {
        --end;
        if (begin < end) {
            swap(*begin, *end);
            ++begin;
        }
    }
}

u64 get_total_cpu_memory();
u64 get_free_cpu_memory();


} // namespace sbwt_lcs_gpu
// clang-format on
