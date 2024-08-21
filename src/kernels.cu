#include "gpu/gpu_utils.cuh"
#include "gpu/kernels.h"
namespace sbwt_lcs_gpu {
__global__ void shifted_array(u64 *array, u64 *out, u64 max) {
    u64 idx = threadIdx.x + blockIdx.x * blockDim.x;
    u64 curr = array[idx];
    u64 by = 29 * 47 * 97 * 1399;
    u64 times = 30;
    for (u64 i = 0; i < times; ++i) {
        curr = (array[curr] * by) % max;
    }
    out[idx] = curr;
}

// void launch_shifted_array(u64 *array, u64 *out, u64 size, GpuStream &stream, GpuEvent &stop_event) {
void launch_shifted_array(GpuPointer<u64> &array, GpuPointer<u64> &out, u64 size, GpuEvent &stop_event) {

    // shifted_array<<<size/1024, 1024>>>(array, out, size);
    // shifted_array<<<size/1024, 1024, 0, *getStream(stream)>>>(array, out, size);
    shifted_array<<<size / 1024, 1024>>>(array.data(), out.data(), size);
    CUDA_CHECK(cudaGetLastError());
    // stream.sync();
    stop_event.record();
    stop_event.sync();
}
void print_supported_devices() {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    std::cout << "device count: " << device_count << std::endl;
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        std::cout << "device: " << i << " name: " << prop.name << std::endl;
    }
}

#if 0
template <bool move_to_key_kmer>
__global__ void
d_search(const u32 kmer_size, const u64 *const c_map, const u64 *const *const acgt, const u64 *const *const layer_0,
         const u64 *const *const layer_1_2, const u64 *const presearch_left, const u64 *const presearch_right,
         const u64 *const kmer_positions, const u64 *const bit_seqs, const u64 *const key_kmer_marks, u64 *out) {
    const u32 idx = get_idx();
    const u64 kmer_index = kmer_positions[idx] * 2;
    const u64 first_part = (bit_seqs[kmer_index / 64] << (kmer_index % 64));
    const u64 second_part = (bit_seqs[kmer_index / 64 + 1] >> (64 - (kmer_index % 64))) &
                            static_cast<u64>(-static_cast<u64>((kmer_index % 64) != 0));
    const u64 kmer = first_part | second_part;
    constexpr const u64 presearch_mask = (2ULL << (presearch_letters * 2)) - 1;
    const u32 presearched = (kmer >> (64 - presearch_letters * 2)) & presearch_mask;
    u64 node_left = presearch_left[presearched];
    u64 node_right = presearch_right[presearched];
    for (u64 i = kmer_positions[idx] + presearch_letters; i < kmer_positions[idx] + kmer_size; ++i) {
        const u32 c = (bit_seqs[i / 32] >> (62 - (i % 32) * 2)) & two_1s;
        node_left = c_map[c] + d_rank(acgt[c], layer_0[c], layer_1_2[c], node_left);
        node_right = c_map[c] + d_rank(acgt[c], layer_0[c], layer_1_2[c], node_right + 1) - 1;
    }
    if (node_left > node_right) {
        out[idx] = -1ULL;
        return;
    }
    if (move_to_key_kmer) {
        while (!d_get_bool_from_bit_vector(key_kmer_marks, node_left)) {
            for (u32 i = 0; i < 4; ++i) {
                if (d_get_bool_from_bit_vector(acgt[i], node_left)) {
                    node_left = c_map[i] + d_rank(acgt[i], layer_0[i], layer_1_2[i], node_left);
                    break;
                }
            }
        }
    }
    out[idx] = node_left;
}

inline __device__ auto d_rank(const u64 *bit_vector, const u64 *layer_0, const u64 *layer_1_2, const u64 index) -> u64 {
    const u64 basicblocks_in_superblock = 4;
    const u64 basicblock_bits = superblock_bits / basicblocks_in_superblock;
    u64 entry_basicblock = 0;
    const auto *bit_vector_128b = reinterpret_cast<const ulonglong2 *>(bit_vector);
    const u64 target_shift = 64U - (index % 64U);
    const u64 ints_in_basicblock = basicblock_bits / 64;
    const u64 in_basicblock_index = (index / 64) % ints_in_basicblock;
#pragma unroll // calculating entry_basicblock 2 ints at a time
    for (u64 i = 0; i < ints_in_basicblock; i += 2) {
        ulonglong2 data_128b = bit_vector_128b[(index / 128) - ((index / 128) % (ints_in_basicblock / 2)) + i / 2];
        entry_basicblock += __popcll((data_128b.x << (((i + 0) == in_basicblock_index) * target_shift)) &
                                     -((((i + 0) == in_basicblock_index) * target_shift) < 64)) *
                                ((i + 0) <= in_basicblock_index) +
                            __popcll((data_128b.y << (((i + 1) == in_basicblock_index) * target_shift)) &
                                     -((((i + 1) == in_basicblock_index) * target_shift) < 64)) *
                                ((i + 1) <= in_basicblock_index);
    }
    const u64 entry_layer_1_2 = layer_1_2[index / superblock_bits];
    u64 entry_layer_2_joined =
        (entry_layer_1_2 & thirty_1s) >> (10 * (3U - ((index % superblock_bits) / basicblock_bits)));
    const u64 entry_layer_2 = ((entry_layer_2_joined >> 20)) + ((entry_layer_2_joined >> 10) & ten_1s) +
                              ((entry_layer_2_joined >> 00) & ten_1s);
    const u64 entry_layer_1 = entry_layer_1_2 >> 32;
    const u64 entry_layer_0 = layer_0[index / hyperblock_bits];
    return entry_basicblock + entry_layer_2 + entry_layer_1 + entry_layer_0;
}
inline __device__ auto d_rank_simple(const u64 *bit_vector, const u64 *layer_1_2, const u64 index) -> u64 {
    const u64 basicblocks_in_superblock = 4;
    const u64 basicblock_bits = superblock_bits / basicblocks_in_superblock;
    u64 entry_basicblock = 0;
    const auto *bit_vector_128b = reinterpret_cast<const ulonglong2 *>(bit_vector);
    const u64 target_shift = 64U - (index % 64U);
    const u64 ints_in_basicblock = basicblock_bits / 64;
    const u64 in_basicblock_index = (index / 64) % ints_in_basicblock;
#pragma unroll // calculating entry_basicblock 2 ints at a time
    for (u64 i = 0; i < ints_in_basicblock; i += 2) {
        ulonglong2 data_128b = bit_vector_128b[(index / 128) - ((index / 128) % (ints_in_basicblock / 2)) + i / 2];
        entry_basicblock += __popcll((data_128b.x << (((i + 0) == in_basicblock_index) * target_shift)) &
                                     -((((i + 0) == in_basicblock_index) * target_shift) < 64)) *
                                ((i + 0) <= in_basicblock_index) +
                            __popcll((data_128b.y << (((i + 1) == in_basicblock_index) * target_shift)) &
                                     -((((i + 1) == in_basicblock_index) * target_shift) < 64)) *
                                ((i + 1) <= in_basicblock_index);
    }
    const u64 entry_layer_1_2 = layer_1_2[index / superblock_bits];
    u64 entry_layer_2_joined =
        (entry_layer_1_2 & thirty_1s) >> (10 * (3U - ((index % superblock_bits) / basicblock_bits)));
    const u64 entry_layer_2 = ((entry_layer_2_joined >> 20)) + ((entry_layer_2_joined >> 10) & ten_1s) +
                              ((entry_layer_2_joined >> 00) & ten_1s);
    const u64 entry_layer_1 = entry_layer_1_2 >> 32;
    return entry_basicblock + entry_layer_2 + entry_layer_1;
}

inline __device__ auto d_get_bool_from_bit_vector(const u64 *bitmap, u64 index) -> bool {
    auto elem_idx = index / u64_bits;
    auto elem_offset = index % u64_bits;
    return (bitmap[elem_idx] & (1ULL << elem_offset)) > 0;
}

#endif

} // namespace sbwt_lcs_gpu