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
//only works with kmer taking up=5 bits
__global__ void sep_search_simple(
    const u32 kmer_size,//size of the kmer
    const u64 *const c_map,//has the start of each ACGT segment in the SBWT, in the thesis 0=$ but here 0=A
    const u64 *const *const acgt,//points to bitvectors of ACGT
    const u64 *const *const layer_0,//points to layer 0 of poppy for each ACGT bitvector
    const u64 *const *const layer_1_2,//points to layer 1 and 2 of poppy for each ACGT bitvector
    const u64 *const lcs,//points to the lcs array interleaved with PSS/NSS answers
    const u64 lcs_size,//size of the lcs array, u5-wise
    const u32 *const seps,//points to the separator values
    const u64 *const is_true_sep,//points to the bitvector marking true separators
    const u64 *const *const is_true_sep_layer_1_2,//points to layer 1 and 2 of poppy for the is_true_sep bitvector
    const u64 *const chars,//points to the bitvector of the characters
    u64 *out){//output array
    const u32 idx = get_idx();
    const u32 start=seps[idx];//start in the bitvector of characters
    // sep range length in the bitvector of characters
    const u32 out_range_len=seps[idx+1]-start-(kmer_size-1)*gpu_get_bitvector_bit(is_true_sep,idx);
    //true separators have a (kmer_size-1)-length buffer of characters until the next output character
    //so, need to do rank query on the is_true_sep bitvector to get the actual start in the output
    const u32 out_start=start-d_rank_simple(is_true_sep,is_true_sep_layer_1_2,idx)*(kmer_size-1);
    u64 node_left=0;
    u64 node_right=c_map[4]-1;
    u64 new_node_left, new_node_right;//set within the loop
    u32 matched_len=0;
    //fill with -1
    for(u32 i=0;i<out_range_len+kmer_size-1;++i){
        out[out_start+i]=-1ULL;
    }
    for(u32 i=0;i<char_range_len;++i){
        //get current character
        const u32 c=gpu_get_two_bit_char(chars,start+i);
        const u64 off=c_map[c];
        new_node_left = off + d_rank(acgt[c], layer_0[c], layer_1_2[c], node_left);
        new_node_right = off + d_rank(acgt[c], layer_0[c], layer_1_2[c], node_right + 1) - 1;
        while(matched_len>0 && new_node_left>new_node_right){//left contraction
            matched_len--;
            u32 val=gpu_get_lcs_u5(lcs,node_left);
            if(val==matched_len){
                node_left=gpu_lcs_PSV(lcs,lcs_size,node_left,val);
            }
            if(node_right+1<lcs_size){
                val=gpu_get_lcs_u5(lcs,node_right+1);
                if(val==matched_len){
                    node_right=gpu_lcs_NSV(lcs,lcs_size,node_right+1,val)-1;
                }
            }
            new_node_left = off + d_rank(acgt[c], layer_0[c], layer_1_2[c], node_left);
            new_node_right = off + d_rank(acgt[c], layer_0[c], layer_1_2[c], node_right + 1) - 1;
        }
        if(new_node_left<=new_node_right){
            node_left=new_node_left;
            node_right=new_node_right;
            matched_len=min(matched_len+1,kmer_size);
            if(matched_len==kmer_size){
                out[out_start+i-kmer_size+1]=node_left;
            }
        }
    }
}


//! only defined for kmer_size=5
//> uses Previous/Next Smaller Sector offsets to minimize number of jumps to answer NSV/PSV query
//> least significant end | <40 bits of packed u5s><12 bits PSS offset><12 bits NSS offset> | most significant end
//> offset = <u5 exponent><u7 mantissa>, value=2^exp*mantissa
__global__ u64 gpu_lcs_PSV(
    // const u32 kmer_size,
    const u64 *const lcs,
    const i64 lcs_size,
    i64 index,
    const u32 val){//extracted value at index using gpu_get_lcs_u5
        u64 curr_sector = lcs[index/8];
        // const u32 val=gpu_get_lcs_u5(curr_sector,index%8);
        //scan backward/forward to check for immediate answer
        for(i32 i=(index%8)-1;i>=0;--i){
            if(gpu_get_lcs_u5(curr_sector,i)<val){
                return index-(index%8)+i;
            }
        }
        //scan forward/backward to check if this is one of the minimum values
        bool is_min=true;
        for(i32 i=(index%8)+1;i<min(lcs_size-(lcs_size%8),8);++i){
            if(gpu_get_lcs_u5(curr_sector,i)<val){
                is_min=false;
            }
        }
        if(!is_min){
            index-=(index%8)+1;//first entry to the left of the current sector
        }else{
            //extract PSV offset and traverse
            const u32 psv_offset=(curr_sector>>40)&0xFFF;
            const u32 psv_exp=psv_offset&0b11111;
            const u32 psv_mantissa=(psv_offset>>5)&0b1111111;
            //pointer starts from first entry to the left of the current sector
            index-=(index%8)+1+((1<<psv_exp)*psv_mantissa);
        }
        while(true){
            if(index<=0){
                return 0;//for NSV should be lcs_size
            }
            curr_sector=lcs[index/8];
            //scan backward/forward to check for answer
            for(i32 i=index%8;i>=0;--i){//not -1 because inclusive
                if(gpu_get_lcs_u5(curr_sector,i)<val){
                    return index-(index%8)+i;
                }
            }
            //traverse psv again
            const u32 psv_offset=(curr_sector>>40)&0xFFF;
            const u32 psv_exp=psv_offset&0b11111;
            const u32 psv_mantissa=(psv_offset>>5)&0b1111111;
            //pointer starts from first entry to the left of the current sector
            index-=(index%8)+1+((1<<psv_exp)*psv_mantissa);
        }
    }
__global__ u64 gpu_lcs_NSV(
    // const u32 kmer_size,
    const u64 *const lcs,
    const i64 lcs_size,
    i64 index,
    const u32 val){//extracted value at index using gpu_get_lcs_u5
        u64 curr_sector = lcs[index/8];
        // const u32 val=gpu_get_lcs_u5(curr_sector,index%8);
        //scan backward/forward to check for immediate answer
        // for(i32 i=(index%8)-1;i>=0;--i){
        for(i32 i=(index%8)+1;i<min(lcs_size-(lcs_size%8),8);++i){
            if(gpu_get_lcs_u5(curr_sector,i)<val){
                return index-(index%8)+i;
            }
        }
        //scan forward/backward to check if this is one of the minimum values
        bool is_min=true;
        for(i32 i=(index%8)-1;i>=0;++i){
            if(gpu_get_lcs_u5(curr_sector,i)<val){
                is_min=false;
            }
        }
        if(!is_min){
            index+=8-index%8;//first entry to the right of the current sector
        }else{
            //extract NSV offset and traverse
            const u32 nsv_offset=(curr_sector>>(40+12))&0xFFF;
            const u32 nsv_exp=nsv_offset&0b11111;
            const u32 nsv_mantissa=(nsv_offset>>5)&0b1111111;
            //pointer starts from first entry to the right of the current sector
            index+=8-(index%8)+((1<<nsv_exp)*nsv_mantissa);
        }
        while(true){
            if(index>=lcs_size){
                return lcs_size;
            }
            curr_sector=lcs[index/8];
            //scan backward/forward to check for answer
            for(i32 i=(index%8);i<min(lcs_size-(lcs_size%8),8);++i){
                if(gpu_get_lcs_u5(curr_sector,i)<val){
                    return index-(index%8)+i;
                }
            }
            //traverse nsv again
            const u32 nsv_offset=(curr_sector>>(40+12))&0xFFF;
            const u32 nsv_exp=nsv_offset&0b11111;
            const u32 nsv_mantissa=(nsv_offset>>5)&0b1111111;
            //pointer starts from first entry to the right of the current sector
            index+=8-(index%8)+((1<<nsv_exp)*nsv_mantissa);
        }
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