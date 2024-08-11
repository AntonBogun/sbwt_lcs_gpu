#pragma once
// clang-format off
#include "utils.h"
#include <sdsl/int_vector.hpp>
#include <sdsl/rank_support.hpp>
namespace sbwt_lcs_gpu {

const u64 layer_1_bits = 32;
const u64 layer_2_bits = 10;
const u64 hyperblock_bits = 1ULL << 32ULL;
const u64 superblock_bits = 1024;
const u64 basicblock_bits = 256;
class Poppy {
public:
  std::vector<u64> layer_0;
  std::vector<u64> layer_1_2;
  u64 total_1s = static_cast<u64>(-1);
    void build(const std::vector<u64> &bits_vector, u64 num_bits) {
        u64 layer_0_count = 0, layer_1_count = 0;//layer_2_count = 0;
        // std::vector<u64> layer_2_temps = std::vector<u64>(3, 0);
        // u64 layer_2_temps_index = 0;
        u64 layer2_0=0,layer2_1=0,layer2_2=0,layer2_3=0;

        layer_0.reserve(ceil_div(num_bits, hyperblock_bits));
        layer_1_2.reserve(ceil_div(num_bits, superblock_bits));

        i64 num_bits_rounded = ceil_div(num_bits, u64_bits)*u64_bits;
        for (i64 i = 0, bits = 0; bits < num_bits_rounded; bits += superblock_bits, i+=superblock_bits/u64_bits) {
            if (bits % hyperblock_bits == 0) {
                layer_0_count += layer_1_count;
                layer_0.push_back(layer_0_count);
                layer_1_count = 0;
            }
            // layer2_0 = block_popcount(bits_vector, i);
            // layer2_1 = layer2_0+block_popcount(bits_vector, i+basicblock_bits/u64_bits);
            // layer2_2 = layer2_1+block_popcount(bits_vector, i+2*basicblock_bits/u64_bits);
            // layer2_3 = layer2_2+block_popcount(bits_vector, i+3*basicblock_bits/u64_bits);
            layer2_0 = block_popcount(bits_vector, i);
            layer2_1 = block_popcount(bits_vector, i+basicblock_bits/u64_bits);
            layer2_2 = block_popcount(bits_vector, i+2*basicblock_bits/u64_bits);
            layer2_3 = block_popcount(bits_vector, i+3*basicblock_bits/u64_bits);
            layer_1_2.push_back(
                layer_1_count << layer_1_bits
                | layer2_0 << (layer_2_bits * 2)
                | layer2_1 << (layer_2_bits * 1)
                | layer2_2 << (layer_2_bits * 0)
            );
            // layer_1_count += layer2_3;
            layer_1_count += layer2_0+layer2_1+layer2_2+layer2_3;

            // layer2_0=0;layer2_1=0;layer2_2=0;layer2_3=0;
        }
        total_1s = layer_0_count+layer_1_count;
    }
    i32 inline block_popcount(const std::vector<u64> &bits_vector, u64 start) const {
        i32 total = 0;
        #pragma unroll
        for (i32 i = 0; i < basicblock_bits/u64_bits && start + i < bits_vector.size(); i++) {
            total += __builtin_popcountll(bits_vector[start + i]);
        }
        return total;
    }
};

//for u32_MAX max sized bitvector
template <typename V_type_in=std::vector<u64>, typename V_type_layer=std::vector<u64>>
class PoppySmall {
public:
    V_type_layer layer_1_2;
    u32 total_1s = static_cast<u32>(-1);

    void build(const V_type_in &bits_vector, u32 num_bits) {
        u64 layer_1_count = 0;
        u64 layer2_0 = 0, layer2_1 = 0, layer2_2 = 0, layer2_3 = 0;

        layer_1_2.reserve(ceil_div(num_bits, superblock_bits));

        u32 num_bits_rounded = ceil_div(num_bits, u64_bits) * u64_bits;
        for (u32 i = 0, bits = 0; bits < num_bits_rounded; bits += superblock_bits, i += superblock_bits / u64_bits) {
            layer2_0 = block_popcount(bits_vector, i);
            layer2_1 = block_popcount(bits_vector, i + basicblock_bits / u64_bits);
            layer2_2 = block_popcount(bits_vector, i + 2 * basicblock_bits / u64_bits);
            layer2_3 = block_popcount(bits_vector, i + 3 * basicblock_bits / u64_bits);

            layer_1_2.push_back(
                layer_1_count << layer_1_bits
                | layer2_0 << (layer_2_bits * 2)
                | layer2_1 << (layer_2_bits * 1)
                | layer2_2 << (layer_2_bits * 0)
            );

            layer_1_count += layer2_0 + layer2_1 + layer2_2 + layer2_3;

        }
        total_1s = layer_1_count;
    }

    u32 inline block_popcount(const V_type_in &bits_vector, u32 start) const {
        u32 total = 0;
        #pragma unroll
        for (u32 i = 0; i < basicblock_bits / u64_bits && start + i < bits_vector.size(); i++) {
            total += __builtin_popcountll(bits_vector[start + i]);
        }
        return total;
    }
};

} // namespace sbwt_lcs_gpu