#pragma once
#include "utils.h"
namespace sbwt_lcs_gpu {
struct StreamFilenamesContainer{
    std::vector<std::string> filenames;
    std::vector<std::string> output_filenames;
    std::vector<i64> lengths;
    i64 total_length;
};
struct FileInterval{
    i64 start;
    i64 end;
};
struct InvalidCharsInterval{
    i64 start;
    i64 len;
};
struct BatchFileBufInfo{
    std::vector<FileInterval> intervals;
    std::vector<i64> fileIds;
    void clear(){
        intervals.clear();
        fileIds.clear();
    }
    bool is_last;
};
struct BatchFileInfo{
    std::vector<FileInterval> intervals;
    std::vector<i64> fileIds;
    // std::vector<InvalidCharsInterval> invalid_intervals;
    i64 id;
    bool is_first;
    bool is_last;
    void clear(){
        intervals.clear();
        fileIds.clear();
        // invalid_intervals.clear();
    }
};
struct ParseVectorBatch{
    OffsetVector<char> chars;
    OffsetVector<u64> seps;
    OffsetVector<u64> bits;
    OffsetVector<u64> rank;
};
struct GPUThreadLookupTableEntry{
    i32 substract;//threadid-substract = threadid in local batch, also used by threads to look through 
    //the vector of entries to find their own
    i32 read_off;//read_section+read_off = read_section in local batch
    i32 sep_off;//separate_section+sep_off = separate_section in local batch
    i32 bit_off;//bitvector_section+bit_off = bit_section in local batch
    i32 rank_off;//rank_section+rank_off = rank_section in local batch
    i32 out_off;//out_start=threadid-(k-1)*(bit_off,rank_off).rank(threadid-substract)+out_off
};
static_assert(sizeof(GPUThreadLookupTableEntry)%sizeof(u64)==0,"GPUThreadLookupTable size must be multiple of u64 for safe reinterpret_cast");
static_assert(alignof(u64)%alignof(GPUThreadLookupTableEntry)==0,"GPUThreadLookupTable must align with u64");
}//namespace sbwt_lcs_gpu