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
    i64 start;//in terms of chars
    i64 end;//exclusive
    i64 file_id;
    FileInterval(i64 _start, i64 _end, i64 _file_id):start(_start),end(_end),file_id(_file_id){}
    //end-start
    inline i64 length() const{
        return end-start;
    }
};
struct FileSepInterval{
    i64 start;//index in terms of seps
    i64 end;//exclusive
    i64 file_id;
    FileSepInterval(i64 _start, i64 _end, i64 _file_id):start(_start),end(_end),file_id(_file_id){}
    //end-start
    inline i64 length() const{
        return end-start;
    }
};
struct InvalidCharsInterval{
    i64 start;
    i64 len;
};
struct BatchFileBufInfo{
    std::vector<FileInterval> intervals;
    // std::vector<i64> fileIds;
    void reset(){
        intervals.clear();
        // fileIds.clear();
        is_last = false;
    }
    bool is_last;
};
struct BatchFileInfo{
    std::vector<FileSepInterval> intervals;
    // std::vector<i64> fileIds;
    // std::vector<InvalidCharsInterval> invalid_intervals;
    i64 id; //used to determine which stream this belongs to (when demultiplexing)
    bool is_first;
    bool is_last;
    void reset(){
        intervals.clear();
        // fileIds.clear();
        // invalid_intervals.clear();
        is_first = false;
        is_last = false;
        id = -1;
    }
};
struct ParseVectorBatch{
    OffsetVector<u64> chars;
    OffsetVector<u64> seps;
    OffsetVector<u64> bits;
    OffsetVector<u64> rank;
    i64 num_packed_chars=0;
    void reset(){
        num_packed_chars = 0;
        chars.clear();
        seps.clear();
        bits.clear();
        rank.clear();
    }
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