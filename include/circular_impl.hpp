#pragma once
#include "utils.h"
#include "structs.hpp"
#include "poppy.hpp"
#include "circular.hpp"
#include "gpu/gpu_utils.h"

#include <mutex>
#include <chrono>

namespace sbwt_lcs_gpu {

static std::mutex io_mutex;




// constexpr u64 poppysmall_from_bitvector_u64s_const(u64 num_u64s);
// struct FileBufSection {
// };
// struct ParseSection {
// };
// struct MultiplexSection {
// };
//doesn't particularly matter, these will be allocated as two different buffers, but useful for u64s calculation

// struct DemultiplexSection {
// };
// struct WriteBufSection {
// };
//u64s
struct MemoryPositions {
    // static constexpr u64 file_buf = 0;
    // static u64 parse;
    // static u64 multiplex;
    // static u64 demultiplex;
    // static u64 write_buf;
    static u64 total;
    // static u64 gpu; //gpu allocated separately
};

auto timenow(){
  return std::chrono::high_resolution_clock::now();
}
auto timeinsec(std::chrono::high_resolution_clock::duration t){
  return std::chrono::duration<double>(t);
}
auto timeinmsec(std::chrono::high_resolution_clock::duration t){
  return std::chrono::duration<double, std::milli>(t);
}
auto mscputime_from(std::clock_t t){
  return 1000.0*(std::clock()-t)/CLOCKS_PER_SEC;
}
extern std::chrono::duration<double> d_reader;
extern std::chrono::duration<double> d_parser;
extern std::chrono::duration<double> d_decoder;
extern std::chrono::duration<double> d_writer;
extern i64 n_reader;
extern i64 n_parser;
extern i64 n_decoder;
extern i64 n_writer;

//===

//multithreaded debug print
#define PRINT_MPDBG(x) do { \
    std::lock_guard<std::mutex> lock(io_mutex); \
    auto now = std::chrono::system_clock::now(); \
    auto now_c = std::chrono::system_clock::to_time_t(now); \
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000; \
    std::cout << "[" << std::put_time(std::localtime(&now_c), "%H:%M:%S") << "." << std::setfill('0') << std::setw(3) << milliseconds.count() << "] " << x << std::endl; \
} while (0)
//multithreaded debug print inline
#define PRINT_MPDBG_I(x) do { \
    std::lock_guard<std::mutex> lock(io_mutex); \
    std::cout << x << std::endl; \
} while (0)

// Helper macros to expand variadic arguments
#define EXPAND(x) x
#define GET_MACRO(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,NAME,...) NAME

// Base macros
#define OUT_0(x) #x ": " << x
#define OUT_1(x) << ", " #x ": " << x

// Variadic macro to handle multiple arguments
#define OUT(...) EXPAND(GET_MACRO(__VA_ARGS__, OUT10, OUT9, OUT8, OUT7, OUT6, OUT5, OUT4, OUT3, OUT2, OUT1)(__VA_ARGS__))

// Define macros for different numbers of arguments
#define OUT1(x1) OUT_0(x1)
#define OUT2(x1, x2) OUT_0(x1) OUT_1(x2)
#define OUT3(x1, x2, x3) OUT_0(x1) OUT_1(x2) OUT_1(x3)
#define OUT4(x1, x2, x3, x4) OUT_0(x1) OUT_1(x2) OUT_1(x3) OUT_1(x4)
#define OUT5(x1, x2, x3, x4, x5) OUT_0(x1) OUT_1(x2) OUT_1(x3) OUT_1(x4) OUT_1(x5)
#define OUT6(x1, x2, x3, x4, x5, x6) OUT_0(x1) OUT_1(x2) OUT_1(x3) OUT_1(x4) OUT_1(x5) OUT_1(x6)
#define OUT7(x1, x2, x3, x4, x5, x6, x7) OUT_0(x1) OUT_1(x2) OUT_1(x3) OUT_1(x4) OUT_1(x5) OUT_1(x6) OUT_1(x7)
#define OUT8(x1, x2, x3, x4, x5, x6, x7, x8) OUT_0(x1) OUT_1(x2) OUT_1(x3) OUT_1(x4) OUT_1(x5) OUT_1(x6) OUT_1(x7) OUT_1(x8)
#define OUT9(x1, x2, x3, x4, x5, x6, x7, x8, x9) OUT_0(x1) OUT_1(x2) OUT_1(x3) OUT_1(x4) OUT_1(x5) OUT_1(x6) OUT_1(x7) OUT_1(x8) OUT_1(x9)
#define OUT10(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10) OUT_0(x1) OUT_1(x2) OUT_1(x3) OUT_1(x4) OUT_1(x5) OUT_1(x6) OUT_1(x7) OUT_1(x8) OUT_1(x9) OUT_1(x10)



#ifdef DEBUG
#define DEBUG_PRINTOUT 1
#else
#define DEBUG_PRINTOUT 0
#endif

#if DEBUG_PRINTOUT
#define DEBUG_PRINT_MPDBG(x) PRINT_MPDBG(x)
#define DEBUG_PRINT_MPDBG_I(x) PRINT_MPDBG_I(x)
#else
#define DEBUG_PRINT_MPDBG(x)
#define DEBUG_PRINT_MPDBG_I(x)
#endif

// using FileNameVec = std::vector<std::vector<std::tuple<std::string,std::string,i64>>>;
class FileReadStream: public CB_data<FileReadStream>{
    public:
    i64 current_file=0;
    static constexpr u64 logic_batch_size = batch_buf_u64s*sizeof(u64);
    FILE* file=nullptr;
    //no logic_batch_size here since v_r is not used
    FileReadStream(i64 _size, i32 _max_writers, i32 _max_readers): 
        CB_data(_size, _max_writers, _max_readers) {}
    // std::string dump_impl(){
    //     std::stringstream ss;
    //     ss <<" gen_num "<<gen_num;
    //     return ss.str();
    // }
};
class FileReadMS: public SharedThreadMultistream<FileReadMS,FileReadStream>{
    public:
    static constexpr bool has_write=false;
    static constexpr bool has_read=true;
    static constexpr StreamType stream_type=SEQUENTIAL;
    static i32 num_threads;
    // FileNameVec files;
    std::vector<StreamFilenamesContainer>& files;
    i64 current_index=0;
    // std::string dump_impl(){
    //     std::stringstream ss;
    //     ss <<" gen_num "<<gen_num
    //     << ", gen_max "<<gen_max;
    //     return ss.str();
    // }
    FileReadMS(std::vector<StreamFilenamesContainer>& _files):
        SharedThreadMultistream(0, num_physical_streams, num_physical_streams, 0, 1, 0),
        files(_files){
            initialize_base(0, num_physical_streams, num_physical_streams, 0, 1, 0);
        }
    void allocate_impl(i32 stream_indx, i64 id){
        // data[stream_indx].gen_num=gen_num;
        data[stream_indx].id=current_index;
        data[stream_indx].size=files[current_index].total_length;
        data[stream_indx].v_r=data[stream_indx].size;
        data[stream_indx].v_w=0;
        current_index++;
        DEBUG_PRINT_MPDBG("FileReadMS::allocate_impl, " << OUT(stream_indx, id));
    }
    void deallocate_impl(i32 stream_indx){
        //nothing to do
        // PRINT("gen_MS_impl::deallocate_impl " << std::this_thread::get_id() << " sid " << stream_indx);
        // if(gen_num==gen_max){
        //     final_dealloc();
        // }
        DEBUG_PRINT_MPDBG("FileReadMS::deallocate_impl, " << OUT(stream_indx));
    }
    bool can_allocate_impl(){
        // return gen_num<gen_max;
        return current_index<files.size();
    }
    bool no_write_can_read_impl(i32 stream_indx){
        auto& R = data[stream_indx];
        return R.current_file<files[R.id].filenames.size();
    }
    bool no_write_ended_impl(i32 stream_indx, i64 buf_S, i64 r_size){
        auto& R = data[stream_indx];
        return R.current_file>=files[R.id].filenames.size();
    }
    i64 get_write_id_impl(i32 stream_indx){
        // return data[stream_indx].id+10;
        return data[stream_indx].id;
    }
    // bool get_is_last_child_impl(i32 stream_indx, i64 buf_E, bool is_last_parent){return 1;}//not relevant
    bool get_is_first_parent_impl(i32 stream_indx, bool is_begin){return is_begin;}
};
class FileBufStream: public CB_data<FileBufStream>{
    public:
    static constexpr u64 batch_u64s = batch_buf_u64s;
    static constexpr u64 u64s=batches_in_stream*batch_u64s;
    static constexpr u64 logic_batch_size = batch_buf_u64s*sizeof(u64);
    static constexpr u64 logic_size = batches_in_stream*logic_batch_size;
    //the entire stream data, not batch
    OffsetVector<char> stream_data;
    std::vector<BatchFileBufInfo> batch_info;
    bool ended=false;
    FileBufStream(i64 _size, i32 _max_writers, i32 _max_readers, OffsetVector<char>&& _data):
        CB_data(_size, _max_writers, _max_readers),
        stream_data(std::move(_data)), batch_info(batches_in_stream){}
};
class FileBufMS: public SharedThreadMultistream<FileBufMS,FileBufStream>{
    public:
    static u64 u64s; //depends on number of streams
    static constexpr u64 data_offset = 0;//in u64s
    static constexpr bool has_write=true;
    static constexpr bool has_read=true;
    static constexpr StreamType stream_type=SEQUENTIAL;
    static i32 num_threads;
    OffsetVector<u64> section_data;
    // FileBufMS(i64 stream_size, i32 num_streams, i32 max_readers, i32 max_writers,
    //     i32 max_readers_per_stream, i32 max_writers_per_stream, OffsetVector<u64>&& _section_data):
    // FileBufMS(OffsetVector<u64>&& _section_data):
    FileBufMS(std::vector<u64>& memory):
        SharedThreadMultistream(0, num_physical_streams, num_physical_streams, num_physical_streams, 1, 1),
        section_data(memory, data_offset, u64s, OffsetVectorOpts::SET_MAX_SIZE){
            initialize_base(0, num_physical_streams, num_physical_streams, num_physical_streams, 1, 1);
        }
        // section_data(std::move(_section_data)) {}

    void emplace_back_data(i64 _1, i32 _2, i32 _3){
        OffsetVector<char> temp(section_data, data.size()*FileBufStream::u64s, FileBufStream::u64s);
        // data.back().stream_data.resize(FileBufStream::u64s*sizeof(u64));
        temp.resize(FileBufStream::u64s*sizeof(u64));
        data.emplace_back(FileBufStream::logic_size, 1, 1, std::move(temp));
    }
    void allocate_impl(i32 stream_indx, i64 id){
        data[stream_indx].ended=false;
        DEBUG_PRINT_MPDBG("FileBufMS::allocate_impl, " << OUT(stream_indx, id));
    }
    void deallocate_impl(i32 stream_indx){
        DEBUG_PRINT_MPDBG("FileBufMS::deallocate_impl, " << OUT(stream_indx));
    }
    i64 get_write_id_impl(i32 stream_indx){return data[stream_indx].id;}
    bool sequential_ended_impl(i32 stream_indx,i64 buf_S, i64 r_size){return data[stream_indx].ended;}
    bool get_is_last_child_impl(i32 stream_indx, i64 buf_E, bool is_last_parent){return is_last_parent;}
    bool get_is_first_parent_impl(i32 stream_indx, bool is_begin){return is_begin;}
};
enum ParseState {
    NEW_FILE,//becomes BEGIN_FROM_SKIP
    BEGIN_FROM_SKIP,//seq_begin -> skip_then_read, qual_begin -> skip_then_skip, else -> skip
    BEGIN_FROM_NEW_READ,//same as above but else -> read
    BEGIN_FROM_READ,//same as above but on change will do finish_seq()
    SKIP,//skip until newline then BEGIN_FROM_SKIP
    SKIP_THEN_READ,//skip until newline then BEGIN_FROM_READ
    //necessary because quality can have @, which is also the start of a new read
    SKIP_THEN_SKIP,//skip until newline then SKIP_UNTIL_CHAR
    SKIP_UNTIL_CHAR,//skip until a non-newline then SKIP
    READ//read until newline then BEGIN_FROM_READ
};
class ParseStream: public CB_data<ParseStream>{
    public:
    static constexpr u64 chars_batch_start=0;
    static constexpr u64 chars_batch_u64s = batch_buf_u64s;

    static constexpr u64 seps_batch_start = chars_batch_start + chars_batch_u64s;
    static constexpr u64 seps_batch_u64s = ceil_div_const(chars_batch_u64s * chars_per_u64, max_read_chars) + 8;

    static constexpr u64 seps_bitvector_batch_start = seps_batch_start + seps_batch_u64s;
    static constexpr u64 seps_bitvector_batch_u64s = ceil_div_const(seps_batch_u64s+8, u64_bits) + bitvector_pad_u64s;

    static constexpr u64 seps_rank_batch_start = seps_bitvector_batch_start + seps_bitvector_batch_u64s;
    static constexpr u64 seps_rank_batch_u64s = poppysmall_from_bitvector_u64s_const(seps_bitvector_batch_u64s);

    static constexpr u64 batch_u64s = chars_batch_u64s + seps_batch_u64s + seps_bitvector_batch_u64s + seps_rank_batch_u64s;
    static constexpr u64 u64s = batch_u64s * batches_in_stream;
    static constexpr u64 logic_batch_size = chars_batch_u64s*chars_per_u64;
    static constexpr u64 logic_size = batches_in_stream*logic_batch_size;

    const i32 k_;//local copy of k
    OffsetVector<u64> stream_data;

    std::vector<BatchFileInfo> batch_info;
    FCVector<ParseVectorBatch> batches;
    // ParseState parse_state = BEGIN_FROM_SKIP;
    FCVector<u64> temp_buffer_;//for storing end of previous batch

    ParseState parse_state = NEW_FILE;
    // i64 num_to_search=0;//store between do_work calls
    OffsetVector<u64> temp_buffer;
    i64 num_chars_in_temp=0;//num of 2-bit chars in temp_buffer
    i64 temp_file_id=0;//file id to be used to create old interval in new batch
    i64 file_interval=0;//index of interval in the file intervals of read batch
    // i64 last_sep=0;//sanity check that (seps[num_seps-2]+(1-bits[num_seps-2])*k)==last_sep
    i64 total_kmers=0;//sanity check that (last_sep-rank(bits)*k)==total_kmers
    void reset(){//on new stream init
        parse_state = NEW_FILE;
        // num_to_search=0;
        temp_buffer.clear();
        num_chars_in_temp=0;//num of 2-bit chars in temp_buffer
        temp_file_id=0;
        file_interval=0;
    }

    ParseStream(i64 _size, i32 _max_writers, i32 _max_readers, OffsetVector<u64>&& _data):
        CB_data(_size, _max_writers, _max_readers), 
        k_(k),
        temp_buffer_(ceil_div(max_read_chars,chars_per_u64)*3,0),//max_read_chars*3 should be enough
        temp_buffer(temp_buffer_),
        stream_data(std::move(_data)), batch_info(batches_in_stream), batches(batches_in_stream) {
            for(i32 i=0;i<batches_in_stream;i++){
                batches.emplace_back(ParseVectorBatch{
                    OffsetVector<u64>(stream_data, i*batch_u64s+chars_batch_start, chars_batch_u64s, OffsetVectorOpts::SET_MAX_SIZE),//!must be fully allocated
                    OffsetVector<u64>(stream_data,  i*batch_u64s+seps_batch_start, seps_batch_u64s),
                    OffsetVector<u64>(stream_data,  i*batch_u64s+seps_bitvector_batch_start, seps_bitvector_batch_u64s),
                    OffsetVector<u64>(stream_data,  i*batch_u64s+seps_rank_batch_start, seps_rank_batch_u64s, OffsetVectorOpts::SET_MAX_SIZE)//!must also be fully allocated
                });
                // batches.back().chars.resize(chars_batch_u64s*sizeof(u64)/sizeof(char));
                // batches.back().seps.resize(seps_batch_u64s);
                // batches.back().bits.resize(seps_bitvector_batch_u64s);
                // batches.back().rank.resize(seps_rank_batch_u64s);
            }
        }
    //!debug
    // bool ended = false;
};
constexpr bool USE_DEBUG_PARSEMS=true;
class ParseMS: public SharedThreadMultistream<ParseMS,ParseStream>{
    public:
    static u64 u64s;
    static u64 data_offset;
    static constexpr bool has_write=true;
    static constexpr bool has_read=true;
    // static constexpr StreamType stream_type=USE_DEBUG_PARSEMS?SEQUENTIAL:PARALLEL;
    static constexpr StreamType stream_type=PARALLEL;
    static i32 num_threads;
    OffsetVector<u64> section_data;
    ParseMS(std::vector<u64>& memory):
        SharedThreadMultistream(0, num_physical_streams, num_physical_streams, num_physical_streams, 1, 1),
        section_data(memory, data_offset, u64s, OffsetVectorOpts::SET_MAX_SIZE){
            initialize_base(0, num_physical_streams, num_physical_streams, num_physical_streams, 1, 1);
        }
    void emplace_back_data(i64 _1, i32 _2, i32 _3){
        OffsetVector<u64> temp(section_data, data.size()*ParseStream::u64s, ParseStream::u64s);
        // data.back().stream_data.resize(ParseStream::u64s);
        temp.resize(ParseStream::u64s);
        data.emplace_back(ParseStream::logic_size, 1, 1, //no reason to use more than 1 reader?
        std::move(temp));
    }
    void allocate_impl(i32 stream_indx, i64 id){
        // data[stream_indx].id=id;
        data[stream_indx].reset();
        //!debug
        // data[stream_indx].ended=false;
        DEBUG_PRINT_MPDBG("ParseMS::allocate_impl, " << OUT(stream_indx, id));
    }
    void deallocate_impl(i32 stream_indx){
        DEBUG_PRINT_MPDBG("ParseMS::deallocate_impl, " << OUT(stream_indx));
    }
    i64 get_write_id_impl(i32 stream_indx){
        if constexpr(USE_DEBUG_PARSEMS){//!debug
            return data[stream_indx].id;
        }else{
            return 0;
        }//multiplex into one stream
    }
    bool get_is_last_child_impl(i32 stream_indx, i64 buf_E, bool is_last_parent){return is_last_parent;}
    bool get_is_first_parent_impl(i32 stream_indx, bool is_begin){return is_begin;}
    //!debug
    // bool sequential_ended_impl(i32 stream_indx,i64 buf_S, i64 r_size){return data[stream_indx].ended;}
};

class MultiplexStream: public CB_data<MultiplexStream>{
    public:
    static constexpr u64 chars_batch_section_start = 0;
    static constexpr u64 chars_batch_section_u64s = ParseStream::chars_batch_u64s * gpu_batch_mult;

    static constexpr u64 seps_batch_section_start = chars_batch_section_start + chars_batch_section_u64s;
    static constexpr u64 seps_batch_section_u64s = ParseStream::seps_batch_u64s * gpu_batch_mult;

    static constexpr u64 seps_bitvector_batch_section_start = seps_batch_section_start + seps_batch_section_u64s;
    static constexpr u64 seps_bitvector_batch_section_u64s = ParseStream::seps_bitvector_batch_u64s * gpu_batch_mult;

    static constexpr u64 seps_rank_batch_section_start = seps_bitvector_batch_section_start + seps_bitvector_batch_section_u64s;
    static constexpr u64 seps_rank_batch_section_u64s = ParseStream::seps_rank_batch_u64s * gpu_batch_mult;

    static constexpr u64 thread_lookup_vector_start = seps_rank_batch_section_start + seps_rank_batch_section_u64s;
    static constexpr u64 thread_lookup_vector_u64s = sizeof(GPUThreadLookupTableEntry) * gpu_batch_mult / sizeof(u64);

    static constexpr u64 batch_section_u64s = chars_batch_section_u64s + seps_batch_section_u64s + seps_bitvector_batch_section_u64s + seps_rank_batch_section_u64s + thread_lookup_vector_u64s;
    static constexpr u64 u64s = batch_section_u64s * batches_in_gpu_stream; //~for now do not scale batches with number of streams

    static constexpr u64 logic_batch_size = 1;
    static constexpr u64 logic_gpu_batch_size = gpu_batch_mult*logic_batch_size;
    static constexpr u64 logic_size = batches_in_gpu_stream*logic_gpu_batch_size;

    OffsetVector<u64> stream_data;
    GpuPointer<u64> stream_gpu_data;
    std::vector<BatchFileInfo> batch_info;
    FCVector<ParseVectorBatch> batches;
    FCVector<OffsetVector<GPUThreadLookupTableEntry>> lookup_tables;
    FCVector<GpuPointer<u64>> gpu_batches;
    MultiplexStream(i64 _size, i32 _max_writers, i32 _max_readers, OffsetVector<u64>&& _data, GpuPointer<u64>&& _gpu_data);
};
struct GPUSection {
    static constexpr u64 in_batch_u64s = MultiplexStream::batch_section_u64s;
    static constexpr u64 out_batch_u64s = MultiplexStream::chars_batch_section_u64s * chars_per_u64+100;//1 u64 per char + padding
    static constexpr u64 in_u64s = in_batch_u64s * batches_in_gpu_stream; //~for now do not scale batches with number of streams
    static constexpr u64 out_u64s = out_batch_u64s * batches_in_gpu_stream; //~for now do not scale batches with number of streams
    
    static constexpr u64 start_in = 0;
    static constexpr u64 start_out = in_u64s;

    static constexpr u64 u64s = in_u64s + out_u64s; //~for now do not scale batches with number of streams
};
MultiplexStream::MultiplexStream(i64 _size, i32 _max_writers, i32 _max_readers, OffsetVector<u64>&& _data, GpuPointer<u64>&& _gpu_data):
        CB_data(_size, _max_writers, _max_readers),
        stream_data(std::move(_data)), batch_info(_size), stream_gpu_data(std::move(_gpu_data)), 
        batches(gpu_batch_mult*batches_in_gpu_stream), lookup_tables(batches_in_gpu_stream), gpu_batches(batches_in_gpu_stream){
            for(i32 i=0;i<batches_in_gpu_stream;i++){
                for(i32 j=0;j<gpu_batch_mult;j++){
                    batches.emplace_back(ParseVectorBatch{
                        OffsetVector<u64>(stream_data, i*batch_section_u64s+chars_batch_section_start+
                        j*chars_batch_section_u64s/gpu_batch_mult, chars_batch_section_u64s/gpu_batch_mult),
                        OffsetVector<u64>(stream_data,  i*batch_section_u64s+seps_batch_section_start+
                        j*seps_batch_section_u64s/gpu_batch_mult, seps_batch_section_u64s/gpu_batch_mult),
                        OffsetVector<u64>(stream_data,  i*batch_section_u64s+seps_bitvector_batch_section_start+
                        j*seps_bitvector_batch_section_u64s/gpu_batch_mult, seps_bitvector_batch_section_u64s/gpu_batch_mult),
                        OffsetVector<u64>(stream_data,  i*batch_section_u64s+seps_rank_batch_section_start+
                        j*seps_rank_batch_section_u64s/gpu_batch_mult, seps_rank_batch_section_u64s/gpu_batch_mult)
                    });
                    // batches.back().chars.resize(chars_batch_section_u64s/gpu_batch_mult*sizeof(u64)/sizeof(char));
                    // batches.back().seps.resize(seps_batch_section_u64s/gpu_batch_mult);
                    // batches.back().bits.resize(seps_bitvector_batch_section_u64s/gpu_batch_mult);
                    // batches.back().rank.resize(seps_rank_batch_section_u64s/gpu_batch_mult);
                }
                lookup_tables.emplace_back(stream_data, i*batch_section_u64s+thread_lookup_vector_start,
                thread_lookup_vector_u64s);
                // lookup_tables.back().resize(thread_lookup_vector_u64s*sizeof(u64)/sizeof(GPUThreadLookupTableEntry));
                gpu_batches.emplace_back(stream_gpu_data, i*GPUSection::in_batch_u64s+GPUSection::start_in,
                GPUSection::in_batch_u64s);
            }
        }
class MultiplexMS: public SharedThreadMultistream<MultiplexMS,MultiplexStream>{
    public:
    static constexpr bool has_write=true;
    static constexpr bool has_read=true;
    static constexpr StreamType stream_type=PARALLEL;
    static i32 num_threads;
    //for now, only one stream
    static constexpr u64 u64s = MultiplexStream::u64s;//~for now do not scale batches with number of streams
    static u64 data_offset;
    OffsetVector<u64> section_data;
    GpuPointer<u64> gpu_data;
    i64 total_num_streams;
    MultiplexMS(std::vector<u64>& memory, GpuPointer<u64>& _gpu_data, i64 _total_num_streams):
        SharedThreadMultistream(0, 1, gpu_readers, num_physical_streams,  gpu_readers, num_physical_streams),
        section_data(memory, data_offset, u64s, OffsetVectorOpts::SET_MAX_SIZE),
        gpu_data(_gpu_data, GPUSection::start_in, GPUSection::in_u64s),
        total_num_streams(_total_num_streams) {
            initialize_base(0, 1, gpu_readers, num_physical_streams,  gpu_readers, num_physical_streams);
        }
    void emplace_back_data(i64 _1, i32 _2, i32 _3){
        OffsetVector<u64> temp(section_data, data.size()*MultiplexStream::u64s, MultiplexStream::u64s);
        // data.back().stream_data.resize(MultiplexStream::u64s);
        temp.resize(MultiplexStream::u64s);
        data.emplace_back(MultiplexStream::logic_size, num_physical_streams, gpu_readers,
        std::move(temp), GpuPointer<u64>(gpu_data, 0, GPUSection::in_u64s));
    }
    void allocate_impl(i32 stream_indx, i64 id){
        // data[stream_indx].id=id;
    }
    void deallocate_impl(i32 stream_indx){}
    i64 get_write_id_impl(i32 stream_indx){return 0;}//multiplexer -> demultiplexer stream
    bool get_is_last_child_impl(i32 stream_indx, i64 buf_E, bool is_last_parent){
        total_num_streams--;
        return total_num_streams==0;
    }
    bool get_is_first_parent_impl(i32 stream_indx, bool is_begin){return is_begin;}//only need to make one stream
};
class DemultiplexStream: public CB_data<DemultiplexStream>{
    public:
    static constexpr u64 indexes_batch_u64s = GPUSection::out_batch_u64s;
    static constexpr u64 u64s = indexes_batch_u64s * batches_in_gpu_stream; //~for now do not scale batches with number of streams
    static constexpr u64 logic_batch_size = MultiplexStream::logic_batch_size;
    static constexpr u64 logic_gpu_batch_size = MultiplexStream::logic_gpu_batch_size;
    static constexpr u64 logic_size = MultiplexStream::logic_size;

    OffsetVector<u64> stream_data;
    GpuPointer<u64> stream_gpu_data;
    std::vector<BatchFileInfo> batch_info;
    std::vector<std::vector<i32>> out_offsets;
    FCVector<GpuPointer<u64>> gpu_batches;
    DemultiplexStream(i64 _size, i32 _max_writers, i32 _max_readers, OffsetVector<u64>&& _data, GpuPointer<u64>&& _gpu_data):
        CB_data(_size, _max_writers, _max_readers),
        stream_data(std::move(_data)), batch_info(_size), stream_gpu_data(std::move(_gpu_data)),
        out_offsets(batches_in_gpu_stream,{gpu_batch_mult}), gpu_batches(batches_in_gpu_stream){
            for(i32 i=0;i<batches_in_gpu_stream;i++){
                gpu_batches.emplace_back(stream_gpu_data, i*GPUSection::out_batch_u64s+GPUSection::start_out,
                GPUSection::out_batch_u64s);
            }
        }
};
class DemultiplexMS: public SharedThreadMultistream<DemultiplexMS,DemultiplexStream>{
    public:
    static constexpr u64 u64s=DemultiplexStream::u64s;//~for now do not scale batches with number of streams
    static u64 data_offset;
    static constexpr bool has_write=true;
    static constexpr bool has_read=true;
    static constexpr StreamType stream_type=PARALLEL;
    static i32 num_threads;
    OffsetVector<u64> section_data;
    GpuPointer<u64> gpu_data;
    DemultiplexMS(std::vector<u64>& memory, GpuPointer<u64>& _gpu_data):
        SharedThreadMultistream(0, 1, num_physical_streams, gpu_readers,  num_physical_streams, gpu_readers),
        section_data(memory, data_offset, u64s, OffsetVectorOpts::SET_MAX_SIZE),
        gpu_data(_gpu_data, GPUSection::start_out, GPUSection::out_u64s) {
            initialize_base(0, 1, num_physical_streams, gpu_readers,  num_physical_streams, gpu_readers);
        }
    void emplace_back_data(i64 _1, i32 _2, i32 _3){
        OffsetVector<u64> temp(section_data, data.size()*DemultiplexStream::u64s, DemultiplexStream::u64s);
        // data.back().stream_data.resize(DemultiplexStream::u64s);
        temp.resize(DemultiplexStream::u64s);
        data.emplace_back(DemultiplexStream::logic_size, gpu_readers, num_physical_streams,
        std::move(temp), GpuPointer<u64>(gpu_data, 0, GPUSection::out_u64s));
    }
    void allocate_impl(i32 stream_indx, i64 id){
        // data[stream_indx].id=id;
    }
    void deallocate_impl(i32 stream_indx){}
    i64 get_write_id_impl(i32 stream_indx){
        return data[stream_indx].batch_info[data[stream_indx].buf_S0].id;
    }
    bool get_is_last_child_impl(i32 stream_indx, i64 buf_E, bool is_last_parent){
        return is_last_parent;
    }
    // return data[stream_indx].batch_info[buf_E].is_last;
    bool get_is_first_parent_impl(i32 stream_indx, bool is_begin){
        return data[stream_indx].batch_info[data[stream_indx].buf_S0].is_first;
    }
};

//!probably have to add queue and a mutex for the queue
class WriteBufStream: public CB_data<WriteBufStream>{
    public:
    static constexpr u64 batch_u64s = ceil_double_div_const(ceil_div_const(DemultiplexStream::indexes_batch_u64s,gpu_batch_mult), out_compression_ratio);
    static constexpr u64 u64s=batches_in_out_buf_stream*batch_u64s;
    static constexpr u64 logic_batch_size = batch_u64s*sizeof(u64);
    static constexpr u64 logic_size = batches_in_out_buf_stream*logic_batch_size;
    OffsetVector<u8> stream_data;
    std::vector<BatchFileBufInfo> batch_info;
    bool ended=false;
    WriteBufStream(i64 _size, i32 _max_writers, i32 _max_readers, OffsetVector<u8>&& _data):
        CB_data(_size, _max_writers, _max_readers),
        stream_data(std::move(_data)), batch_info(batches_in_out_buf_stream) {}
};
class WriteBufMS: public SharedThreadMultistream<WriteBufMS,WriteBufStream>{
    public:
    static u64 u64s;
    static u64 data_offset;
    static constexpr bool has_write=true;
    static constexpr bool has_read=true;
    static constexpr StreamType stream_type=SEQUENTIAL;
    static i32 num_threads;
    OffsetVector<u64> section_data;
    WriteBufMS(std::vector<u64>& memory):
        SharedThreadMultistream(0, num_physical_streams, num_physical_streams, num_physical_streams, 1, 1),
        section_data(memory, data_offset, u64s, OffsetVectorOpts::SET_MAX_SIZE) {
            initialize_base(0, num_physical_streams, num_physical_streams, num_physical_streams, 1, 1);
        }
    void emplace_back_data(i64 _1, i32 _2, i32 _3){
        OffsetVector<u8> temp(section_data, data.size()*WriteBufStream::u64s, WriteBufStream::u64s);
        // data.back().stream_data.resize(WriteBufStream::u64s*sizeof(u64));
        temp.resize(WriteBufStream::u64s*sizeof(u64));
        data.emplace_back(WriteBufStream::logic_size, 1, 1, std::move(temp));
    }
    void allocate_impl(i32 stream_indx, i64 id){
        data[stream_indx].ended=false;
    }
    void deallocate_impl(i32 stream_indx){}
    i64 get_write_id_impl(i32 stream_indx){return data[stream_indx].id;}
    bool sequential_ended_impl(i32 stream_indx,i64 buf_S, i64 r_size){return data[stream_indx].ended;}
    bool get_is_last_child_impl(i32 stream_indx, i64 buf_E, bool is_last_parent){
        return data[stream_indx].batch_info[buf_E/WriteBufStream::batch_u64s].is_last;
    }
    bool get_is_first_parent_impl(i32 stream_indx, bool is_begin){return is_begin;}
};
//DEBUG
class DebugWriteBufStream: public CB_data<DebugWriteBufStream>{
    public:
    static constexpr u64 batch_u64s = ceil_div_const(ParseStream::logic_batch_size+ParseStream::seps_batch_u64s,sizeof(u64));
    static constexpr u64 u64s=batches_in_out_buf_stream*batch_u64s;
    static constexpr u64 logic_batch_size = batch_u64s*sizeof(u64);
    static constexpr u64 logic_size = batches_in_out_buf_stream*logic_batch_size;
    OffsetVector<u8> stream_data;
    std::vector<BatchFileBufInfo> batch_info;
    bool ended=false;
    DebugWriteBufStream(i64 _size, i32 _max_writers, i32 _max_readers, OffsetVector<u8>&& _data):
        CB_data(_size, _max_writers, _max_readers),
        stream_data(std::move(_data)), batch_info(batches_in_out_buf_stream) {}
};
class DebugWriteBufMS: public SharedThreadMultistream<DebugWriteBufMS,DebugWriteBufStream>{
    public:
    static u64 u64s;
    static u64 data_offset;
    static constexpr bool has_write=true;
    static constexpr bool has_read=true;
    static constexpr StreamType stream_type=SEQUENTIAL;
    static i32 num_threads;
    OffsetVector<u64> section_data;
    DebugWriteBufMS(std::vector<u64>& memory):
        SharedThreadMultistream(0, num_physical_streams, num_physical_streams, num_physical_streams, 1, 1),
        section_data(memory, data_offset, u64s, OffsetVectorOpts::SET_MAX_SIZE) {
            initialize_base(0, num_physical_streams, num_physical_streams, num_physical_streams, 1, 1);
        }
    void emplace_back_data(i64 _1, i32 _2, i32 _3){
        OffsetVector<u8> temp(section_data, data.size()*DebugWriteBufStream::u64s, DebugWriteBufStream::u64s);
        // data.back().stream_data.resize(DebugWriteBufStream::u64s*sizeof(u64));
        temp.resize(DebugWriteBufStream::u64s*sizeof(u64));
        data.emplace_back(DebugWriteBufStream::logic_size, 1, 1, std::move(temp));
    }
    void allocate_impl(i32 stream_indx, i64 id){
        data[stream_indx].ended=false;
        DEBUG_PRINT_MPDBG("DebugWriteBufMS::allocate_impl, " << OUT(stream_indx, id));
    }
    void deallocate_impl(i32 stream_indx){
        DEBUG_PRINT_MPDBG("DebugWriteBufMS::deallocate_impl, " << OUT(stream_indx));
    }
    i64 get_write_id_impl(i32 stream_indx){return data[stream_indx].id;}
    bool sequential_ended_impl(i32 stream_indx,i64 buf_S, i64 r_size){return data[stream_indx].ended;}
    bool get_is_last_child_impl(i32 stream_indx, i64 buf_E, bool is_last_parent){
        return is_last_parent;
    }
    bool get_is_first_parent_impl(i32 stream_indx, bool is_begin){return is_begin;}
};

class FileOutStream: public CB_data<FileOutStream>{
    public:
    static constexpr u64 logic_batch_size = WriteBufStream::logic_batch_size;
    FILE* file=nullptr;
    i64 fileid=-1;
    FileOutStream(i64 _size, i32 _max_writers, i32 _max_readers): 
        CB_data(_size, _max_writers, _max_readers) {}
};
class FileOutMS: public SharedThreadMultistream<FileOutMS,FileOutStream>{
    public:
    static constexpr bool has_write=true;
    static constexpr bool has_read=false;
    // static constexpr StreamType stream_type=SEQUENTIAL;//don't need since no read
    std::vector<StreamFilenamesContainer>& files;
    i64 current_index=0;
    FileOutMS(std::vector<StreamFilenamesContainer>& _files):
        SharedThreadMultistream(FileOutStream::logic_batch_size, num_physical_streams, 0, num_physical_streams, 0, 1),
        files(_files) {
            initialize_base(FileOutStream::logic_batch_size, num_physical_streams, 0, num_physical_streams, 0, 1);
        }
    void allocate_impl(i32 stream_indx, i64 id){
        //none of this should matter since !has_read
        // data[stream_indx].size=files[current_index].total_length;
        // data[stream_indx].v_w=data[stream_indx].size;
        data[stream_indx].fileid=-1;
        DEBUG_PRINT_MPDBG("FileOutMS::allocate_impl, " << OUT(stream_indx, id));
    }
    void deallocate_impl(i32 stream_indx){}
    bool get_is_last_child_impl(i32 stream_indx, i64 buf_E, bool is_last_parent){return is_last_parent;}
};




void update_sections(){
    FileBufMS::u64s = FileBufStream::u64s * num_physical_streams;
    ParseMS::u64s = ParseStream::u64s * num_physical_streams;
    // MultiplexSection::u64s = MultiplexSection::batch_section_u64s * num_physical_streams;
    // GPUSection::in_u64s = GPUSection::in_batch_u64s * num_physical_streams;
    // GPUSection::out_u64s = GPUSection::out_batch_u64s * num_physical_streams;
    // GPUSection::u64s = GPUSection::in_u64s + GPUSection::out_u64s;
    // DemultiplexSection::u64s = DemultiplexSection::indexes_batch_u64s * num_physical_streams;
    WriteBufMS::u64s = WriteBufStream::u64s * num_physical_streams;
    DebugWriteBufMS::u64s = DebugWriteBufStream::u64s * num_physical_streams;//!debug

    ParseMS::data_offset = FileBufMS::data_offset + FileBufMS::u64s;

    MultiplexMS::data_offset = ParseMS::data_offset + ParseMS::u64s;
    DemultiplexMS::data_offset = MultiplexMS::data_offset + MultiplexMS::u64s;
    WriteBufMS::data_offset = DemultiplexMS::data_offset + DemultiplexMS::u64s;
    // MemoryPositions::total = WriteBufMS::data_offset + WriteBufMS::u64s;//!debug

    DebugWriteBufMS::data_offset = ParseMS::data_offset + ParseMS::u64s;//!debug
    MemoryPositions::total = DebugWriteBufMS::data_offset + DebugWriteBufMS::u64s;//!debug

    FileReadMS::num_threads = num_physical_streams;
    FileBufMS::num_threads = num_physical_streams;
    ParseMS::num_threads = num_physical_streams;
    MultiplexMS::num_threads = 4;
    DemultiplexMS::num_threads = num_physical_streams;
    WriteBufMS::num_threads = num_physical_streams;
    DebugWriteBufMS::num_threads = num_physical_streams;
    // total_threads = FileReadMS::num_threads + FileBufMS::num_threads + ParseMS::num_threads + MultiplexMS::num_threads + DemultiplexMS::num_threads + WriteBufMS::num_threads;
    total_threads = FileReadMS::num_threads + FileBufMS::num_threads + ParseMS::num_threads + DebugWriteBufMS::num_threads;
}



class FileReadWorker: public MS_Worker<FileReadWorker,FileReadMS,FileBufMS>{
    public:
    FileReadWorker(FileReadMS& _MS_r, FileBufMS& _MS_w):
        MS_Worker(_MS_r, _MS_w,FileReadStream::logic_batch_size,FileBufStream::logic_batch_size) {}
    //always aligned to batch size
    std::pair<i64,i64> do_work(
        i32 read_indx, i32 write_indx, i64 r_size, i64 buf_read, i64 buf_write,
        i64 read_step, i64 write_step,  bool is_first_child, bool is_last_parent){
        auto start=timenow();
        DEBUG_PRINT_MPDBG("FileReadWorker::do_work begin: " << OUT(read_indx, write_indx, r_size, buf_read, buf_write, read_step, write_step, is_first_child, is_last_parent));
        auto& R = MS_r.data[read_indx];
        auto& W = MS_w.data[write_indx];
        i64 total_read = 0;
        i64 batch_index = buf_write / FileBufStream::logic_batch_size;
        //valid because always read batch size, or at the end read until end of file(s)
        if(buf_write%FileBufStream::logic_batch_size!=0){//!err
            PRINT_MPDBG("FileReadWorker::do_work: buf_write not aligned to batch size: "<<buf_write);
            throw std::runtime_error("FileReadWorker::do_work: buf_write not aligned to batch size");
        }
        
        
        BatchFileBufInfo& batch_info = W.batch_info[batch_index];
        batch_info.reset();
        
        while (total_read < r_size && R.current_file < MS_r.files[R.id].filenames.size()) {
            if (R.file == nullptr) {
                R.file = fopen(MS_r.files[R.id].filenames[R.current_file].c_str(), "rb");
                #if DEBUG_PRINTOUT
                auto filename = MS_r.files[R.id].filenames[R.current_file];
                auto filenameid = R.current_file;
                DEBUG_PRINT_MPDBG("FileReadWorker::do_work open file: " << OUT(read_indx, write_indx, read_step, write_step, filename, filenameid));
                #endif
                if (R.file == nullptr) {//!err
                    PRINT_MPDBG("FileReadWorker::do_work: could not open file " << MS_r.files[R.id].filenames[R.current_file]);
                    R.current_file++;
                    continue;
                }
            }

            i64 remaining = r_size - total_read;
            i64 file_remaining = MS_r.files[R.id].lengths[R.current_file] - ftell(R.file);
            i64 to_read = std::min(remaining, file_remaining);
            if (to_read==0){//!err
                PRINT_MPDBG("FileReadWorker::do_work: to_read==0, id: "<<R.id<<", current_file: "<<R.current_file);
                throw std::runtime_error("FileReadWorker::do_work: to_read==0");
            }

            i64 actually_read = fread(W.stream_data.data() + buf_write + total_read, 1, to_read, R.file);
            if (actually_read!=to_read) {//!err
                PRINT_MPDBG("FileReadWorker::do_work: actually_read!=to_read: "<<actually_read<<"!="<<to_read<<", id: "<<R.id<<", current_file: "<<R.current_file);
                throw std::runtime_error("FileReadWorker::do_work: actually_read!=to_read");
            }
        
            batch_info.intervals.push_back({total_read, total_read + actually_read, R.current_file});
            // batch_info.fileIds.push_back(R.current_file);
            total_read += actually_read;

            if (ftell(R.file) == MS_r.files[R.id].lengths[R.current_file]) {
                #if DEBUG_PRINTOUT
                auto filename = MS_r.files[R.id].filenames[R.current_file];
                auto filenameid = R.current_file;
                DEBUG_PRINT_MPDBG("FileReadWorker::do_work close file: " << OUT(read_indx, write_indx, read_step, write_step, filename, filenameid));
                #endif
                fclose(R.file);
                R.file = nullptr;
                R.current_file++;
                if (R.current_file == MS_r.files[R.id].filenames.size()) {
                    batch_info.is_last = true;
                    break;
                }
            }
        }
        #if DEBUG_PRINTOUT
        auto filetellg = R.file!=nullptr?ftell(R.file):-1;
        DEBUG_PRINT_MPDBG("FileReadWorker::do_work end: " << OUT(read_indx, write_indx, read_step, write_step, total_read,batch_info.is_last, filetellg));
        #endif
        d_reader+=timeinsec(timenow()-start);
        n_reader++;
        return {total_read, total_read};
    }
};
class ParserClass{
    public:
    const i64 offset_w;
    const i64 offset_r;
    const i64 buf_read;
    ParseVectorBatch& write_batch;
    BatchFileInfo& write_batch_info;
    BatchFileBufInfo& read_batch_info;
    i64& seq_begin;
    i64& total_write;
    u64& curr_u64;
    FileBufStream& R;
    ParseStream& W;
    ParseState& state;
    ParserClass(i64 _offset_w, i64 _offset_r, i64 _buf_read, 
    ParseVectorBatch& _write_batch, BatchFileBufInfo& _read_batch_info, BatchFileInfo& _write_batch_info,
    i64& _seq_begin, i64& _total_write, u64& _curr_u64, FileBufStream& _R, ParseStream& _W, ParseState& _state):
        offset_w(_offset_w), offset_r(_offset_r), buf_read(_buf_read), 
        write_batch(_write_batch), read_batch_info(_read_batch_info), write_batch_info(_write_batch_info),
        seq_begin(_seq_begin), total_write(_total_write), curr_u64(_curr_u64),
        R(_R), W(_W), state(_state) {}
    inline bool is_header_begin(char c){
        return c == '@' || c == '>';
    };
    inline bool is_skip(char c){
        return c == '+';
    };
    inline bool is_newline(char c){
        return c == '\n';
    };
    inline i64 read_pos(i64 n){
        // return (buf_read+n)%FileBufStream::logic_batch_size;
        return buf_read+n;//never expected to be out of bounds
    }
    //sets chars[(offset_w+total_write)/chars_per_u64] to curr_u64
    inline void commit_current_u64(){
        write_batch.chars[(offset_w+total_write)/chars_per_u64] = curr_u64;
    }
    //!pos does not have automatically offset_w added
    inline void reload_current_u64(i64 pos){
        curr_u64 = write_batch.chars[pos/(chars_per_u64)];
        //only keep chars below offset_w, not inclusive
        curr_u64 &= bitmask_len((pos % chars_per_u64) * bits_per_char);
        write_batch.chars[pos/(chars_per_u64)] = curr_u64;
    }
    //moves .seps by 1 u64 and .bits by 1 bit, sets sep to new_pos; bit to is_true_sep  
    //also sets seq_begin to new_pos
    void commit_sep(i64 new_pos, bool is_true_sep) {
        if (write_batch.seps.size() % u64_bits == 0) {
            write_batch.bits.push_back(0);
        }
        set_bitvector_bit(write_batch.bits, write_batch.seps.size(), is_true_sep);
        write_batch.seps.push_back(offset_w + seq_begin);
        W.total_kmers += new_pos - seq_begin - is_true_sep*W.k_;
        seq_begin = new_pos;
    }
    void final_sep(){
        if(seq_begin!=total_write){//!sanity err
            std::string s=prints_new("ParserClass::final_sep: seq_begin!=total_write, seq_begin:",seq_begin,", total_write:",total_write);
            PRINT_MPDBG(s);
            throw std::runtime_error(s);
        }
        commit_sep(total_write,0);//is_true_sep does not matter, but 0 wouldn't change total_kmers
    }
    //adds char c as 2 bit char to curr_u64, commits curr_u64 if full, does total_write++
    //commits 0 sep if total_write-seq_begin>2*max_read_chars
    inline void add_char(char c){
        int num_char_offset = (offset_w+total_write) % chars_per_u64;
        curr_u64 |= static_cast<u64>(char_to_alphabet(c)) << (num_char_offset*bits_per_char);
        if (num_char_offset+1 == chars_per_u64) {
            //relies on flooring away the remainder
            commit_current_u64();
            curr_u64 = 0;
        }
        total_write++;
        //not just max_read_chars but *2 so that after committing sep, the remainder is still around max_read_chars
        if(total_write-seq_begin>=2*max_read_chars){
            commit_sep(seq_begin+max_read_chars,0);
        }
    }
    //if total_write-seq_begin>max_reader_chars, splits commit into a 0 and 1 sep
    //if >=k then just 1 sep
    //otherwise undoes total_write to seq_begin and clears curr_u64 with bitmask
    void finish_seq(){
        if(total_write-seq_begin>max_read_chars){
            //spread evenly
            commit_sep((seq_begin+total_write)/2,0);
            commit_sep(total_write,1);
        }else if(total_write-seq_begin>=k){
            commit_sep(total_write,1);
        }else{
            //rollback write
            //guaranteed to be safe because no non-true seps can be added before total is >max_read_chars>k
            //and after one is added, it stays >max_read_chars/2>k
            commit_current_u64();
            total_write=seq_begin;
            curr_u64=write_batch.chars[(offset_w+total_write)/chars_per_u64];
            // curr_u64&=~safe_lbitshift(full_u64_mask,((offset_w+total_write)%chars_per_u64)*bits_per_char);
            curr_u64&=bitmask_len(((offset_w+total_write)%chars_per_u64)*bits_per_char);
            // commit_current_u64();
        }
    }
    //do not leave empty interval at the end of a batch
    //!call before adding final_sep
    void remove_interval_if_empty(bool write_ended, bool read_ended, bool is_last){
        //only remove in case of !read_ended || write_ended || is_last
        //.size() points to the next added sep (given final_sep hasn't been added yet)
        write_batch_info.intervals.back().end=write_batch.seps.size();
        if((!read_ended || write_ended || is_last)
        && write_batch_info.intervals.back().length()<=0){
            write_batch_info.intervals.pop_back();
        }
    }
    bool interval_ended(i64 total_read){
        return read_batch_info.intervals[W.file_interval].end<=(offset_r+total_read);
    }

    //~assumes current interval is already last in read_batch_info
    void finish_interval(bool read_ended, bool write_ended, bool restored=false){
        // if(total_write-seq_begin>0){
        //     finish_seq();
        // }
        #if DEBUG_PRINTOUT
        auto fileid=read_batch_info.intervals[W.file_interval].file_id;
        auto totalids=read_batch_info.intervals.size();
        auto id_info=prints_new(fileid,"= (",W.file_interval+1,"/",totalids,")");
        DEBUG_PRINT_MPDBG("ParserClass::finish_interval: " << OUT(read_ended, write_ended, restored, id_info));
        #endif
        if(read_ended&&(W.file_interval+1!=read_batch_info.intervals.size())){//!err
            std::string s=prints_new("ParserClass::finish_interval: read_ended does not imply last interval: file_interval:",W.file_interval,", intervals.size:",read_batch_info.intervals.size());
            PRINT_MPDBG(s);
            throw std::runtime_error(s);
        }
        commit_current_u64();
        //not read_batch_info.is_last:
        //neither: no - inner interval in write batch
        //read_ended: yes - don't know if file continues, need to flush
        //write_ended: yes - file continues, need to flush
        //both: yes - don't know if file continues, need to flush
        
        //read_batch_info.is_last:
        //neither: no - inner interval in final batch
        //read_ended: no - definitely ended, space left in final batch
        //write_ended: yes - file continues, need to flush
        //both: no - definitely ended, space left in final batch
        bool test=(read_ended && !read_batch_info.is_last) || (!read_ended && write_ended);
        if(test){
            //unknown if file actually ended or continues later in case of read_ended,
            //need to flush to temp_buffer
            W.temp_buffer.clear();
            i64 len=total_write-seq_begin;
            W.num_chars_in_temp=len;

            //!debug
            // for(i64 i=0; i<write_batch.seps.size()-1; i++){
            //     PRINT_MPDBG(print_genome_vec_new(write_batch.chars, offset_w+write_batch.seps[i], offset_w+write_batch.seps[i+1]));
            // }
            // PRINT_MPDBG(print_genome_vec_new(write_batch.chars, offset_w+write_batch.seps.back(), offset_w+seq_begin));
            // PRINT_MPDBG(prints_new("undoing:",print_genome_vec_new(write_batch.chars, offset_w+seq_begin, offset_w+total_write)));

            if(len>0){
                W.temp_buffer.resize(ceil_div(len*bits_per_char,64));
                copy_bitvector(write_batch.chars, W.temp_buffer, (offset_w+seq_begin)*bits_per_char, len*bits_per_char, 0);
                // PRINT_MPDBG(prints_new("in temp:",print_genome_vec_new(W.temp_buffer, 0, len)));
                W.temp_file_id=write_batch_info.intervals.back().file_id;
                total_write=seq_begin;
            }
            //technically can only remove if write_ended
            remove_interval_if_empty(write_ended,read_ended,read_batch_info.is_last);
            if(write_ended){
                final_sep();
            }
        }else{//internal or last interval in stream
            //not test=(!read_ended || is_last) && (read_ended || !write_ended)
            //=(read_ended && is_last) || (!read_ended && !write_ended)
            finish_seq();
            remove_interval_if_empty(write_ended,read_ended,read_batch_info.is_last);
            if(!restored){//otherwise should keep the current interval
                W.file_interval++;
            }
            if(read_ended&&read_batch_info.is_last){//last interval and last batch
                final_sep();
                write_batch_info.is_last=true;
                DEBUG_PRINT_MPDBG("ParserClass::finish_interval: last interval and last batch");
            }else{//(!read_ended&&!write_ended){//inner, need to start new interval
                //=having no final_sep, .size() points to the next added sep
                // write_batch_info.intervals.push_back(
                //     {write_batch.seps.size(),write_batch.seps.size(),
                //     read_batch_info.intervals[W.file_interval].file_id});
                add_interval(write_batch.seps.size(),write_batch.seps.size(),
                read_batch_info.intervals[W.file_interval].file_id);
                state=NEW_FILE;
            }
        }
        if(write_ended||(read_ended && read_batch_info.is_last)){//sanity
            i64 num_seps=write_batch.seps.size();
            if(num_seps>=2){
                i64 last=write_batch.seps[num_seps-1];
                if(last!=seq_begin&&last!=total_write){//!err
                    std::string s=prints_new("ParserClass::finish_interval: last, seq_begin and total_write do not match, last:",last,", seq_begin:",seq_begin,", total_write:",total_write);
                }
            }else{//!err
                std::string s=prints_new("ParserClass::finish_interval: too little seps on end, num_seps:",num_seps);
                PRINT_MPDBG(s);
                throw std::runtime_error(s);
            }
        }
    }
    //start inclusive, end exclusive (usually start=end since updated later anyway), both indexes on seps
    //file_id is the file_id of the interval 
    void add_interval(i64 start, i64 end, i64 file_id){
        write_batch_info.intervals.push_back({start,end,file_id});
    }
};

class FileBufWorker: public MS_Worker<FileBufWorker,FileBufMS,ParseMS>{
    public:
    FileBufWorker(FileBufMS& _MS_r, ParseMS& _MS_w):
        MS_Worker(_MS_r, _MS_w,FileBufStream::logic_batch_size,ParseStream::logic_batch_size) {}
    std::pair<i64,i64> do_work(
        i32 read_indx, i32 write_indx, i64 r_size, i64 buf_read, i64 buf_write,
        i64 read_step, i64 write_step,  bool is_first_child, bool is_last_parent){
        auto start=timenow();
        DEBUG_PRINT_MPDBG("FileBufWorker::do_work begin: " << OUT(read_indx, write_indx, r_size, buf_read, buf_write, read_step, write_step, is_first_child, is_last_parent));
        //= reading terminates on either R or W end
        auto& R = MS_r.data[read_indx];
        auto& W = MS_w.data[write_indx];
        //during read of char, points to 0-index of 8bit char, otherwise current length read
        i64 total_read = 0;
        //during write of char, points to 0-index of 2-bit char, otherwise current length written
        i64 total_write = 0;

        //no guarantee that the buffer is aligned to the batch size
        i64 read_batch_index = buf_read / FileBufStream::logic_batch_size;
        i64 write_batch_index = buf_write / ParseStream::logic_batch_size;
        //=align writing to batch size, avoids extra logic
        // i64 next_write_batch_index = (write_batch_index + 1) % R.batch_info.size();

        //has FileIntervals (start,end,file_id) intervals vec, is_last bool
        BatchFileBufInfo& read_batch_info = R.batch_info[read_batch_index];

        //has FileIntervals (start,end,file_id) intervals vec, i64 id, is_first/is_last bool
        BatchFileInfo& write_batch_info = W.batch_info[write_batch_index];

        //chars bitvec, seps vec, bits bitvec, rank bitvec, i64 num_packed_chars
        ParseVectorBatch& write_batch = W.batches[write_batch_index];

        //=align reading to batch size, use temp_buffer to store end of previous batch        
        // i64 next_read_batch_index = (read_batch_index + 1) % R.batch_info.size();
        // bool has_next=!(read_batch_info.is_last)&&(buf_read%FileBufStream::logic_size!=0);
        
        //buf_write within the write batch, in 2 bit chars, so 32 offset_w per u64
        i64 offset_w = buf_write % ParseStream::logic_batch_size;
        //used to reset file_interval and translate total_read into read_pos (for interval end checking)
        i64 offset_r = buf_read % FileBufStream::logic_batch_size;
        //current working u64 that is logically at (offset_w+total_write)/chars_per_u64
        //or (offset_w+total_write)*bits_per_char/64



        i64 remaining = ParseStream::logic_batch_size - offset_w-max_read_chars*3;
        
        if (remaining <= 0) {//!err
            PRINT_MPDBG("FileBufWorker::do_work: remaining<=0: "<<remaining);
            throw std::runtime_error("FileBufWorker::do_work: remaining<=0");
        }
        //= align with read batch size
        i64 remaining_read = min(r_size, static_cast<i64>(FileBufStream::logic_batch_size - offset_r));
        if (remaining_read <= 0) {//!err
            PRINT_MPDBG("FileBufWorker::do_work: remaining_read<=0: "<<remaining);
            throw std::runtime_error("FileBufWorker::do_work: remaining<=0");
        }
        
        //if aligned, clear write
        if (offset_w == 0) {
            write_batch_info.reset();
            write_batch.reset();
            // W.num_to_search=0;
            W.total_kmers=0;
        }
        //if aligned, new read, reset file_interval
        if (offset_r == 0) {
            W.file_interval=0;
        }

        if(W.file_interval>=read_batch_info.intervals.size()){//!err
            std::string s=prints_new("FileBufWorker::do_work: W.file_interval>=read_batch_info.intervals.size(), file_interval:",W.file_interval,", intervals.size:",read_batch_info.intervals.size());
            PRINT_MPDBG(s);
            throw std::runtime_error(s);
        }
        if(read_batch_info.intervals.back().end<offset_r+remaining_read){//!err
            std::string s=prints_new("FileBufWorker::do_work: read_batch_info.intervals.back().end<offset_r+remaining_read, end:",read_batch_info.intervals.back().end,", offset_r:",offset_r,", remaining_read:",remaining_read);
            PRINT_MPDBG(s);
            throw std::runtime_error(s);
        }

        FileInterval curr_interval=read_batch_info.intervals[W.file_interval];
        //parse the FASTQ/FASTA etc file in the read buffer
        ParseState state = W.parse_state;
        //inclusive, left, in write
        //used for rollback in case of incomplete read

        u64 curr_u64 = 0;
        i64 seq_begin = 0;
        ParserClass P(offset_w,offset_r,buf_read,
            write_batch,read_batch_info, write_batch_info,
            seq_begin,total_write,curr_u64,
            R,W,state);

        //technically total_write should be 0 here
        P.reload_current_u64(offset_w+total_write);

        //load temp if not empty
        if(W.num_chars_in_temp > 0){
            copy_bitvector(W.temp_buffer,write_batch.chars,
                0,W.num_chars_in_temp*bits_per_char,offset_w*bits_per_char);
            // PRINT_MPDBG(prints_new("copied over:",print_genome_vec_new(write_batch.chars, offset_w, offset_w+W.num_chars_in_temp)));//!debug
            total_write+=W.num_chars_in_temp;

            //!forgot to load new curr_u64 previously
            P.reload_current_u64(offset_w+total_write);

            if(write_batch_info.intervals.size()==0){
                //means also new write
                if(offset_w!=0){//!err
                    std::string s=prints_new("FileBufWorker::do_work: different file_id in temp_buffer while not at start, offset_w:",offset_w,", temp_file_id:",W.temp_file_id,", file_id:",curr_interval.file_id);
                    PRINT_MPDBG(s);
                    throw std::runtime_error(s);
                }
                P.add_interval(0, 0, W.temp_file_id);
            }else if(write_batch_info.intervals.back().file_id!=W.temp_file_id){//!err
                std::string s=prints_new("FileBufWorker::do_work: different file_id in temp_buffer, file_id:",write_batch_info.intervals.back().file_id,", temp_file_id:",W.temp_file_id);
                PRINT_MPDBG(s);
                throw std::runtime_error(s);
            }

            if(W.temp_file_id!=curr_interval.file_id){
                P.finish_interval(false,false,true);
            }
            //no need to set state here, it is done earlier and is overwritten if file_ids differ
            W.num_chars_in_temp=0;
        }else if (offset_w == 0) {
            if (write_batch_info.intervals.size() != 0){//sanity
                std::string s=prints_new("FileBufWorker::do_work: write_batch_info.intervals.size()!=0 with 0 offset_w:",write_batch_info.intervals.size());
                PRINT_MPDBG(s);
                throw std::runtime_error(s);
            }
            P.add_interval(0, 0, curr_interval.file_id);//must add interval if new write
        }else{
            if (write_batch_info.intervals.size() == 0){//sanity
                std::string s=prints_new("FileBufWorker::do_work: write_batch_info.intervals.size()==0 with non-zero offset_w:",write_batch_info.intervals.size(),", offset_w:",offset_w);
                PRINT_MPDBG(s);
                throw std::runtime_error(s);
            }
        }


        bool read_ended=false;
        bool write_ended=false;
        while (total_read<remaining_read
        && total_write<remaining
        && write_batch.seps.size()<write_batch.seps.capacity()-8){
            if (P.interval_ended(total_read)) {
                P.finish_interval(false,false);//guaranteed does not modify total_write or total_read
                // if(W.file_interval>=read_batch_info.intervals.size()){
                //     break;
                // }
                curr_interval=read_batch_info.intervals[W.file_interval];
                continue;
            }
            char c = R.stream_data[P.read_pos(total_read)];
            if (state == NEW_FILE) {//skip until first header
                state=BEGIN_FROM_SKIP;
            }
            if(state==BEGIN_FROM_SKIP||state==BEGIN_FROM_NEW_READ||state==BEGIN_FROM_READ){
                if(P.is_header_begin(c)){
                    if(state==BEGIN_FROM_READ){
                        P.finish_seq();
                    }
                    state=SKIP_THEN_READ;
                }else if(P.is_skip(c)){
                    if(state==BEGIN_FROM_READ){
                        P.finish_seq();
                    }
                    state=SKIP_THEN_SKIP;//necessary instead of SKIP because next quality line can have @
                }else if(P.is_newline(c)){
                    //do nothing, state does not change, empty lines ignored
                }else if(state==BEGIN_FROM_NEW_READ||state==BEGIN_FROM_READ){
                    state=READ;
                    P.add_char(c);
                    // if(write_step==1){
                    //     std::cout<<c;
                    // }
                }else{//begin from skip
                    state=SKIP;
                }
            }else if(state==SKIP||state==SKIP_THEN_READ||state==SKIP_THEN_SKIP||state==SKIP_UNTIL_CHAR){
                if(P.is_newline(c)){
                    if(state==SKIP){
                        state=BEGIN_FROM_SKIP;
                    }else if(state==SKIP_THEN_READ){
                        state=BEGIN_FROM_NEW_READ;
                    }else if(state==SKIP_THEN_SKIP){
                        state=SKIP_UNTIL_CHAR;
                    }else{//SKIP_UNTIL_CHAR
                    }
                }else if(state==SKIP_UNTIL_CHAR){
                    state=SKIP;
                }
            }else{//READ
                if(P.is_newline(c)){
                    state=BEGIN_FROM_READ;
                }else{
                    P.add_char(c);
                }
                // if(write_step==1){
                //     std::cout<<c;
                // }
            }
            total_read++;
        }
        if(total_read==remaining_read){
            read_ended=true;
        }
        if(total_read>remaining_read){//!sanity
            std::string s=prints_new("FileBufWorker::do_work: total_read>remaining_read, total_read:",total_read,", r_size:",r_size,", remaining_read:",remaining_read);
            PRINT_MPDBG(s);
            throw std::runtime_error(s);
        }
        if((total_write>=remaining) || (write_batch.seps.size()>=write_batch.seps.capacity()-8)){
            write_ended=true;
        }
        if(!(read_ended||write_ended)){//!err
            std::string s=prints_new("FileBufWorker::do_work: !read_ended&&!write_ended, total_read:",total_read,", r_size:",r_size,", total_write:",total_write,", remaining:",remaining,", seps.size:",write_batch.seps.size());
            PRINT_MPDBG(s);
            throw std::runtime_error(s);
        }
        P.finish_interval(read_ended,write_ended);
        if(write_ended||(read_ended && read_batch_info.is_last)){
            //build the rank bitvector
            PoppySmall<OffsetVector<u64>,OffsetVector<u64>> poppy(write_batch.rank);
            if(write_batch.rank.capacity()==0){//!err
                std::string s=prints_new("FileBufWorker::do_work: write_batch.rank.capacity()==0");
                PRINT_MPDBG(s);
                throw std::runtime_error(s);
            }
            poppy.build(write_batch.bits,write_batch.seps.size());
            //~sanity
            if((write_batch.seps.back()-(poppy.total_1s)*k)!=W.total_kmers){//!err
                std::string s=prints_new("FileBufWorker::do_work: total_kmers does not match rank, total_kmers:",W.total_kmers,", rank:",poppy.total_1s,", k:",k,", seps.back:",write_batch.seps.back(),", total:",write_batch.seps.back()-(poppy.total_1s)*k);
                PRINT_MPDBG(s);
                throw std::runtime_error(s);
            }
            //align total_write to batch size
            total_write=ParseStream::logic_batch_size-(offset_w);
            if((total_write+buf_write)%FileBufStream::logic_batch_size!=0){//!err
                std::string s=prints_new("FileBufWorker::do_work: total_write not aligned to batch size, total_write:",total_write,", buf_write:",buf_write);
                PRINT_MPDBG(s);
                throw std::runtime_error(s);
            }
        }
        if(write_batch_info.is_last){//!set ended in R
            R.ended=true; 
        }
        W.parse_state=state;
        DEBUG_PRINT_MPDBG("FileBufWorker::do_work end: " << OUT(read_indx, write_indx, read_step, write_step, total_read, total_write,write_batch_info.is_last));
        d_parser+=timeinsec(timenow()-start);
        n_parser++;
        return {total_read, total_write};
    }
};
class DebugParseWorker: public MS_Worker<DebugParseWorker,ParseMS,DebugWriteBufMS>{
    public:
    DebugParseWorker(ParseMS& _MS_r, DebugWriteBufMS& _MS_w):
        MS_Worker(_MS_r, _MS_w,ParseStream::logic_batch_size,DebugWriteBufStream::logic_batch_size) {}
    std::pair<i64,i64> do_work(
        i32 read_indx, i32 write_indx, i64 r_size, i64 buf_read, i64 buf_write,
        i64 read_step, i64 write_step,  bool is_first_child, bool is_last_parent){
        auto start=timenow();
        DEBUG_PRINT_MPDBG("DebugParseWorker::do_work begin: " << OUT(read_indx, write_indx, r_size, buf_read, buf_write, read_step, write_step, is_first_child, is_last_parent));
        //= reading terminates on either R or W end
        auto& R = MS_r.data[read_indx];
        auto& W = MS_w.data[write_indx];

        i64 total_read = 0;
        i64 total_write = 0;

        i64 read_batch_index = buf_read / ParseStream::logic_batch_size;
        i64 write_batch_index = buf_write / DebugWriteBufStream::logic_batch_size;

        BatchFileInfo& read_batch_info = R.batch_info[read_batch_index];
        ParseVectorBatch& read_batch = R.batches[read_batch_index];

        BatchFileBufInfo& write_batch_info = W.batch_info[write_batch_index];


        i64 offset_r = buf_read % ParseStream::logic_batch_size;
        i64 offset_w = buf_write % DebugWriteBufStream::logic_batch_size;
        if(offset_w!=0 || offset_r!=0){//!err
            std::string s=prints_new("DebugParseWorker::do_work: offset_w or offset_r not aligned to batch size, offset_w:",offset_w,", offset_r:",offset_r);
            PRINT_MPDBG(s);
            throw std::runtime_error(s);
        }

        write_batch_info.reset();
        write_batch_info.is_last=read_batch_info.is_last;
        // i64 max_read=
        if(read_batch_info.intervals.size()==0){//!err
            std::string s=prints_new("DebugParseWorker::do_work: read_batch_info.intervals.size()==0");
            PRINT_MPDBG(s);
            throw std::runtime_error(s);
        }
        for(int i=0;i<read_batch_info.intervals.size();i++){
            FileSepInterval& curr_interval=read_batch_info.intervals[i];
            //push new interval
            //end will be overwritten later
            write_batch_info.intervals.push_back({total_write,total_write,curr_interval.file_id});
            for(i64 j=curr_interval.start;j<curr_interval.end;j++){
                if(read_batch.seps.size()<=curr_interval.end){//!err
                    std::string s=prints_new("DebugParseWorker::do_work: read_batch.seps.size()<=curr_interval.end, seps.size:",read_batch.seps.size(),", end:",curr_interval.end);
                    PRINT_MPDBG(s);
                    throw std::runtime_error(s);
                }
                i64 from=read_batch.seps[j];
                i64 to=read_batch.seps[j+1];
                bool newline=get_bitvector_bit(read_batch.bits,j);//at the end
                for(i64 k=from;k<to;k++){
                    if(total_write>=DebugWriteBufStream::logic_batch_size){//!err
                        std::string s=prints_new("DebugParseWorker::do_work: total_write>=DebugWriteBufStream::logic_batch_size, total_write:",total_write);
                        PRINT_MPDBG(s);
                        throw std::runtime_error(s);
                    }
                    W.stream_data[buf_write+total_write]=
                        alphabet_to_char(static_cast<Alphabet>(access_alphabet_val(read_batch.chars,k)));
                    total_write++;
                }
                if(newline){
                    if(total_write>=DebugWriteBufStream::logic_batch_size){//!err
                        std::string s=prints_new("DebugParseWorker::do_work: total_write>=DebugWriteBufStream::logic_batch_size, total_write:",total_write);
                        PRINT_MPDBG(s);
                        throw std::runtime_error(s);
                    }
                    W.stream_data[buf_write+total_write]='\n';
                    total_write++;
                }
            }
            write_batch_info.intervals.back().end=total_write;
        }
        DEBUG_PRINT_MPDBG("DebugParseWorker::do_work end: " << OUT(read_indx, write_indx, read_step, write_step, total_read, total_write,write_batch_info.is_last));
        d_decoder+=timeinsec(timenow()-start);
        n_decoder++;
        return {r_size, DebugWriteBufStream::logic_batch_size};
    }
};
class DebugWriteBufWorker: public MS_Worker<DebugWriteBufWorker,DebugWriteBufMS,FileOutMS>{
    public:
    DebugWriteBufWorker(DebugWriteBufMS& _MS_r, FileOutMS& _MS_w):
        MS_Worker(_MS_r, _MS_w,DebugWriteBufStream::logic_batch_size,FileOutStream::logic_batch_size) {}
    std::pair<i64,i64> do_work(
        i32 read_indx, i32 write_indx, i64 r_size, i64 buf_read, i64 buf_write,
        i64 read_step, i64 write_step,  bool is_first_child, bool is_last_parent){
        auto start=timenow();
        DEBUG_PRINT_MPDBG("DebugWriteBufWorker::do_work begin: " << OUT(read_indx, write_indx, r_size, buf_read, buf_write, read_step, write_step, is_first_child, is_last_parent));
        //= reading terminates on either R or W end
        auto& R = MS_r.data[read_indx];
        auto& W = MS_w.data[write_indx];

        i64 total_read = 0;

        i64 read_batch_index = buf_read / DebugWriteBufStream::logic_batch_size;

        BatchFileBufInfo& read_batch_info = R.batch_info[read_batch_index];
        FILE*& file = W.file;
        i64& fileid = W.fileid;

        i64 offset_r = buf_read % DebugWriteBufStream::logic_batch_size;
        if(offset_r!=0){//!err
            std::string s=prints_new("DebugWriteBuf::do_work: offset_w or offset_r not aligned to batch size, offset_r:",offset_r);
            PRINT_MPDBG(s);
            throw std::runtime_error(s);
        }

        // i64 max_read=
        if(read_batch_info.intervals.size()==0){//!err
            std::string s=prints_new("DebugWriteBuf::do_work: read_batch_info.intervals.size()==0");
            PRINT_MPDBG(s);
            throw std::runtime_error(s);
        }
        if(MS_w.files.size()<=W.id){//!err
            std::string s=prints_new("DebugWriteBufWorker::do_work: MS_w.files.size()<=W.id, size:",MS_w.files.size(),", id:",W.id);
            PRINT_MPDBG(s);
            throw std::runtime_error(s);
        }
        for(int i=0;i<read_batch_info.intervals.size();i++){
            FileInterval& curr_interval=read_batch_info.intervals[i];
            if(fileid!=curr_interval.file_id){
                if(file!=nullptr){
                    DEBUG_PRINT_MPDBG("DebugWriteBufWorker::do_work close file: " << OUT(read_indx, write_indx, read_step, write_step, fileid));
                    fclose(file);
                }
                if(MS_w.files[W.id].output_filenames.size()<=curr_interval.file_id){//!err
                    std::string s=prints_new("DebugWriteBufWorker::do_work: MS_w.files[W.id].output_filenames.size()<=curr_interval.file_id, size:",MS_w.files[W.id].output_filenames.size(),", file_id:",curr_interval.file_id);
                    PRINT_MPDBG(s);
                    throw std::runtime_error(s);
                }
                file = fopen(MS_w.files[W.id].output_filenames[curr_interval.file_id].c_str(), "wb");
                if (file == nullptr) {//!err
                    std::string s=prints_new("DebugWriteBufWorker::do_work: could not open file ",MS_w.files[W.id].output_filenames[curr_interval.file_id]);
                    PRINT_MPDBG(s);
                    throw std::runtime_error(s);
                }
                fileid=curr_interval.file_id;
                DEBUG_PRINT_MPDBG("DebugWriteBufWorker::do_work open file: " << OUT(read_indx, write_indx, read_step, write_step, fileid));
            }
            //copy the entire range at once
            u8* data=R.stream_data.data()+buf_read+curr_interval.start;
            auto success=fwrite(data,1,curr_interval.length(),file);
            if(success!=curr_interval.length()){//!err
                std::string s=prints_new("DebugWriteBufWorker::do_work: fwrite failed, success:",success,", length:",curr_interval.length());
                PRINT_MPDBG(s);
                throw std::runtime_error(s);
            }
        }
        //close if last
        if(read_batch_info.is_last){
            if(file!=nullptr){
                fclose(file);
                DEBUG_PRINT_MPDBG("DebugWriteBufWorker::do_work close file: " << OUT(read_indx, write_indx, read_step, write_step, fileid));
            }
            R.ended=true;
        }
        DEBUG_PRINT_MPDBG("DebugWriteBufWorker::do_work end: " << OUT(read_indx, write_indx, read_step, write_step, total_read));
        d_writer+=timeinsec(timenow()-start);
        n_writer++;
        return {r_size, FileOutStream::logic_batch_size};
    }
};
}//namespace sbwt_lcs_gpu
