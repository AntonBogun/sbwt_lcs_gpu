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





constexpr u64 poppysmall_from_bitvector_u64s_const(u64 num_u64s);
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
    // static const u64 file_buf = 0;
    // static u64 parse;
    // static u64 multiplex;
    // static u64 demultiplex;
    // static u64 write_buf;
    static u64 total;
    // static u64 gpu; //gpu allocated separately
};




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


// using FileNameVec = std::vector<std::vector<std::tuple<std::string,std::string,i64>>>;
class FileReadStream: public CB_data<FileReadStream>{
    public:
    i64 current_file=0;
    FILE* file=nullptr;
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
    std::vector<StreamFilenamesContainer> files;
    i64 current_index=0;
    // std::string dump_impl(){
    //     std::stringstream ss;
    //     ss <<" gen_num "<<gen_num
    //     << ", gen_max "<<gen_max;
    //     return ss.str();
    // }
    FileReadMS(std::vector<StreamFilenamesContainer>&& _files):
        SharedThreadMultistream(0, num_physical_streams, num_physical_streams, 0, 1, 0),
        files(std::move(_files)) {}
    void allocate_impl(i32 stream_indx, i64 id){
        // data[stream_indx].gen_num=gen_num;
        data[stream_indx].id=current_index;
        data[stream_indx].size=files[current_index].total_length;
        data[stream_indx].v_r=data[stream_indx].size;
        data[stream_indx].v_w=0;
        current_index++;
    }
    void deallocate_impl(i32 stream_indx){
        //nothing to do
        // PRINT("gen_MS_impl::deallocate_impl " << std::this_thread::get_id() << " sid " << stream_indx);
        // if(gen_num==gen_max){
        //     final_dealloc();
        // }
    }
    bool can_allocate_impl(){
        // return gen_num<gen_max;
        return current_index<files.size();
    }
    bool no_write_can_read_impl(i32 stream_indx){
        return data[stream_indx].v_r>0;
    }
    bool no_write_ended_impl(i32 stream_indx, i64 buf_S, i64 r_size){
        return data[stream_indx].v_r<=0;
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
    static const u64 batch_u64s = batch_buf_u64s;
    static const u64 u64s=batches_in_stream*batch_u64s;
    OffsetVector<char> stream_data;
    std::vector<BatchFileBufInfo> batch_info;
    bool ended=false;
    FileBufStream(i64 _size, i32 _max_writers, i32 _max_readers, OffsetVector<char>&& _data):
        CB_data(_size, _max_writers, _max_readers),
        stream_data(std::move(_data)), batch_info(batches_in_stream) {}
};
class FileBufMS: public SharedThreadMultistream<FileBufMS,FileBufStream>{
    public:
    static u64 u64s; //depends on number of streams
    static const u64 data_offset = 0;//in u64s
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
        // section_data(std::move(_section_data)) {}
        section_data(memory, data_offset, u64s) {}

    void emplace_back_data(i64 _1, i32 _2, i32 _3){
        OffsetVector<char> temp(section_data, data.size()*FileBufStream::u64s, FileBufStream::u64s);
        // data.back().stream_data.resize(FileBufStream::u64s*sizeof(u64));
        temp.resize(FileBufStream::u64s*sizeof(u64));
        data.emplace_back(FileBufStream::u64s*sizeof(u64), 1, 1, std::move(temp));
    }
    void allocate_impl(i32 stream_indx, i64 id){
        data[stream_indx].ended=false;
    }
    void deallocate_impl(i32 stream_indx){}
    i64 get_write_id_impl(i32 stream_indx){return data[stream_indx].id;}
    bool sequential_ended_impl(i32 stream_indx,i64 buf_S, i64 r_size){return data[stream_indx].ended;}
    bool get_is_last_child_impl(i32 stream_indx, i64 buf_E, bool is_last_parent){return is_last_parent;}
    bool get_is_first_parent_impl(i32 stream_indx, bool is_begin){return is_begin;}
};
class ParseStream: public CB_data<ParseStream>{
    public:
    static const u64 chars_batch_start=0;
    static const u64 chars_batch_u64s = batch_buf_u64s;

    static const u64 seps_batch_start = chars_batch_start + chars_batch_u64s;
    static const u64 seps_batch_u64s = ceil_div_const(chars_batch_u64s * u64_bits / bits_per_char, max_read_chars) + 2;

    static const u64 seps_bitvector_batch_start = seps_batch_start + seps_batch_u64s;
    static const u64 seps_bitvector_batch_u64s = ceil_div_const(seps_batch_u64s, u64_bits) + bitvector_pad_u64s;

    static const u64 seps_rank_batch_start = seps_bitvector_batch_start + seps_bitvector_batch_u64s;
    static const u64 seps_rank_batch_u64s = poppysmall_from_bitvector_u64s_const(seps_bitvector_batch_u64s);

    static const u64 batch_u64s = chars_batch_u64s + seps_batch_u64s + seps_bitvector_batch_u64s + seps_rank_batch_u64s;
    static const u64 u64s = batch_u64s * batches_in_stream;
    OffsetVector<u64> stream_data;
    std::vector<BatchFileInfo> batch_info;
    FCVector<ParseVectorBatch> batches;
    ParseStream(i64 _size, i32 _max_writers, i32 _max_readers, OffsetVector<u64>&& _data):
        CB_data(_size, _max_writers, _max_readers),
        stream_data(std::move(_data)), batch_info(batches_in_stream), batches(batches_in_stream) {
            for(i32 i=0;i<batches_in_stream;i++){
                batches.emplace_back(ParseVectorBatch{
                    OffsetVector<char>(stream_data, i*batch_u64s+chars_batch_start, chars_batch_u64s),
                    OffsetVector<u64>(stream_data,  i*batch_u64s+seps_batch_start, seps_batch_u64s),
                    OffsetVector<u64>(stream_data,  i*batch_u64s+seps_bitvector_batch_start, seps_bitvector_batch_u64s),
                    OffsetVector<u64>(stream_data,  i*batch_u64s+seps_rank_batch_start, seps_rank_batch_u64s)
                });
                // batches.back().chars.resize(chars_batch_u64s*sizeof(u64)/sizeof(char));
                // batches.back().seps.resize(seps_batch_u64s);
                // batches.back().bits.resize(seps_bitvector_batch_u64s);
                // batches.back().rank.resize(seps_rank_batch_u64s);
            }
        }
};
class ParseMS: public SharedThreadMultistream<ParseMS,ParseStream>{
    public:
    static u64 u64s;
    static u64 data_offset;
    static constexpr bool has_write=true;
    static constexpr bool has_read=true;
    static constexpr StreamType stream_type=PARALLEL;
    static i32 num_threads;
    OffsetVector<u64> section_data;
    ParseMS(std::vector<u64>& memory):
        SharedThreadMultistream(0, num_physical_streams, num_physical_streams, num_physical_streams, 1, 1),
        section_data(memory, data_offset, u64s) {}
    void emplace_back_data(i64 _1, i32 _2, i32 _3){
        OffsetVector<u64> temp(section_data, data.size()*ParseStream::u64s, ParseStream::u64s);
        // data.back().stream_data.resize(ParseStream::u64s);
        temp.resize(ParseStream::u64s);
        data.emplace_back(ParseStream::u64s, 1, 1, //no reason to use more than 1 reader?
        std::move(temp));
    }
    void allocate_impl(i32 stream_indx, i64 id){
        // data[stream_indx].id=id;
    }
    void deallocate_impl(i32 stream_indx){}
    i64 get_write_id_impl(i32 stream_indx){return 0;}//multiplex into one stream
    // bool sequential_ended_impl(i32 stream_indx,i64 buf_S, i64 r_size){return 0;}
    bool get_is_last_child_impl(i32 stream_indx, i64 buf_E, bool is_last_parent){return is_last_parent;}
    bool get_is_first_parent_impl(i32 stream_indx, bool is_begin){return is_begin;}
};

class MultiplexStream: public CB_data<MultiplexStream>{
    public:
    static const u64 chars_batch_section_start = 0;
    static const u64 chars_batch_section_u64s = ParseStream::chars_batch_u64s * gpu_batch_mult;

    static const u64 seps_batch_section_start = chars_batch_section_start + chars_batch_section_u64s;
    static const u64 seps_batch_section_u64s = ParseStream::seps_batch_u64s * gpu_batch_mult;

    static const u64 seps_bitvector_batch_section_start = seps_batch_section_start + seps_batch_section_u64s;
    static const u64 seps_bitvector_batch_section_u64s = ParseStream::seps_bitvector_batch_u64s * gpu_batch_mult;

    static const u64 seps_rank_batch_section_start = seps_bitvector_batch_section_start + seps_bitvector_batch_section_u64s;
    static const u64 seps_rank_batch_section_u64s = ParseStream::seps_rank_batch_u64s * gpu_batch_mult;

    static const u64 thread_lookup_vector_start = seps_rank_batch_section_start + seps_rank_batch_section_u64s;
    static const u64 thread_lookup_vector_u64s = sizeof(GPUThreadLookupTableEntry) * gpu_batch_mult / sizeof(u64);

    static const u64 batch_section_u64s = chars_batch_section_u64s + seps_batch_section_u64s + seps_bitvector_batch_section_u64s + seps_rank_batch_section_u64s + thread_lookup_vector_u64s;
    static const u64 u64s = batch_section_u64s * batches_in_gpu_stream; //~for now do not scale batches with number of streams

    OffsetVector<u64> stream_data;
    GpuPointer<u64> stream_gpu_data;
    std::vector<BatchFileInfo> batch_info;
    FCVector<ParseVectorBatch> batches;
    FCVector<OffsetVector<GPUThreadLookupTableEntry>> lookup_tables;
    FCVector<GpuPointer<u64>> gpu_batches;
    MultiplexStream(i64 _size, i32 _max_writers, i32 _max_readers, OffsetVector<u64>&& _data, GpuPointer<u64>&& _gpu_data);
};
struct GPUSection {
    static const u64 in_batch_u64s = MultiplexStream::batch_section_u64s;
    static const u64 out_batch_u64s = MultiplexStream::chars_batch_section_u64s * u64_bits / bits_per_char+100;//1 u64 per char + padding
    static const u64 in_u64s = in_batch_u64s * batches_in_gpu_stream; //~for now do not scale batches with number of streams
    static const u64 out_u64s = out_batch_u64s * batches_in_gpu_stream; //~for now do not scale batches with number of streams
    
    static const u64 start_in = 0;
    static const u64 start_out = in_u64s;

    static const u64 u64s = in_u64s + out_u64s; //~for now do not scale batches with number of streams
};
MultiplexStream::MultiplexStream(i64 _size, i32 _max_writers, i32 _max_readers, OffsetVector<u64>&& _data, GpuPointer<u64>&& _gpu_data):
        CB_data(_size, _max_writers, _max_readers),
        stream_data(std::move(_data)), batch_info(_size), stream_gpu_data(std::move(_gpu_data)), 
        batches(gpu_batch_mult*batches_in_gpu_stream), lookup_tables(batches_in_gpu_stream), gpu_batches(batches_in_gpu_stream){
            for(i32 i=0;i<batches_in_gpu_stream;i++){
                for(i32 j=0;j<gpu_batch_mult;j++){
                    batches.emplace_back(ParseVectorBatch{
                        OffsetVector<char>(stream_data, i*batch_section_u64s+chars_batch_section_start+
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
    static const u64 u64s = MultiplexStream::u64s;//~for now do not scale batches with number of streams
    static u64 data_offset;
    OffsetVector<u64> section_data;
    GpuPointer<u64> gpu_data;
    i64 total_num_streams;
    MultiplexMS(std::vector<u64>& memory, GpuPointer<u64>& _gpu_data, i64 _total_num_streams):
        SharedThreadMultistream(0, 1, gpu_readers, num_physical_streams,  gpu_readers, num_physical_streams),
        section_data(memory, data_offset, u64s), gpu_data(_gpu_data, GPUSection::start_in, GPUSection::in_u64s),
        total_num_streams(_total_num_streams) {}
    void emplace_back_data(i64 _1, i32 _2, i32 _3){
        OffsetVector<u64> temp(section_data, data.size()*MultiplexStream::u64s, MultiplexStream::u64s);
        // data.back().stream_data.resize(MultiplexStream::u64s);
        temp.resize(MultiplexStream::u64s);
        data.emplace_back(gpu_batch_mult*batches_in_gpu_stream, num_physical_streams, gpu_readers,
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
    static const u64 indexes_batch_u64s = GPUSection::out_batch_u64s;
    static const u64 u64s = indexes_batch_u64s * batches_in_gpu_stream; //~for now do not scale batches with number of streams
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
    static const u64 u64s=DemultiplexStream::u64s;//~for now do not scale batches with number of streams
    static u64 data_offset;
    static constexpr bool has_write=true;
    static constexpr bool has_read=true;
    static constexpr StreamType stream_type=PARALLEL;
    static i32 num_threads;
    OffsetVector<u64> section_data;
    GpuPointer<u64> gpu_data;
    DemultiplexMS(std::vector<u64>& memory, GpuPointer<u64>& _gpu_data):
        SharedThreadMultistream(0, 1, num_physical_streams, gpu_readers,  num_physical_streams, gpu_readers),
        section_data(memory, data_offset, u64s), gpu_data(_gpu_data, GPUSection::start_out, GPUSection::out_u64s) {}
    void emplace_back_data(i64 _1, i32 _2, i32 _3){
        OffsetVector<u64> temp(section_data, data.size()*DemultiplexStream::u64s, DemultiplexStream::u64s);
        // data.back().stream_data.resize(DemultiplexStream::u64s);
        temp.resize(DemultiplexStream::u64s);
        data.emplace_back(gpu_batch_mult*batches_in_gpu_stream, gpu_readers, num_physical_streams,
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
    static const u64 batch_u64s = ceil_double_div_const(ceil_div_const(DemultiplexStream::indexes_batch_u64s,batches_in_gpu_stream*gpu_batch_mult), out_compression_ratio);
    static const u64 u64s=batches_in_out_buf_stream*batch_u64s;
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
        section_data(memory, data_offset, u64s) {}
    void emplace_back_data(i64 _1, i32 _2, i32 _3){
        OffsetVector<u8> temp(section_data, data.size()*WriteBufStream::u64s, WriteBufStream::u64s);
        // data.back().stream_data.resize(WriteBufStream::u64s*sizeof(u64));
        temp.resize(WriteBufStream::u64s*sizeof(u64));
        data.emplace_back(WriteBufStream::u64s*sizeof(u64), 1, 1, std::move(temp));
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
class FileOutStream: public CB_data<FileOutStream>{
    public:
    i64 current_file=0;
    FILE* file=nullptr;
    FileOutStream(i64 _size, i32 _max_writers, i32 _max_readers): 
        CB_data(_size, _max_writers, _max_readers) {}
};
class FileOutMS: public SharedThreadMultistream<FileOutMS,FileOutStream>{
    public:
    static constexpr bool has_write=true;
    static constexpr bool has_read=false;
    // static constexpr StreamType stream_type=SEQUENTIAL;//don't need since no read
    std::vector<StreamFilenamesContainer> files;
    i64 current_index=0;
    FileOutMS(std::vector<StreamFilenamesContainer>&& _files):
        SharedThreadMultistream(0, num_physical_streams, 0, num_physical_streams, 0, 1),
        files(std::move(_files)) {}
    void allocate_impl(i32 stream_indx, i64 id){
        //none of this should matter since !has_read
        // data[stream_indx].size=files[current_index].total_length;
        // data[stream_indx].v_w=data[stream_indx].size;
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

    ParseMS::data_offset = FileBufMS::data_offset + FileBufMS::u64s;
    MultiplexMS::data_offset = ParseMS::data_offset + ParseMS::u64s;
    DemultiplexMS::data_offset = MultiplexMS::data_offset + MultiplexMS::u64s;
    WriteBufMS::data_offset = DemultiplexMS::data_offset + DemultiplexMS::u64s;
    MemoryPositions::total = WriteBufMS::data_offset + WriteBufMS::u64s;

    FileReadMS::num_threads = num_physical_streams;
    FileBufMS::num_threads = num_physical_streams;
    ParseMS::num_threads = num_physical_streams;
    MultiplexMS::num_threads = 4;
    DemultiplexMS::num_threads = num_physical_streams;
    WriteBufMS::num_threads = num_physical_streams;
    total_threads = FileReadMS::num_threads + FileBufMS::num_threads + ParseMS::num_threads + MultiplexMS::num_threads + DemultiplexMS::num_threads + WriteBufMS::num_threads;
}
class FileReadWorker: public MS_Worker<FileReadWorker,FileReadMS,FileBufMS>{
    public:
    FileReadWorker(FileReadMS& _MS_r, FileBufMS& _MS_w):
        MS_Worker(_MS_r, _MS_w,FileBufStream::batch_u64s*sizeof(u64),FileBufStream::batch_u64s*sizeof(u64)) {}
    std::pair<i64,i64> do_work(
        i32 read_indx, i32 write_indx, i64 r_size, i64 buf_read, i64 buf_write,
        i64 read_step, i64 write_step,  bool is_first_child, bool is_last_parent){
        
    }
};

}//namespace sbwt_lcs_gpu
