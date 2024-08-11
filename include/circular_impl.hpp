#pragma once
#include "utils.h"
#include "structs.hpp"
#include "circular.hpp"

#include <mutex>
#include <chrono>

namespace sbwt_lcs_gpu {

static std::mutex io_mutex;
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
class FileReadData: public CB_data<FileReadData>{
    public:
    i64 current_file=0;
    FileReadData(i64 _size, i32 _max_writers, i32 _max_readers, i64 write_chunk_size): 
        CB_data(_size, _max_writers, _max_readers, write_chunk_size) {}
    // std::string dump_impl(){
    //     std::stringstream ss;
    //     ss <<" gen_num "<<gen_num;
    //     return ss.str();
    // }
};
class FileReadMS: public SharedThreadMultistream<FileReadMS,FileReadData>{
    public:
    static constexpr bool has_write=false;
    static constexpr bool has_read=true;
    static constexpr StreamType stream_type=SEQUENTIAL;
    static constexpr bool debug_bool=0;
    // FileNameVec files;
    std::vector<StreamFilenamesContainer> files;
    i64 current_index=0;
    // std::string dump_impl(){
    //     std::stringstream ss;
    //     ss <<" gen_num "<<gen_num
    //     << ", gen_max "<<gen_max;
    //     return ss.str();
    // }
    FileReadMS(i64 stream_size, i32 num_streams, i32 max_readers, i32 max_writers,
        i32 max_readers_per_stream, i32 max_writers_per_stream, i64 write_chunk_size, i64 _gen_max, std::vector<StreamFilenamesContainer>&& _files):
        SharedThreadMultistream(stream_size, num_streams, max_readers, max_writers, max_readers_per_stream, max_writers_per_stream, write_chunk_size),
        files(std::move(_files)) {}
    void allocate_impl(i32 stream_indx, i64 id){
        // data[stream_indx].gen_num=gen_num;
        data[stream_indx].id=current_index;
        data[stream_indx].v_r=files[current_index].total_length;
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
        return data[stream_indx].id+10;
    }
    bool get_is_last_child_impl(i32 stream_indx, i64 buf_E, bool is_last_parent){return 1;}//not relevant
    bool get_is_first_parent_impl(i32 stream_indx, bool is_begin){return is_begin;}
};
class middle_data_impl: public CB_data<middle_data_impl>{//just for
    public:
    std::vector<i64> data;
    std::vector<std::pair<i64,i64>> chunk_len_and_id;
    std::vector<std::pair<bool,bool>> first_last_chunk;
    std::string dump_impl(){
        std::stringstream ss;
        ss <<" data: "<<printout_vec(data)
        <<",\n chunk_len_and_id: "<<debug_print_hint(chunk_len_and_id)
        <<",\n first_last_chunk: "<<debug_print_hint(first_last_chunk);
        return ss.str();
    }
    middle_data_impl(i64 _size, i32 _max_writers, i32 _max_readers, i64 write_chunk_size):
        CB_data(_size, _max_writers, _max_readers, write_chunk_size),
        data(_size), chunk_len_and_id(ceil_div(_size,write_chunk_size)),first_last_chunk(ceil_div(_size,write_chunk_size)){

        // PRINT("middle_data_impl::middle_data_impl " << std::this_thread::get_id() << " size " << _size);
    }
};
class middle_MS_impl: public SharedThreadMultistream<middle_MS_impl,middle_data_impl>{
    public:
    using SharedThreadMultistream::SharedThreadMultistream;
    static constexpr bool has_write=true;
    static constexpr bool has_read=true;
    static constexpr StreamType stream_type=PARALLEL;
    static constexpr bool debug_bool=0;

    // i64 global_id=0;
    void allocate_impl(i32 stream_indx, i64 id){
        // PRINT("middle_MS_impl"<<global_id<<"::allocate_impl " << std::this_thread::get_id() << " sid " << stream_indx << " id " << id);
    }
    void deallocate_impl(i32 stream_indx){
        //nothing to do
        // PRINT("middle_MS_impl"<<global_id<<"::deallocate_impl " << std::this_thread::get_id() << " sid " << stream_indx);
    }
    i64 get_write_id_impl(i32 stream_indx){
        return 0;
    }
    bool get_is_last_child_impl(i32 stream_indx, i64 buf_E, bool is_last_parent){
        return is_last_parent;
    }
    bool get_is_first_parent_impl(i32 stream_indx, bool is_begin){return is_begin;}
};
class group_data_impl: public CB_data<group_data_impl>{//just for
    public:
    std::vector<i64> data;
    std::vector<std::pair<i64,i64>> chunk_len_and_id;
    std::vector<std::pair<bool,bool>> first_last_chunk;
    std::string dump_impl(){
        std::stringstream ss;
        ss <<" data: "<<printout_vec(data)
        <<",\n chunk_len_and_id: "<<debug_print_hint(chunk_len_and_id)
        <<",\n first_last_chunk: "<<debug_print_hint(first_last_chunk);
        return ss.str();
    }
    group_data_impl(i64 _size, i32 _max_writers, i32 _max_readers, i64 write_chunk_size):
        CB_data(_size, _max_writers, _max_readers, write_chunk_size),
        data(_size), chunk_len_and_id(ceil_div(_size,write_chunk_size)), first_last_chunk(ceil_div(_size,write_chunk_size)){

        // PRINT("group_data_impl::group_data_impl " << std::this_thread::get_id() << " size " << _size);
    }
};
class group_MS_impl: public SharedThreadMultistream<group_MS_impl,group_data_impl>{
    public:
    using SharedThreadMultistream::SharedThreadMultistream;
    static constexpr bool has_write=true;
    static constexpr bool has_read=true;
    static constexpr StreamType stream_type=PARALLEL;
    static constexpr bool debug_bool=0;//!debug_bool

    i64 num_streams=0;
    std::string dump_impl(){
        std::stringstream ss;
        ss <<" num_streams "<<num_streams;
        return ss.str();
    }
    void allocate_impl(i32 stream_indx, i64 id){
        // PRINT("group_MS_impl"<<global_id<<"::allocate_impl " << std::this_thread::get_id() << " sid " << stream_indx << " id " << id);
    }
    void deallocate_impl(i32 stream_indx){
        //nothing to do
        // PRINT("group_MS_impl"<<global_id<<"::deallocate_impl " << std::this_thread::get_id() << " sid " << stream_indx);
    }
    i64 get_write_id_impl(i32 stream_indx){
        i64 indx=data[stream_indx].buf_S0/write_chunk_size;
        return data[stream_indx].chunk_len_and_id[indx].second+10;
    }
    bool get_is_last_child_impl(i32 stream_indx, i64 buf_E, bool is_last_parent){
        num_streams-=is_last_parent;
        return num_streams==0;
    }
    bool get_is_first_parent_impl(i32 stream_indx, bool is_begin){
        // return is_first_parent;
        i64 indx=data[stream_indx].buf_S0/write_chunk_size;
        return data[stream_indx].first_last_chunk[indx].first;
    }

};
class end_buf_data_impl: public CB_data<end_buf_data_impl>{
    public:
    std::vector<i64> data;
    std::vector<std::pair<i64,i64>> chunk_len_and_id;
    std::vector<std::pair<bool,bool>> first_last_chunk;
    std::string dump_impl(){
        std::stringstream ss;
        ss <<" data: "<<printout_vec(data)
        <<",\n chunk_len_and_id: "<<debug_print_hint(chunk_len_and_id)
        <<",\n first_last_chunk: "<<debug_print_hint(first_last_chunk);
        return ss.str();
    }
    end_buf_data_impl(i64 _size, i32 _max_writers, i32 _max_readers, i64 write_chunk_size):
        CB_data(_size, _max_writers, _max_readers, write_chunk_size),
        data(_size), chunk_len_and_id(ceil_div(_size,write_chunk_size)), first_last_chunk(ceil_div(_size,write_chunk_size)){
        // PRINT("end_buf_data_impl::end_buf_data_impl " << std::this_thread::get_id());
    }
};
class end_buf_MS_impl: public SharedThreadMultistream<end_buf_MS_impl,end_buf_data_impl>{
    public:
    using SharedThreadMultistream::SharedThreadMultistream;
    static constexpr bool has_write=true;
    static constexpr bool has_read=true;
    static constexpr StreamType stream_type=SEQUENTIAL;
    static constexpr bool debug_bool=0;//!debug_bool

    void allocate_impl(i32 stream_indx, i64 id){
        data[stream_indx].v_w=data[stream_indx].size;
        // data[stream_indx].v_r=0;
        
        // PRINT("end_buf_MS_impl::allocate_impl " << std::this_thread::get_id() << " sid " << stream_indx  << " id " << id);
    }
    void deallocate_impl(i32 stream_indx){
        //nothing to do
        // PRINT("end_buf_MS_impl::deallocate_impl " << std::this_thread::get_id() << " sid " << stream_indx);
    }
    i64 get_write_id_impl(i32 stream_indx){
        return data[stream_indx].id+10;
    }
    bool sequential_ended_impl(i32 stream_indx,i64 buf_S, i64 r_size){//!necessary
        i64 indx=buf_S/write_chunk_size;
        return data[stream_indx].first_last_chunk[indx].second;
    }
    bool get_is_last_child_impl(i32 stream_indx, i64 buf_E, bool is_last_parent){
        i64 indx=buf_E/write_chunk_size;
        return data[stream_indx].first_last_chunk[indx].second;
    }
    bool get_is_first_parent_impl(i32 stream_indx, bool is_begin){return is_begin;}
};
class end_data_impl: public CB_data<end_data_impl>{
    public:
    end_data_impl(i64 _size, i32 _max_writers, i32 _max_readers, i64 write_chunk_size):
        CB_data(_size, _max_writers, _max_readers, write_chunk_size){
        // PRINT("end_data_impl::end_data_impl " << std::this_thread::get_id());
    }
};
class end_MS_impl: public SharedThreadMultistream<end_MS_impl,end_data_impl>{
    public:
    using SharedThreadMultistream::SharedThreadMultistream;
    static constexpr bool has_write=true;
    static constexpr bool has_read=false;
    static constexpr bool debug_bool=0;

    // static constexpr StreamType stream_type=PARALLEL; //not needed because last?
    void allocate_impl(i32 stream_indx, i64 id){
        data[stream_indx].v_w=data[stream_indx].size;
        // data[stream_indx].v_r=0;
        
        // PRINT("end_MS_impl::allocate_impl " << std::this_thread::get_id() << " sid " << stream_indx  << " id " << id);
    }
    void deallocate_impl(i32 stream_indx){
        //nothing to do
        // PRINT("end_MS_impl::deallocate_impl " << std::this_thread::get_id() << " sid " << stream_indx);
    }
    i64 get_write_id_impl(i32 stream_indx){//should never be called
        return data[stream_indx].id;
    }
    bool get_is_last_child_impl(i32 stream_indx, i64 buf_E, bool is_last_parent){
        return is_last_parent;
    }
    bool get_is_first_parent_impl(i32 stream_indx, bool is_begin){return is_begin;}//should never be called
};


}//namespace sbwt_lcs_gpu