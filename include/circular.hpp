#pragma once
#include "utils.h"
#include <mutex>
#include <condition_variable>
#include <queue>
namespace sbwt_lcs_gpu {
template<typename vector_pair_T>
std::string debug_print_hint(vector_pair_T& hint){
    std::stringstream ss;
    ss<<"[";
    for(i32 i=0; i<hint.size(); i++){
        ss << "(" << hint[i].first << "," << hint[i].second << ")";
        if(i<hint.size()-1) ss << ",";
    }
    ss<<"]";
    return ss.str();
}
template<typename pair_T>
std::string to_string_pair(const pair_T& pair){
    std::stringstream ss;
    ss << "(" << pair.first << "," << pair.second << ")";
    return ss.str();
}
//for non-negative and monotonic unique ids
class ConditionVariableWithCounter {
public:
    inline void wait(std::unique_lock<std::mutex>& lock) {
        waiting_count++;
        cv.wait(lock);
        waiting_count--;
    }
    inline void wait(std::unique_lock<std::mutex>& lock, std::function<bool()> pred) {
        waiting_count++;
        cv.wait(lock, pred);
        waiting_count--;
    }

    inline void notify_one() {
        cv.notify_one();
    }

    inline void notify_all() {
        cv.notify_all();
    }

    inline int get_waiting_count() const {
        return waiting_count;
    }

private:
    std::condition_variable cv;
    std::atomic<int> waiting_count = 0;
};



// enum CB_SyncType {
//     ST_SB,//single thread, single buffer
//     MT_SB,//multi thread, single buffer
//     MT_MB//multi thread, multi buffer
// };
enum CB_State {
    BEGIN,//initial
    UNBLOCKED,//normal open state
    BLOCKED,//right after initial, to avoid thread accumulation on uninitialized corresponding write buffer
    ENDED//pending deallocation
};
static_assert((CB_State::BEGIN<CB_State::UNBLOCKED) && (CB_State::UNBLOCKED<CB_State::BLOCKED) && (CB_State::BLOCKED<CB_State::ENDED), "CB_State is not ordered");
enum WaitHeapType {
    // E0,
    S,
    E
};
enum StreamType {
    PARALLEL,
    SEQUENTIAL//!must have max_readers/max_writers=1 within the stream/child stream
};

enum BufStepType{
    S0_buf,
    E0_buf
};

//Circular Buffer _data
template<typename impl>
struct CB_data{
    bool valid=false;//if false, unallocated
    // bool reserved_alloc=false;//if true, not valid yet but about to try to allocate
    bool final=false;//if true, no more writes
    i64 id=-1;
    i32 indx=-1;//indx of self
    //!must be reset to -1 after each reservation
    i32 write_indx=-1;//precomputed indx of corresponding write buffer for current step
    CB_State state=BEGIN;
    i64 S=0;//read_start_step (Start)
    i64 S0=0;//read_current_step (Current start)
    i64 E=0;//write_start_step (End)
    i64 E0=0;//write_current_step (Current end)


    //buf_S0/E0 mostly used in sequential sections, should be %size
    i64 buf_S0=0;//read_start
    i64 buf_E0=0;//write_start
    // i64 buf_S=0;//read_current //unused
    template<BufStepType T>
    void step(i64 by){
        if constexpr (T==S0_buf){
            // buf_S0=(buf_S0+by)%size;
            buf_S0=mod(buf_S0+by, size);//allows negative by
        }else{
            buf_E0=mod(buf_E0+by, size);
        }
    }

    //!v_r/v_w are not enforced in !has_write/!has_read respectively
    i64 v_r=0;//read_available
    i64 v_w;//write_available
    i64 size;//not in steps, size_step (on both read and write workers) should divide this (to avoid tearing)
    //!size must be >= size_max_write on the write buf
    //!Note: for !has_write, size has to be at least size_max_read*max_readers
    //!- and for !has_read,  size has to be at least size_max_write*max_writers
    //similarly for SEQUENTIAL for obvious reasons

    //circular buffer structure:
    //[ -> to be written ... (S) being read ... (S0) to be read ... (E) being written ... (E0) to be written ... -> ]
    //logically S<=S0<=E<=E0
    //v_r = elements (steps*step_size) between S0 and E
    //v_w = elements (steps*step_size) between E0 and S

    i32 num_writers=0;
    i32 num_readers=0;
    //!following must be 1 on the reader + writer side of a sequential section
    i32 max_writers;//expected to be positive
    i32 max_readers;//expected to be positive

    //<step, cv_index>
    std::priority_queue<std::pair<i64,i32>, std::vector<std::pair<i64,i32>>, CompareFirst> S_id_heap;//wait for (S==step)
    std::priority_queue<std::pair<i64,i32>, std::vector<std::pair<i64,i32>>, CompareFirst> E_id_heap;//wait for (E==step)
    CB_data(i64 _size, i32 _max_writers, i32 _max_readers):
        size(_size), v_w(_size), max_writers(_max_writers), max_readers(_max_readers){}
    ////impl constructor should set id, valid=true
    
    std::string dump(){
        std::stringstream ss;
        ss<<"valid: "<<valid
        <<", final: "<<final
        <<", id: "<<id
        <<", indx: "<<indx
        <<", write_indx: "<<write_indx
        <<", state: "<<state
        <<", S: "<<S
        <<", S0: "<<S0
        <<", E: "<<E
        <<", E0: "<<E0
        <<", buf_S0: "<<buf_S0
        <<", buf_E0: "<<buf_E0
        <<", v_r: "<<v_r
        <<", v_w: "<<v_w
        <<", size: "<<size
        <<", num_writers: "<<num_writers
        <<", num_readers: "<<num_readers
        <<", max_writers: "<<max_writers
        <<", max_readers: "<<max_readers
        <<", S_id_heap size: "<<S_id_heap.size()
        <<", E_id_heap size: "<<E_id_heap.size();
        ss<<"\nimpl: "<<static_cast<impl*>(this)->dump_impl();
        return ss.str();
    }
    std::string dump_impl(){//to be implemented
        return "";
    }

    //delete copy/move
    CB_data(const CB_data&)=delete;
    CB_data(CB_data&&)=delete;
    CB_data& operator=(const CB_data&)=delete;
    CB_data& operator=(CB_data&&)=delete;
};
template<typename impl,typename data_impl>
class SharedThreadMultistream {
    public:
    //sync_type_read,write
    //!must define static constexpr bool has_write and has_read
    //has_write=false means only read (generator), has_read=false means only write (consumer)
    //!must define static constexpr StreamType stream_type
    //PARALLEL means multiple readers/writers and constant size blocks,
    //SEQUENTIAL means only one reader/writer and variable size blocks

    std::mutex m;//everything below is owned by m unless otherwise specified (and except parent_m)

    //wait for ((v_r>=need) or final)[in case has_write] and (num_readers<max_readers)
    ConditionVariableWithCounter cv_S0;

    FCVector<data_impl> data;//circular buffer of streams

    bool allocate_hint=true;//hint if exists !valid in child MS

    FCVector<ConditionVariableWithCounter> cvs;//condition variable pool
    std::queue<i32> q;//queue of available condition variables



    //!these 5 pointers must be set after construction using setup_connections
    //!({parent_m, parent_cv_S0, parent_allocate_hint}=nullptr when !has_write,
    //!     and {child_hint, child_id_map}=nullptr when !has_read)
    std::mutex* parent_m=nullptr;//grab when deallocating

    //! parent_cv_S0,parent_allocate_hint owned by parent_m
    ConditionVariableWithCounter* parent_cv_S0=nullptr;//alert when deallocating
    bool* parent_allocate_hint=nullptr;//alert when deallocating

    //! child_hint,child_id_map owned by self
    //hint if (v_w>=size (ignored if !has_read), num_writers<max_writers) in child MS
    FCVector<std::pair<bool,bool>>* child_hint=nullptr;
    ContinuousVector<FCVector<std::pair<i64,i32>>>* child_id_map=nullptr;//converts id to indx

    //! self_hint,id_map owned by parent_m
    FCVector<std::pair<bool,bool>> self_hint;//grab parent_m and update
    ContinuousVector<FCVector<std::pair<i64,i32>>> id_map;//have to delete indx when deallocating

    bool exit=false;//threads will exit when true when waiting on cv_S0
    bool parent_exit=false;//threads will exit when (true and when all streams are !valid) when waiting on cv_S0

    // const i64 write_chunk_size;//useful to make a vector of chunk info structs of size (size/write_chunk_size)
    //!note that size of chunk being written always == write_chunk_size on the MS it is written to (in PARALLEL)
    //!so, if MS_A -> MS_B using MS_worker, MS_B.write_chunk_size==MS_worker.size_max_write (ensure this is the case)
    //PARALLEL writing will never offset from these cells (when using buf_E0) given that write_chunk_size divides size
    //this is still useful in SEQUENTIAL since r_size on MS_B will always be constant (unless final)


    //init data, self_hint, id_map, cvs, q, write_chunk_size
    SharedThreadMultistream(i64 stream_size, i32 num_streams, i32 max_readers, i32 max_writers,
        i32 max_readers_per_stream, i32 max_writers_per_stream): data(num_streams), self_hint(num_streams), id_map(num_streams), cvs(max_readers+max_writers){
        for(i32 i=0; i<num_streams; i++){
            // data.emplace_back(stream_size, max_writers_per_stream, max_readers_per_stream, write_chunk_size);
            static_cast<impl*>(this)->emplace_back_data(stream_size, max_writers_per_stream, max_readers_per_stream);
            self_hint.emplace_back(true, true);
        }
        for(i32 i=0; i<max_readers+max_writers; i++){
            cvs.emplace_back();
            q.push(i);
        }
    }
    void emplace_back_data(i64 stream_size, i32 max_writers_per_stream, i32 max_readers_per_stream){
        data.emplace_back(stream_size, max_writers_per_stream, max_readers_per_stream); //default implementation
    }
    //see earlier comment for setup_connections
    void setup_connections(std::mutex* _parent_m, ConditionVariableWithCounter* _parent_cv_S0, bool* _parent_allocate_hint,
        FCVector<std::pair<bool,bool>>* _child_hint, ContinuousVector<FCVector<std::pair<i64,i32>>>* _child_id_map){
        parent_m=_parent_m;
        parent_cv_S0=_parent_cv_S0;
        parent_allocate_hint=_parent_allocate_hint;
        child_hint=_child_hint;
        child_id_map=_child_id_map;
    }
    //wait on the cv pool with an id, only the least id is notified
    template<WaitHeapType heap_type>
    void add_wait(std::unique_lock<std::mutex>& locked_m, i32 stream_indx, i64 id, std::function<bool()> pred){
        if(pred()){
            return;
        }
        if(q.empty()){
            throw std::runtime_error("No available condition variables");
        }
        i32 cv_indx=q.front();
        q.pop();
        // data[stream_indx].cvs[cv_indx].wait(locked_m, pred);
        // if constexpr(heap_type==E0) data[stream_indx].E0_id_heap.push({id, cv_indx});
        if constexpr(heap_type==S) data[stream_indx].S_id_heap.push({id, cv_indx});
        else data[stream_indx].E_id_heap.push({id, cv_indx});
        cvs[cv_indx].wait(locked_m, pred);
        // if constexpr(heap_type==E0) data[stream_indx].E0_id_heap.pop();
        if constexpr(heap_type==S) data[stream_indx].S_id_heap.pop();
        else data[stream_indx].E_id_heap.pop();
        q.push(cv_indx);
    }
    //!has to be done under lock
    //notify the least id in the heap
    template<WaitHeapType heap_type>
    void notify(i32 stream_indx){
        // if constexpr(heap_type==E0){
        //     if(data[stream_indx].E0_id_heap.empty()) return;
        //     i32 cv_indx=data[stream_indx].E0_id_heap.top().second;
        //     cvs[cv_indx].notify_all();
        // }
        if constexpr(heap_type==S){
            if(data[stream_indx].S_id_heap.empty()) return;
            i32 cv_indx=data[stream_indx].S_id_heap.top().second;
            cvs[cv_indx].notify_all();
        }
        else if constexpr(heap_type==E){
            if(data[stream_indx].E_id_heap.empty()) return;
            i32 cv_indx=data[stream_indx].E_id_heap.top().second;
            cvs[cv_indx].notify_all();
        }
    }

    //!has to be called under lock
    void allocate(i32 stream_indx, i64 id){
        auto& B=data[stream_indx];
        B.id=id; //would always be -1 when !has_write
        B.indx=stream_indx;
        B.valid=true;
        B.state=BEGIN;
        B.final=false;
        B.write_indx=-1;
        B.E0=0;
        B.E=0;
        B.S0=0;
        B.S=0;
        B.buf_S0=0;
        B.buf_E0=0;
        B.num_writers=0;
        B.num_readers=0;
        //size, v_w, v_r should get automatically reset given correct implementation
        //!unless !has_write or !has_read so its better to reset them here anyway
        B.v_w=B.size;
        B.v_r=0;
        static_cast<impl*>(this)->allocate_impl(stream_indx, id);
    }
    

    //!has to be called under lock
    bool can_allocate(){
        return static_cast<impl*>(this)->can_allocate_impl();
    }
    //!has to be called under lock
    bool sequential_ended(i32 stream_indx, i64 buf_S, i64 r_size){
        return static_cast<impl*>(this)->sequential_ended_impl(stream_indx,buf_S,r_size);
    }
    //!has to be called under a lock
    bool no_write_ended(i32 stream_indx, i64 buf_S, i64 r_size){
        return static_cast<impl*>(this)->no_write_ended_impl(stream_indx,buf_S,r_size);
    }
    //!has to be called under a lock
    bool no_write_can_read(i32 stream_indx){
        return static_cast<impl*>(this)->no_write_can_read_impl(stream_indx);
    }

    //!has to be called under lock
    i64 get_write_id(i32 stream_indx){
        return static_cast<impl*>(this)->get_write_id_impl(stream_indx);
    }
    //!has to be called under lock
    bool get_is_first_parent(i32 stream_indx, bool is_begin){
        return static_cast<impl*>(this)->get_is_first_parent_impl(stream_indx, is_begin);
    }
    //!has to be called under lock
    bool get_is_last_child(i32 stream_indx, i64 buf_E, bool is_last_parent){
        return static_cast<impl*>(this)->get_is_last_child_impl(stream_indx, buf_E, is_last_parent);
    }
    //!has to be called under lock
    void deallocate(i32 stream_indx){
        data[stream_indx].valid=false;
        static_cast<impl*>(this)->deallocate_impl(stream_indx);
    }


    //allocates the immediately necessary resources, called from the first writer
    //if !has_write, called separately from do_work cycle
    //anything that takes a while should be put into implementation of do_work with a separate mutex (use is_first)
    //!NOTE: id will be -1 when !has_write, and implementation must set id itself
    //!NOTE: if this changes size, v_w has to be changed accordingly. Honestly go see what allocate() does
    void allocate_impl(i32 stream_indx, i64 id);//to be implemented

    //returns true if allocation is possible, up to implementation
    //! only relevant when !has_write
    //~does NOT automatically cache
    bool can_allocate_impl();//to be implemented
    //called at the end of each chunk, returns true if the stream has ended (last=true)
    //!Note: expected that do_work returns read size==r_size when this returns true
    //! only relevant in SEQUENTIAL && has_write class
    bool sequential_ended_impl(i32 stream_indx, i64 buf_S, i64 r_size);//to be implemented

    //called at the end of each chunk, returns true if the stream has ended (last=true)
    //!Note: no_write_can_read_impl should return false if the previous chunk would return true here when it ends
    //! only relevant in !has_write class
    bool no_write_ended_impl(i32 stream_indx, i64 buf_S, i64 r_size);//to be implemented
    //called at the beginning of each chunk, returns true if the stream can be read
    //is in place of (v_r<=size_max_read && final) check for normal MS
    //!Note: if the previous chunk would return true with no_write_ended_impl when it ends, this should return false
    //! only relevant in !has_write class
    //~does NOT automatically cache
    bool no_write_can_read_impl(i32 stream_indx);//to be implemented
    //! tip: if no_write_ended and no_write_can_read can't be implemented together with the restriction above,
    //! you should probably use SEQUENTIAL (since max_readers<1 will be enforced before no_write_can_read)

    //returns the desired id in the write buffer of the current S0 chunk in read
    ////partial cache support (may be called many times while allocating a new write buffer)
    //~does NOT automatically cache
    //! only relevant in has_read class
    i64 get_write_id_impl(i32 stream_indx);//to be implemented

    //whether current E chunk in write is the last in the stream
    //true allows future deallocation of child when all reads have been processed
    //unlike in do_work, is_last_parent is trustworthy here even if SEQUENTIAL
    //*default implementation would just be "return is_last_parent"
    //!note however that by this point the parent buffer could be deallocated
    //! only relevant when has_write
    bool get_is_last_child_impl(i32 stream_indx, i64 buf_E, bool is_last_parent);//to be implemented

    //whether current S0 chunk is read is the first in the virtual stream
    //true allows allocating the write buffer (by id) if it does not exist yet,
    //false would error if write buffer does not exist
    //*default implementation would just be "return is_begin"
    //! only relevant when has_read
    bool get_is_first_parent_impl(i32 stream_indx, bool is_begin);

    //deallocates the last parts of a resource
    //anything that takes a while should be put in implementation of do_work with a separate mutex
    //(use is_last_child or manual deduction if SEQUENTIAL)
    void deallocate_impl(i32 stream_indx);//to be implemented


    //asked at the end of each deallocation, true allows immediate exit
    //implementation expected to keep track of current state (likely with help from deallocate_impl)
    // bool can_exit_impl(i32 stream_indx);//to be implemented, called under parent m

    std::string dump(){
        std::stringstream ss;
        ss<<"cv_S0 waiting_count: "<<cv_S0.get_waiting_count()
        <<", allocate_hint: "<<allocate_hint
        <<", exit: "<<exit
        <<", parent_exit: "<<parent_exit
        // <<", write_chunk_size: "<<write_chunk_size
        <<",\n id_map: "<<debug_print_hint(id_map)
        <<",\n self_hint: "<<debug_print_hint(self_hint);

        for(i32 i=0; i<data.size(); i++){
            ss<<"\n- data["<<i<<"]:   "<<data[i].dump();
        }
        ss<<"\nimpl: "<<dump_impl();
        return ss.str();
    }
    std::string dump_impl(){//to be implemented
        return "";
    }

    //delete copy/move
    SharedThreadMultistream(const SharedThreadMultistream&)=delete;
    SharedThreadMultistream(SharedThreadMultistream&&)=delete;
    SharedThreadMultistream& operator=(const SharedThreadMultistream&)=delete;
    SharedThreadMultistream& operator=(SharedThreadMultistream&&)=delete;
};
enum S0_result{
    EXIT,

    //need index:
    BEGIN_RESERVE,//valid but not write buffer
    RESERVE,//both valid
    SELF_ALLOC,//when !has_write

    PARENT_EXIT
};
inline std::string S0_result_str(S0_result res){
    switch(res){
        case EXIT: return "EXIT";
        case BEGIN_RESERVE: return "BEGIN_RESERVE";
        case RESERVE: return "RESERVE";
        case SELF_ALLOC: return "SELF_ALLOC";
        case PARENT_EXIT: return "PARENT_EXIT";
        default: return "INVALID";
    }
}
static_assert(
    // (S0_result::FALSE<S0_result::EXIT) && 
    (S0_result::EXIT<S0_result::BEGIN_RESERVE) && 
    (S0_result::BEGIN_RESERVE<S0_result::RESERVE) && 
    (S0_result::RESERVE<S0_result::SELF_ALLOC)&&
    (S0_result::SELF_ALLOC<S0_result::PARENT_EXIT)
, "S0_result is not ordered");


//Todo (immediate): support custom predicates for waiting
//Ability to tell if a chunk was the last in a virtual stream, therefore allowing 1:n stream to virtual stream relation
//!Assume:
//- when parent_exit, all !valid will also be !reserved_alloc
//- sequential mode still requires v_r=size_max_read and v_w=_write initially but adjusts after end of batch
//- for PARALLEL write size never exceeds size_max_write no matter what read, given it is <=size_max_read
//- always allocate size_max_write
//- does not assume that r_size is not zero
//!if child stream X has virtual streams A1..An that are created by parent streams B1..Bm and is deallocated only after
//!B1..Bm are deallocated, then the B1..Bm should be guaranteed to be allocated together continuously in some order
//!otherwise, some other C1..Ck could fill up the parent stream slots before B1..Bm can be deallocated,
//!making X unable to be deallocated (child stream starvation)

//!only schedules between top and bottom MS layer, no overall virtual stream scheduling so does not prevent multi-layer deadlocks

//!the is_last_child and is_true_last is checked after do_work, so any pre-deallocation long process has to be manually
//!determined, synchronized and executed within do_work (for sequential is self explanatory, 
//!for parallel can e.g. keep track of atomic "progress" int and do final deallocate when it equals to known stream size)

//!MS_r and MS_w do not get deleted until the end of run

//===

//!Invariants
//- if a chunk A in a virtual stream is before B, A will be reserved before B and A will be deallocated before B
//  (for both write and read)
//- a thread never waits between reservation and do_work, it waits for reservation to be possible and then allocates and
//  does the work at the same time
//- a stream will be deallocated when the true_last chunk is un-reserved
//- a stream A will be allocated when any virtual parent stream requests A's id as write_id

//~fake assumptions
//- BEGIN reader does not need to respect max_readers
//- each stream corresponds 1:1 to some virtual stream, meaning if a stream ends, a virtual stream also ends and so the
//  child stream carrying the virtual stream will always be alerted when its last virtual stream ends

//Todo (potential features): give can_allocate(?) and allocate current parent(?) (maybe only for !has_read?) 
//- stream slots to avoid the child stream starvation problem
//! final on the hint doesn't seem to do anything
template<typename impl, typename read_ms_impl, typename write_ms_impl>
class MS_Worker {//MultiStream_Worker
    public:
    read_ms_impl& MS_r; //read multistream
    write_ms_impl& MS_w; //write multistream
    const i64 size_max_read;//must be positive
    const i64 size_max_write;//must be positive
    //!DEBUG
    std::string debug_str="";
    //bool matters only in single_thread mode, true means true exit, false means condition variable exit
    template<bool single_thread>
    bool _run(){
        //!debug profile//
        // std::thread::id thread_id=std::this_thread::get_id();
        // i32 thread_indx=-1;
        while(true){
            i32 indx=-1;
            i32 write_indx=-1;
            i64 id=-1;
            i64 write_id=-1;
            i64 step=-1;//current read step
            i64 write_step=-1;//write step
            i64 buf_S0=-1;//local start
            i64 buf_E0=-1;//local end
            i64 r_size=-1;//local size
            // i64 w_size=-1;//write size //not used since use std::pair do_work return
            bool last_parent=false;
            bool first_child=false;
            bool was_begin=false;
            //curr_start_(read|write) = (step*size_max_(read|write))%B_(r|w).size
            //(do not use in implementation since could have start==end):
            //curr_end_(read|write) = (curr_start_(read|write)+r_size)%B_(r|w).size
            //left inclusive, right exclusive

            //(allocate read buffer?) + grab read buffer
            // bool allocating=false;//redundant due to first_child
            S0_result res=S0_result::EXIT;//will be changed
            {
                std::unique_lock<std::mutex> lock(MS_r.m);
                //WAIT_CONDITION
                auto cond=[
                    // this,&indx,&res,&was_begin,&thread_id,&thread_indx
                    this,&indx,&res,&was_begin
                ](){//wait for exit or allocation or read availability
                    if(MS_r.exit){
                        res=EXIT;
                        return true;
                    }
                    //valid, full read or final, and
                    // - BEGIN, allocate hint 
                    // - UNBLOCKED, not max readers, (available write space) and (not max writers) hints
                    for(i32 i=0; i<MS_r.data.size(); i++){
                        auto& B_r=MS_r.data[i];
                        if(B_r.valid && B_r.state<BLOCKED && B_r.num_readers<B_r.max_readers){
                            if constexpr(MS_r.has_write){
                                if(!(B_r.v_r>=size_max_read || B_r.final)){
                                    continue;
                                }
                            }else{
                                if(!MS_r.no_write_can_read(i)){
                                    continue;
                                }
                            }
                            // do{
                            i32 _write_indx=B_r.write_indx;
                            if(_write_indx<0){//not cached
                                _write_indx=find_map(*MS_r.child_id_map, MS_r.get_write_id(i));
                                if(_write_indx<0){//invalid
                                    // if(B_r.state==BEGIN){
                                    if(MS_r.get_is_first_parent(i,B_r.state==BEGIN)){
                                        if(MS_r.allocate_hint){
                                            indx=i;
                                            res=BEGIN_RESERVE;
                                            was_begin=B_r.state==BEGIN;
                                            return true;
                                        }else continue;
                                    }
                                    std::stringstream ss;
                                    ss<<debug_str<<" ";
                                    ss << "No write buffer found for unblocked read buffer " << i 
                                    << " id " << B_r.id <<" buf_S0 "<<B_r.buf_S0<< " curr_id " << MS_r.get_write_id(i);
                                    // if constexpr (MS_r.debug_bool){//!debug
                                    //     ss<<"\n debug:\n";
                                    //     ss<<debug_print_hint(B_r.first_last_chunk) <<"\n";
                                    //     ss<<debug_print_hint(B_r.chunk_len_and_id);
                                    // }
                                    throw std::runtime_error(ss.str());
                                }
                                B_r.write_indx=_write_indx;//cache write buffer index
                            }
                            std::pair<bool,bool>& hint=(*MS_r.child_hint)[_write_indx];
                            if constexpr(!MS_w.has_read){
                                if(hint.second){
                                    indx=i;
                                    res=RESERVE;
                                    return true;                                            
                                }
                            }else {
                                if(hint.first && hint.second){
                                    indx=i;
                                    res=RESERVE;
                                    return true;
                                }
                            }
                        // }while(0);//needed to continue given BEGIN failure
                        }
                    }
                    if constexpr (!MS_r.has_write){//need to allocate ourself
                        // if(MS_r.can_allocate() && MS_r.allocate_hint){
                        if(MS_r.can_allocate()){
                            i32 i=first_of(MS_r.data, [](const auto& vec, i32 i){
                                // return !vec[i].valid && !vec[i].reserved_alloc;
                                return !vec[i].valid;//allocate immediately and then continue
                            });
                            if(i>=0){
                                indx=i;
                                res=SELF_ALLOC;
                                return true;
                            }
                        }
                    }
                    if constexpr(MS_r.has_write){
                        if(MS_r.parent_exit){//exit in case parent done and have nothing left
                            bool none_left=all_of(MS_r.data, [](const auto& vec, i32 i){
                                return !vec[i].valid;
                            });
                            if(none_left){
                                res=PARENT_EXIT;
                                return true;
                            }
                        }
                    }
                    //!debug profile//
                    // {
                    //     std::stringstream ss;
                    //     prints(ss,thread_id,debug_str,"cv_S0");
                    //     debug_object->add_string(ss.str(),thread_id,thread_indx);
                    // }
                    return false;
                };
                if constexpr(single_thread){
                    if(cond()){
                        lock.unlock();
                        return false;
                    }
                }else{
                    MS_r.cv_S0.wait(lock, cond);
                }


                if(res==EXIT){
                    MS_r.cv_S0.notify_one();
                    //!debug profile//
                    // {
                    //     std::stringstream ss;
                    //     prints(ss,thread_id,debug_str,"exit");
                    //     debug_object->add_string(ss.str(),thread_id,thread_indx);
                    // }
                    break;
                }
                if(res==PARENT_EXIT){
                    MS_r.exit=true;
                    MS_r.cv_S0.notify_one();
                    lock.unlock();
                    std::unique_lock<std::mutex> lock_w(MS_w.m);
                    MS_w.parent_exit=true;
                    MS_w.cv_S0.notify_one();
                    lock_w.unlock();
                    //!debug profile//
                    // {
                    //     std::stringstream ss;
                    //     prints(ss,thread_id,debug_str,"parent_exit");
                    //     debug_object->add_string(ss.str(),thread_id,thread_indx);
                    // }
                    break;
                }
                if(indx<0) throw std::runtime_error("No available read buffer after cv_S0");
                auto& B_r=MS_r.data[indx];
                if(res==SELF_ALLOC){
                    MS_r.allocate(indx, -1);
                    //!debug profile//
                    // {
                    //     std::stringstream ss;
                    //     prints(ss,thread_id,debug_str,"self_alloc","indx",indx,"id",B_r.id);
                    //     debug_object->add_string(ss.str(),thread_id,thread_indx);
                    // }
                    continue;
                }
                if(res==BEGIN_RESERVE || res==RESERVE){
                    if(res==BEGIN_RESERVE){
                        first_child=true;
                        MS_r.allocate_hint=false;//reset hint
                        write_id=MS_r.get_write_id(indx);
                        // if constexpr(MS_r.debug_bool){//!debug
                        //     std::stringstream ss;
                        //     ss<<debug_str<<" get_id: "<<indx<<" id,buf_S0 "<<write_id<<","<<B_r.buf_S0<<" - "<<debug_print_hint(B_r.chunk_len_and_id);
                        //     ss<<"\n first-last: "<<debug_print_hint(B_r.first_last_chunk);
                        //     syncprint(ss.str());
                        // }
                    }else{//no write_indx on first_child
                        write_indx=B_r.write_indx;//has to be cached by this point
                        if(write_indx<0) throw std::runtime_error("No write buffer at RESERVE");
                        (*MS_r.child_hint)[write_indx]={false,false};//reset hint
                    }
                    B_r.state=BLOCKED;
                    id=B_r.id;
                    step=B_r.S0;
                    buf_S0=B_r.buf_S0;

                    //!debug profile//
                    // {
                    //     std::stringstream ss;
                    //     prints(ss,thread_id,debug_str,"");
                    //     if(res==BEGIN_RESERVE){
                    //         prints(ss,"begin_reserve_r","indx",indx,"id",id,"step",step,"buf_S0",buf_S0,"write_id",write_id);
                    //     }else{
                    //         prints(ss,"reserve_r","indx",indx,"id",id,"step",step,"buf_S0",buf_S0,"write_indx",write_indx);
                    //     }
                    //     debug_object->add_string(ss.str(),thread_id,thread_indx);
                    // }
                    // can't do these yet since reservation may fail
                    // B_r.num_readers++;
                    // r_size=min(B_r.v_r, size_max_read);
                    // if(B_r.v_r==r_size && B_r.final){
                    //     last_parent=true;
                    //     B_r.state=ENDED;
                    // }
                    // step=B_r.S0;
                    // B_r.S0++;
                    // B_r.v_r-=r_size;
                // }else{//allocate, !has_write
                //     // if(!allocating) throw std::runtime_error("Not allocating when expecting");
                //     first_child=true;
                //     MS_r.data[i].reserved_alloc=true;
                //     MS_r.allocate_hint=false;//reset hint
                }
            }
            MS_r.cv_S0.notify_one();
            bool failed_reservation=true;
            bool new_alloc_hint=false;
            bool update_id_map=false;
            std::pair<bool,bool> new_hint={false,false};
            //(allocate write buffer) + grab write buffer
            {
                std::unique_lock<std::mutex> lock(MS_w.m);
                do{
                    if(first_child){
                        //avoid allocating again if id already exists
                        write_indx=first_of(MS_w.data, [write_id](const auto& vec, i32 i){
                            return vec[i].valid && vec[i].id==write_id;
                        });
                        if(write_indx<0){//allocate
                            write_indx=first_of(MS_w.data, [](const auto& vec, i32 i){
                                // return !vec[i].valid && !vec[i].reserved_alloc;
                                return !vec[i].valid;
                            });
                            if(write_indx<0){
                                // if constexpr(MS_r.debug_bool){//!debug
                                // if constexpr(1){//!debug
                                //     std::stringstream ss;
                                //     ss<<debug_str<<" fail: alloc "<<indx<<","<<write_id<<" "<<S0_result_str(res);
                                //     syncprint(ss.str());
                                // }
                                //!debug profile//
                                // {
                                //     std::stringstream ss;
                                //     prints(ss,thread_id,debug_str,"fail_alloc_w","indx",indx,"id",id,"step",step,"buf_S0",buf_S0,"write_id",write_id);
                                //     debug_object->add_string(ss.str(),thread_id,thread_indx);
                                // }
                                continue; //failed allocation
                                //failed_reservation implicitly remains true
                            }
                            // MS_w.data[write_indx].reserved_alloc=true;
                            MS_w.allocate(write_indx, write_id);
                            //now need to update id_map once grab MS_r.m
                            update_id_map=true;

                        }else{
                            //!debug profile//
                            // {
                            //     std::stringstream ss;
                            //     prints(ss,thread_id,debug_str,"alloc_exists","indx",indx,"id",id,"step",step,"buf_S0",buf_S0,"write_id",write_id,"write_indx",write_indx);
                            //     debug_object->add_string(ss.str(),thread_id,thread_indx);
                            // }
                            first_child=false;
                        }
                        new_alloc_hint=any_of(MS_w.data, [](const auto& vec, i32 i){
                            return !vec[i].valid;
                        });
                    }

                    if(write_indx<0) throw std::runtime_error("No write buffer at write reserve");
                    auto& B_w=MS_w.data[write_indx];
                    //(available write space) and (not max writers)
                    if((!MS_w.has_read || B_w.v_w>=size_max_write) && B_w.num_writers<B_w.max_writers){
                        write_step=B_w.E0;
                        buf_E0=B_w.buf_E0;
                        write_id=B_w.id;
                        
                        B_w.num_writers++;
                        B_w.E0++;
                        B_w.template step<E0_buf>(size_max_write);
                        //!assumes that buffer size is at least size_max_write*max_writers if !has_read
                        B_w.v_w-=size_max_write;

                        new_hint={B_w.v_w>=size_max_write, B_w.num_writers<B_w.max_writers};
                    }else if(first_child){
                        throw std::runtime_error("Unexpected failed to reserve allocated write buffer");
                    }else{
                        new_hint={B_w.v_w>=size_max_write, B_w.num_writers<B_w.max_writers};
                        //!debug profile//
                        // {
                        //     std::stringstream ss;
                        //     prints(ss,thread_id,debug_str,"fail_res_w","indx",indx,"id",id,"step",step,"buf_S0",buf_S0,"write_id",write_id,"write_indx",write_indx, "hint", to_string_pair(new_hint));
                        //     debug_object->add_string(ss.str(),thread_id,thread_indx);
                        // }
                        continue;//failed reservation
                        //failed_reservation implicitly remains true
                    }
                    //!debug profile//
                    // {
                    //     std::stringstream ss;
                    //     prints(ss,thread_id,debug_str,"");
                    //     if(first_child){
                    //         prints(ss,"begin_reserve_w");
                    //     }else{
                    //         prints(ss,"reserve_w");
                    //     }
                    //     prints(ss,"","indx",indx,"id",id,"step",step,"buf_S0",buf_S0,"write_id",write_id,"write_indx",write_indx,"write_step",write_step,"buf_E0",buf_E0,"hint",to_string_pair(new_hint));
                    //     debug_object->add_string(ss.str(),thread_id,thread_indx);
                    // }
                    failed_reservation=false;
                }while(0);
            }
            {
                std::unique_lock<std::mutex> lock(MS_r.m);
                auto& B_r=MS_r.data[indx];
                if(failed_reservation){
                    // if constexpr(MS_r.debug_bool){//!debug
                    //     std::stringstream ss;
                    //     ss<<debug_str<<" failed_res "<<indx<<","<<write_id<<" "<<S0_result_str(res);
                    //     syncprint(ss.str());
                    // }
                    //!debug profile//
                    // {
                    //     std::stringstream ss;
                    //     prints(ss,thread_id,debug_str,"fail_res_r","indx",indx,"id",id,"step",step,"buf_S0",buf_S0,"res",py_str(S0_result_str(res)));
                    //     if(!first_child){
                    //         prints(ss," write_indx",write_indx,"hint",to_string_pair(new_hint));
                    //     }
                    //     if(res==BEGIN_RESERVE){
                    //         prints(ss," write_id",write_id,"alloc_hint",new_alloc_hint);
                    //     }
                    //     debug_object->add_string(ss.str(),thread_id,thread_indx);
                    // }
                    if(res==BEGIN_RESERVE){//since first_child could be false but still fail from BEGIN_RESERVE
                        MS_r.allocate_hint|=new_alloc_hint;
                    }
                    if(first_child){
                        if(was_begin){
                            B_r.state=BEGIN;
                        }else{
                            B_r.state=UNBLOCKED;
                        }
                    }else{
                        B_r.state=UNBLOCKED;
                        auto& curr_pair=(*MS_r.child_hint)[write_indx];
                        curr_pair.first|=new_hint.first;
                        curr_pair.second|=new_hint.second;
                    }
                    //no need to reset write_indx because it didn't change
                    continue;
                }
                B_r.state=UNBLOCKED;

                // step=B_r.S0;
                // buf_S0=B_r.buf_S0;
                if(update_id_map){
                    // if constexpr(MS_r.debug_bool){//!debug
                    //     std::stringstream ss;
                    //     ss<<debug_str<<" update_id_map before "<<id<<","<<step;
                    //     syncprint(ss.str());
                    // }
                    // syncprint(debug_str<<" update_id_map before "<<id<<","<<step);
                    try{
                        (*MS_r.child_id_map).emplace_back(write_id, write_indx);
                        //!debug profile//
                        // {
                        //     std::stringstream ss;
                        //     prints(ss,thread_id,debug_str,"update_id_map","indx",indx,"id",id,"step",step,"buf_S0",buf_S0,"write_id",write_id,"write_indx",write_indx);
                        //     debug_object->add_string(ss.str(),thread_id,thread_indx);
                        // }
                    }catch(std::exception& e){
                        std::stringstream ss;
                        ss<<debug_str<<" ";
                        //dump all local variables
                        ss << "Failed to update id_map: " << e.what()
                        << ", current id_map: " << debug_print_hint(*MS_r.child_id_map)
                        << ",\n indx " << indx
                        << ", write_indx " << write_indx
                        << ", id " << id
                        << ", write_id " << write_id
                        << ", step " << step
                        << ", write_step " << write_step
                        << ", buf_S0 " << buf_S0
                        << ", buf_E0 " << buf_E0
                        // << ", r_size " << r_size
                        << ",\n last_parent " << last_parent
                        << ", first_child " << first_child
                        << ", was_begin " << was_begin
                        << ", res " << S0_result_str(res)
                        << ", failed_reservation " << failed_reservation
                        << ", new_alloc_hint " << new_alloc_hint
                        << ", new_hint " << to_string_pair(new_hint)<<"\n";
                        // debug_object->dump(&MS_r.m, ss.str());
                        throw e;
                    }
                    // syncprint(debug_str<<" update_id_map after "<<id<<","<<step);
                    // if constexpr(MS_r.debug_bool){//!debug
                    //     std::stringstream ss;
                    //     ss<<debug_str<<" update_id_map after "<<id<<","<<step;
                    //     syncprint(ss.str());
                    // }
                }

                B_r.S0++;
                B_r.num_readers++;
                if constexpr(MS_r.has_write){
                    if(B_r.v_r<size_max_read && !B_r.final) throw std::runtime_error("Not enough read space when expecting");
                    r_size=min(B_r.v_r, size_max_read);
                    if(B_r.v_r==r_size && B_r.final){
                        // if constexpr(MS_r.debug_bool){//!debug
                        //     std::stringstream ss;
                        //     ss<<debug_str<<" set_last_parent "<<indx<<" id "<<id<<" res,v_r,final "<<S0_result_str(res)<<","<<B_r.v_r<<","<<B_r.final;
                        //     syncprint(ss.str());
                        // }
                        // syncprint(debug_str <<" : last_parent SET, id=" << id);
                        last_parent=true;
                        B_r.state=ENDED;//need to prevent further reads
                    }
                }else{
                    r_size=size_max_read;//!assumes that buffer size is at least size_max_read*max_readers
                    //can't know about last_parent (replaced with no_write_ended() at the end)
                    //state=ENDED is replaced with no_write_can_read() at the cv_S0 wait
                }
                B_r.v_r-=r_size;
                B_r.template step<S0_buf>(r_size);
                
                MS_r.allocate_hint|=new_alloc_hint;
                std::pair<bool,bool>& hint=(*MS_r.child_hint)[write_indx];
                hint.first|=new_hint.first;//ignored if !MS_w.has_read
                hint.second|=new_hint.second;
                B_r.write_indx=-1;//must be reset
                //!debug profile//
                // {
                //     std::stringstream ss;
                //     prints(ss,thread_id,debug_str,"");
                //     prints(ss,
                //     "success_res","indx",indx,"id",id,"step",step,"buf_S0",buf_S0,
                //     "write_id",write_id,"write_indx",write_indx,"write_step",write_step,"buf_E0",buf_E0,
                //     "hint",to_string_pair(new_hint),"alloc_hint",new_alloc_hint);
                //     debug_object->add_string(ss.str(),thread_id,thread_indx);
                // }
                if(new_alloc_hint || (MS_w.has_read && new_hint.first) || new_hint.second){
                    MS_r.cv_S0.notify_one();
                }
                // if constexpr(MS_r.debug_bool){//!debug
                //     std::stringstream ss;
                //     ss<<debug_str<<" success_res "<<indx<<" "<<S0_result_str(res)<<" id,indx,buf_S0 "<<write_id<<","<<write_indx<<","<<buf_S0;
                //     syncprint(ss.str());
                // }
            }
            //do work
            //! if(r_size>0) //may lead to edge cases, add a throw into the do_work manually
            std::pair<i64,i64> consumed=static_cast<impl*>(this)->do_work(indx, write_indx, r_size, buf_S0, buf_E0,  step, write_step, first_child, last_parent);
            //!debug profile//
            // {
            //     std::stringstream ss;
            //     prints(ss,thread_id,debug_str,"do_work_finish","indx",indx,"id",id,"step",step,"buf_S0",buf_S0,"write_id",write_id,"write_indx",write_indx,"write_step",write_step,"buf_E0",buf_E0,"consumed",to_string_pair(consumed));
            //     debug_object->add_string(ss.str(),thread_id,thread_indx);
            // }
            //release write buffer
            //release read buffer
            //!read part mostly repeated below for write
            bool about_to_exit=false;
            bool parent_exit=false;
            update_id_map=false;
            new_alloc_hint=false;
            new_hint={false,false};
            {
                std::unique_lock<std::mutex> lock(MS_r.m);
                //WAIT_CONDITION
                MS_r.template add_wait<S>(lock, indx, step, [this,indx,step](){
                    return MS_r.data[indx].S==step;
                });
                auto& B_r=MS_r.data[indx];
                B_r.S++;
                B_r.num_readers--;
                if constexpr(MS_r.stream_type==SEQUENTIAL){
                    // if constexpr(MS_r.debug_bool){//!debug
                    //     std::stringstream ss;
                    //     ss<<debug_str<<" free MS_r BEFORE "<<indx<<" id "<<id<<" step "<<step<<" last_parent "<<last_parent<<" v_w,v_r,S0_buf "<<B_r.v_w<<","<<B_r.v_r<<","<<B_r.buf_S0;
                    //     syncprint(ss.str());
                    // }
                    B_r.v_w+=consumed.first;//actual amount read
                    B_r.template step<S0_buf>(consumed.first-r_size);//if consumed less than r_size, step back
                    B_r.v_r+=r_size-consumed.first;//undo too much read

                    r_size=consumed.first;//give correct amount to sequential_ended and no_write_ended

                    if(consumed.first<r_size){
                        B_r.state=UNBLOCKED;
                    }
                    if constexpr(MS_r.has_write){
                        last_parent=MS_r.sequential_ended(indx, buf_S0, r_size);//use no_write_ended otherwise
                    }
                    // if constexpr(MS_r.debug_bool){//!debug
                    //     std::stringstream ss;
                    //     ss<<debug_str<<" free MS_r AFTER "<<indx<<" id "<<id<<" step "<<step<<" last_parent "<<last_parent<<" v_w,v_r,S0_buf "<<B_r.v_w<<","<<B_r.v_r<<","<<B_r.buf_S0;
                    //     syncprint(ss.str());
                    // }
                }else{
                    B_r.v_w+=r_size;
                }
                if constexpr(!MS_r.has_write){
                    last_parent=MS_r.no_write_ended(indx, buf_S0, r_size);
                }
                //!debug profile//
                // {
                //     std::stringstream ss;
                //     prints(ss,thread_id,debug_str,"");
                //     prints(ss,"release_r","indx",indx,"id",id,"new_v_w",B_r.v_w,"new_v_r",B_r.v_r,"new_S",B_r.S,"new_readers",B_r.num_readers,"last_parent",last_parent);
                //     debug_object->add_string(ss.str(),thread_id,thread_indx);
                // }


                if(last_parent){
                    // if constexpr(MS_r.debug_bool){//!debug
                    //     std::stringstream ss;
                    //     ss<<debug_str<<" dealloc_read "<<indx<<" id "<<id;
                    //     syncprint(ss.str());
                    // }
                    // syncprint(debug_str <<" : last_parent, id=" << id);
                    //////potentially sets exit if final_dealloc is called
                    // syncprint(debug_str<<" deallocating indx,id "<<indx<<","<<id);//!debug
                    if constexpr(!MS_r.has_write){
                        MS_r.deallocate(indx);
                        // if(((!MS_r.has_write)&&(!MS_r.can_allocate()))||(MS_r.has_write&&MS_r.parent_exit)){
                        if(!MS_r.can_allocate()){
                            MS_r.exit=all_of(MS_r.data, [](const auto& vec, i32 i){
                                return !vec[i].valid;
                            });
                            if(MS_r.exit) about_to_exit=true;
                        }
                        //!debug profile//
                        // {
                        //     std::stringstream ss;
                        //     prints(ss,thread_id,debug_str,"dealloc_read","indx",indx,"id",id,"about_to_exit",about_to_exit);
                        //     debug_object->add_string(ss.str(),thread_id,thread_indx);
                        // }
                    }else{
                        update_id_map=true;
                        //!can't do it here because would be deallocating before updating id_map, and false positives are not allowed
                    }
                    // MS_r.deallocate(indx);//!can't do it before updating id_map since false positives are not allowed
                    // if(((!MS_r.has_write)&&(!MS_r.can_allocate()))||(MS_r.has_write&&MS_r.parent_exit)){
                    //     MS_r.exit=all_of(MS_r.data, [](const auto& vec, i32 i){
                    //         return !vec[i].valid;
                    //     });
                    // }
                    B_r.state=ENDED;
                    // new_alloc_hint=any_of(MS_r.data, [](const auto& vec, i32 i){
                    //     return !vec[i].valid;
                    // });
                    // new_alloc_hint=true;//since deallocated
                }else{
                    new_hint={B_r.v_w>=size_max_write, B_r.num_readers<B_r.max_readers};
                }
                MS_r.cv_S0.notify_one();//since num_readers changed
                MS_r.template notify<S>(indx);//since S changed
                // MS_r.template notify<E0>(indx);//since v_w changed
                parent_exit=MS_r.parent_exit;
                // if(MS_r.exit) about_to_exit=true;
            }
            if constexpr(MS_r.has_write){
                if(update_id_map){//dealloc
                    if(!parent_exit){
                        if(MS_r.parent_m==nullptr) throw std::runtime_error("No parent_m");
                        std::unique_lock<std::mutex> lock(*MS_r.parent_m);
                        i32 i=0;
                        i32 num_removed=0;
                        while(i<MS_r.id_map.size()){
                            auto& pair=MS_r.id_map[i];
                            if(pair.first==id){
                                // syncprint(debug_str<<" removing id_map "<<pair.first<<","<<pair.second);
                                MS_r.id_map.remove(i);//replaces with last
                                num_removed++;
                                //!debug profile//
                                // {
                                //     std::stringstream ss;
                                //     prints(ss,thread_id,debug_str,"");
                                //     prints(ss,"del_parent_id_map","indx",indx,"id",id,"pos",i);
                                //     debug_object->add_string(ss.str(),thread_id,thread_indx);
                                // }
                            }else{
                                i++;
                            }
                        }
                        if(num_removed!=1){
                            std::stringstream ss;
                            ss<<debug_str<<" removed!=1: "<<num_removed<<" id "<<id;
                            throw std::runtime_error(ss.str());
                        }
                        // MS_r.parent_cv_S0->notify_one();//since v_w changed
                    }
                    {
                        std::unique_lock<std::mutex> lock(MS_r.m);
                        MS_r.deallocate(indx);//!has to be done after updating the id_map
                        if(MS_r.parent_exit){
                            MS_r.exit=all_of(MS_r.data, [](const auto& vec, i32 i){
                                return !vec[i].valid;
                            });
                            if(MS_r.exit) about_to_exit=true;
                        }
                        //!debug profile//
                        // {
                        //     std::stringstream ss;
                        //     prints(ss,thread_id,debug_str,"dealloc_read","indx",indx,"id",id,"about_to_exit",about_to_exit);
                        //     debug_object->add_string(ss.str(),thread_id,thread_indx);
                        // }
                        new_alloc_hint=true;//since deallocated
                        parent_exit=MS_r.parent_exit;
                        MS_r.cv_S0.notify_one();
                    }
                }

                if(!parent_exit){
                    if(MS_r.parent_m==nullptr) throw std::runtime_error("No parent_m");
                    if(new_alloc_hint || (new_hint.first && new_hint.second)){
                        //will always be reached if id_map is updated
                        std::unique_lock<std::mutex> lock(*MS_r.parent_m);
                        auto& hint=MS_r.self_hint[indx];
                        //!debug profile//
                        // {
                        //     std::stringstream ss;
                        //     prints(ss,thread_id,debug_str,"");
                        //     prints(ss,"update_parent_hints_r","indx",indx,"alloc_hint",(*MS_r.parent_allocate_hint),"hint",to_string_pair(hint), "new_alloc_hint",new_alloc_hint, "new_hint",to_string_pair(new_hint));
                        //     debug_object->add_string(ss.str(),thread_id,thread_indx);
                        // }
                        hint.first|=new_hint.first;
                        hint.second|=new_hint.second;
                        (*MS_r.parent_allocate_hint)|=new_alloc_hint;
                        MS_r.parent_cv_S0->notify_one();//since v_w changed
                    }
                }
            }
            //!write mostly a copy of read
            update_id_map=false;
            new_alloc_hint=false;
            new_hint={false,false};
            {
                std::unique_lock<std::mutex> lock(MS_w.m);
                //WAIT_CONDITION
                MS_w.template add_wait<E>(lock, write_indx, write_step, [this,write_indx,write_step](){
                    return MS_w.data[write_indx].E==write_step;
                });
                auto& B_w=MS_w.data[write_indx];
                B_w.E++;
                B_w.num_writers--;
                if constexpr(MS_r.stream_type==SEQUENTIAL){
                    //only relevant in sequential mode
                    B_w.v_r+=consumed.second;//actual amount written
                    B_w.template step<E0_buf>(consumed.second-size_max_write);//if consumed less than size_max_write, step back
                    B_w.v_w+=size_max_write-consumed.second;//undo too much write
                    // if constexpr(MS_r.debug_bool){//!debug
                    //     std::stringstream ss;
                    //     ss<<debug_str<<" free MS_w "<<write_indx<<" id "<<write_id<<" step "<<write_step<<" last_parent "<<last_parent<<" v_r,v_w,E0_buf "<<B_w.v_r<<","<<B_w.v_w<<","<<B_w.buf_E0;
                    //     syncprint(ss.str());
                    // }
                }else{
                    B_w.v_r+=size_max_write;
                }
                // if(last_parent){
                if(MS_w.get_is_last_child(write_indx,buf_E0,last_parent)){
                    // B_w.final=MS_w.is_true_last(write_indx);
                    B_w.final=true;
                    //!can't do it here because would be deallocating before updating id_map, and false positives are not allowed
                    if constexpr(!MS_w.has_read){
                        // if(B_w.final){
                            // MS_w.deallocate(write_indx)
                            // new_alloc_hint=true;//since deallocated
                        // }
                        if(B_w.final){
                            update_id_map=true;
                            B_w.state=ENDED;//probably doesn't actually do anything though
                        }
                    }
                }else{
                    new_hint={B_w.v_w>=size_max_write, B_w.num_writers<B_w.max_writers};
                }
                // if constexpr(!MS_w.has_read){
                //     PRINT("end_MS end: id"<<id<<", step"<<write_step<<", hint "<<new_hint.first<<" "<<new_hint.second);
                // }

                //!debug profile//
                // {
                //     std::stringstream ss;
                //     prints(ss,thread_id,debug_str,"");
                //     prints(ss,"release_w","write_indx",write_indx,"write_id",write_id,"new_v_r",B_w.v_r,"new_v_w",B_w.v_w,"new_E",B_w.E,"new_writers",B_w.num_writers,"final",B_w.final);
                //     debug_object->add_string(ss.str(),thread_id,thread_indx);
                // }

                MS_w.parent_exit=about_to_exit;//useless if !MS_w.has_read
                // MS_w.template notify<E0>(write_indx);//since num_writers changed
                MS_w.template notify<E>(write_indx);//since E changed
                MS_w.cv_S0.notify_one();//since v_r changed
            }
            if constexpr(!MS_w.has_read){
                if(update_id_map){
                    if(!about_to_exit){
                        std::unique_lock<std::mutex> lock(MS_r.m);
                        i32 i=0;
                        i32 num_removed=0;
                        while(i<MS_w.id_map.size()){
                            auto& pair=MS_w.id_map[i];
                            if(pair.first==write_id){
                                MS_w.id_map.remove(i);//replaces with last
                                num_removed++;
                                //!debug profile//
                                // {
                                //     std::stringstream ss;
                                //     prints(ss,thread_id,debug_str,"");
                                //     prints(ss,"del_MS_r_id_map","write_indx",write_indx,"write_id",write_id,"pos",i);
                                //     debug_object->add_string(ss.str(),thread_id,thread_indx);
                                // }
                            }else{
                                i++;
                            }
                        }
                        if(num_removed!=1){//!debug
                            std::stringstream ss;
                            ss<<debug_str<<" removed MS_w!=1: "<<num_removed<<" w_id,w_indx "<<write_id<<","<<write_indx;
                            throw std::runtime_error(ss.str());
                        }
                    }
                    {
                        std::unique_lock<std::mutex> lock(MS_w.m);
                        MS_w.deallocate(write_indx);//!has to be done after updating the id_map
                        new_alloc_hint=true;//since deallocated
                        //!debug profile//
                        // {
                        //     std::stringstream ss;
                        //     prints(ss,thread_id,debug_str,"dealloc_write","write_indx",write_indx,"write_id",write_id);
                        //     debug_object->add_string(ss.str(),thread_id,thread_indx);
                        // }
                    }
                }
            }
            if(!about_to_exit){
                if(new_alloc_hint || ((!MS_w.has_read || new_hint.first) && new_hint.second)){
                    std::unique_lock<std::mutex> lock(MS_r.m);
                    auto& hint=MS_w.self_hint[write_indx];
                    //!debug profile//
                    // {
                    //     std::stringstream ss;
                    //     prints(ss,thread_id,debug_str,"");
                    //     prints(ss,"update_parent_hints_w","write_indx",write_indx,"alloc_hint",MS_r.allocate_hint,"hint",to_string_pair(hint),"new_alloc_hint",new_alloc_hint,"new_hint",to_string_pair(new_hint));
                    //     debug_object->add_string(ss.str(),thread_id,thread_indx);
                    // }
                    MS_r.allocate_hint|=new_alloc_hint;
                    hint.first|=new_hint.first;
                    hint.second|=new_hint.second;
                    //redundant since will go to cv_S0 after this anyway
                    // MS_r.cv_S0.notify_one();//since v_w changed
                }
            }
        }
        return true;
        // syncprint(debug_str<< "::run end");
    }
    void run(){
        _run<false>();//multithreaded
    }
    bool run_single(){
        return _run<true>();//single threaded
    }
    MS_Worker(read_ms_impl& _MS_r, write_ms_impl& _MS_w, i64 _size_max_read, i64 _size_max_write):
        MS_r(_MS_r), MS_w(_MS_w), size_max_read(_size_max_read), size_max_write(_size_max_write){}

    //!implementation should be able to deal with (buf_read+r_size<buf_read)%size 
    // or (buf_write+size_max_write<buf_write)%size
    //!Note: is_last_parent is meaningless in (SEQUENTIAL MS_r or !has_read MS_w) and is up to implementation
    //!r_size==size_max_write for SEQUENTIAL MS_r or !has_write MS_r
    //!return (r_size,w_size) pair is only used in SEQUENTIAL MS_r
    std::pair<i64,i64> do_work(i32 read_indx, i32 write_indx, i64 r_size, i64 buf_read, i64 buf_write, i64 read_step, i64 write_step,  bool is_first_child, bool is_last_parent);//to be implemented
};


}//namespace sbwt_lcs_gpu