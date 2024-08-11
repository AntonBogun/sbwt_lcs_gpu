#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <cstdlib>
#include "SBWT.hpp"
#include "gpu/kernels.h"
#include "utils.h"
#include <omp.h>
#include "balance_files.hpp"
#include "sanity_test.h"

constexpr int error_exit_code = -1;

using namespace sbwt_lcs_gpu;
int cerr_and_return(const std::string msg, int exit_code) {
    std::cerr << msg << std::endl;
    return exit_code;
}
enum class DeviceType {
    CPU,
    GPU
};
enum class ParseToggle {
    None,
    In,
    Out
};
int main(int argc, char *argv[]) {
    int i = 1;
    std::string sbwt_file;
    std::vector<std::string> in_files;
    // bool in_file_toggle = false;
    std::vector<std::string> out_files;
    // bool out_file_toggle = false;
    std::string in_list_file;
    std::string out_list_file;
    ParseToggle file_toggle = ParseToggle::None;
    // std::cout<<"parsing"<<std::endl;//!debug
    while (i < argc) {
        std::string s(argv[i++]);
        // std::cout<<"i: "<<i<<" s: "<<s<<std::endl;//!debug
        if (s.compare("-sbwt") == 0) {
            if (i>=argc) return cerr_and_return("Missing argument for -sbwt", error_exit_code);
            sbwt_file = std::string(argv[i++]);

        }else if (s.compare("-if") == 0) {
            if (i>=argc) return cerr_and_return("Missing argument for -if", error_exit_code);
            if (in_files.size() > 0) return cerr_and_return("Cannot mix -if and -i", error_exit_code);
            in_list_file = std::string(argv[i++]);
            // std::cout<<in_list_file<<std::endl;//!debug

        } else if (s.compare("-of") == 0) {
            if (i>=argc) return cerr_and_return("Missing argument for -of", error_exit_code);
            if (out_files.size() > 0) return cerr_and_return("Cannot mix -of and -o", error_exit_code);
            out_list_file = std::string(argv[i++]);

        } else if (s.compare("-i") == 0) {
            if (i>=argc) return cerr_and_return("Missing argument for -i", error_exit_code);
            if (in_list_file.size() > 0) return cerr_and_return("Cannot mix -if and -i", error_exit_code);
            in_files.push_back(std::string(argv[i++]));
            // in_file_toggle = true;
            file_toggle = ParseToggle::In;
            continue;
        } else if (s.compare("-o") == 0) {
            if (i>=argc) return cerr_and_return("Missing argument for -o", error_exit_code);
            if (out_list_file.size() > 0) return cerr_and_return("Cannot mix -of and -o", error_exit_code);
            out_files.push_back(std::string(argv[i++]));
            // out_file_toggle = true;
            file_toggle = ParseToggle::Out;
            continue;
        } else if (file_toggle==ParseToggle::In) {
            in_files.push_back(s);
            continue;

        } else if (file_toggle==ParseToggle::Out) {
            out_files.push_back(s);
            continue;

        } else {
            return cerr_and_return("Unknown argument: " + s, error_exit_code);
        }
        // in_file_toggle = false;
        // out_file_toggle = false;
        file_toggle = ParseToggle::None;
    }
    //=end of parsing
    {
        sanity_test_cpu();
        sanity_test_gpu();
        std::cout << "sanity tests passed" << std::endl;   
    }

    //=sbwt file validation
    if(sbwt_file.empty()) return cerr_and_return("No SBWT file given", error_exit_code);
    if(!ThrowingIfstream::check_file_exists(sbwt_file)){
        return cerr_and_return("SBWT file does not exist: " + sbwt_file, error_exit_code);
    } 

    //=in/out list files
    if(in_list_file.size() > 0){
        if(!ThrowingIfstream::check_file_exists(in_list_file)){
            return cerr_and_return("Input list file does not exist: " + in_list_file, error_exit_code);
        }
        ThrowingIfstream in_list_stream(in_list_file, std::ios::in);
        std::string in_file;
        bool not_eof;
        do{
            not_eof=in_list_stream.read_line(in_file);
            if(in_file.size() > 0){
                in_files.push_back(in_file);
            }
        }while(not_eof);
    }
    if(out_list_file.size() > 0){
        if(!ThrowingIfstream::check_file_exists(out_list_file)){
            return cerr_and_return("Output list file does not exist: " + out_list_file, error_exit_code);
        }
        ThrowingIfstream out_list_stream(out_list_file, std::ios::in);
        std::string out_file;
        bool not_eof;
        do{
            not_eof=out_list_stream.read_line(out_file);
            if(out_file.size() > 0){
                out_files.push_back(out_file);
            }
        }while(not_eof);
    }
    //=validate size
    if(in_files.empty()){
        std::cerr << "No input files given" << std::endl
                    << "Use -i or -if to specify input files" << std::endl;
        return error_exit_code;
    }
    if(out_files.empty()){
        std::cerr << "No output files given" << std::endl
                    << "Use -o or -of to specify output files" << std::endl;
        return error_exit_code;
    }
    if(in_files.size() != out_files.size()){
        std::cerr << "Number of input files does not match number of output files" << std::endl
                    << "Number of input files: " << in_files.size() << std::endl
                    << "Number of output files: " << out_files.size() << std::endl;
        return error_exit_code;
    }

    //=check all unique
    std::vector<std::string> all_files;
    all_files.reserve(in_files.size() + out_files.size());
    all_files.insert(all_files.end(), in_files.begin(), in_files.end());
    all_files.insert(all_files.end(), out_files.begin(), out_files.end());
    std::sort(all_files.begin(), all_files.end());
    for(size_t i = 1; i < all_files.size(); ++i){
        if(all_files[i] == all_files[i-1]){
            return cerr_and_return("Duplicate file: " + all_files[i], error_exit_code);
        }
    }
    //=check exist in and valid out paths
    std::string missing_file;
    if(any_of(in_files, [&missing_file](auto& vec, i32 i){
        if(!ThrowingIfstream::check_file_exists(vec[i])){
            missing_file = vec[i];
            return true;
        }
        return false;
    })){
        return cerr_and_return("Input file does not exist: " + missing_file, error_exit_code);
    }
    std::string invalid_file;
    if(any_of(out_files, [&invalid_file](auto& vec, i32 i){
        if(!ThrowingOfstream::check_path_valid(vec[i])){
            invalid_file = vec[i];
            return true;
        }
        return false;
    })){
        return cerr_and_return("Invalid output file: " + invalid_file, error_exit_code);
    }


    auto streams=balance_files(in_files, out_files);



    //print out the files
    std::cout << "sbwt_file: " << sbwt_file << std::endl;
    // for(size_t i = 0; i < in_files.size(); ++i){
    //     std::cout << prints_new("in_file:", in_files[i], " out_file:", out_files[i]) << std::endl;
    // }
    for(i64 i = 0; i < streams.size(); ++i){
        std::cout << "stream " << i << std::endl;
        // i64 total_size = 0;
        for(i64 j = 0; j < streams[i].filenames.size(); ++j){
            // std::cout << prints_new("in_file:", std::get<0>(streams[i][j]), " out_file:", std::get<1>(streams[i][j]), " size:", std::get<2>(streams[i][j])) << std::endl;
            // total_size += std::get<2>(streams[i][j]);
            std::cout << prints_new("in_file:", streams[i].filenames[j], " out_file:", streams[i].output_filenames[j], " size:", streams[i].lengths[j]) << std::endl;
        }
        // std::cout << "total_size: " << total_size << std::endl;
        std::cout << "total_size: " << streams[i].total_length << std::endl;
        std::cout << std::endl;
    }

    //print free cpu and gpu memory
    std::cout << "free_cpu_memory: " << get_free_cpu_memory() << std::endl;
    std::cout << "free_gpu_memory: " << get_free_gpu_memory() << std::endl;

    // print the supported cuda devices
    // print_supported_devices();

    // load sbwt
    SBWTContainerCPU sbwt_cpu;
    sbwt_cpu.load_from_file(sbwt_file);
    sbwt_cpu.print_info();
    u64 kmer_size = sbwt_cpu.get_kmer_size();
    u64 num_bits = sbwt_cpu.get_num_bits();

    // i64 num_cpu_threads = std::string(std::getenv("NUM_CPU_THREADS")).empty() ? std::thread::hardware_concurrency() : std::stoi(std::getenv("NUM_CPU_THREADS"));
    // i64 thread_concurrency=std::thread::hardware_concurrency();
    i64 num_cpu_threads=1;
    #pragma omp parallel
    {
        #pragma omp single
        {
            num_cpu_threads = omp_get_num_threads();
        }
    }
    // std::cout << "thread_concurrency: " << thread_concurrency << std::endl;
    std::cout << "free_cpu_memory: " << get_free_cpu_memory() << std::endl;
    std::cout << "free_gpu_memory: " << get_free_gpu_memory() << std::endl;

    std::cout << "num_cpu_threads: " << num_cpu_threads << std::endl;
    //print slurm variables
    // const char* slurm_cpus_per_task = std::getenv("SLURM_CPUS_PER_TASK");
    // std::cout << "SLURM_CPUS_PER_TASK: " << (slurm_cpus_per_task == nullptr ? "null" : slurm_cpus_per_task) << std::endl;
    // const char* slurm_cpus_per_node = std::getenv("SLURM_JOB_CPUS_PER_NODE");
    // std::cout << "SLURM_JOB_CPUS_PER_NODE: " << (slurm_cpus_per_node == nullptr ? "null" : slurm_cpus_per_node) << std::endl;
    // const char* slurm_cpus_on_node = std::getenv("SLURM_CPUS_ON_NODE");
    // std::cout << "SLURM_CPUS_ON_NODE: " << (slurm_cpus_on_node == nullptr ? "null" : slurm_cpus_on_node) << std::endl;
    // const char* slurm_cpus_per_gpu = std::getenv("SLURM_CPUS_PER_GPU");
    // std::cout << "SLURM_CPUS_PER_GPU: " << (slurm_cpus_per_gpu == nullptr ? "null" : slurm_cpus_per_gpu) << std::endl;
    // const char* slurm_mem_per_cpu = std::getenv("SLURM_MEM_PER_CPU");
    // std::cout << "SLURM_MEM_PER_CPU: " << (slurm_mem_per_cpu == nullptr ? "null" : slurm_mem_per_cpu) << std::endl;
    SBWTContainerGPU sbwt_gpu(sbwt_cpu);
    std::cout << "built gpu sbwt" << std::endl;
    std::cout << "free_cpu_memory: " << get_free_cpu_memory() << std::endl;
    std::cout << "free_gpu_memory: " << get_free_gpu_memory() << std::endl;

    sbwt_cpu.clear();//not needed anymore
    std::cout << "cleared cpu sbwt" << std::endl;
    std::cout << "free_cpu_memory: " << get_free_cpu_memory() << std::endl;
    std::cout << "free_gpu_memory: " << get_free_gpu_memory() << std::endl;

    // uint64_t size = static_cast<uint64_t>(2) * 1024 * 1024 * 1024 / sizeof(uint64_t);
    // std::vector<uint64_t> array(size);
    // std::vector<uint64_t> out(size);
    // for (uint64_t i = 0; i < size; ++i) {
    //   array[i] = i;
    // }
    // std::cout << "size: " << size << std::endl;
    // // uint64_t *d_array{};
    // // uint64_t *d_out{};
    // // CUDA_CHECK(cudaMalloc(&d_array, size * sizeof(uint64_t)));
    // // CUDA_CHECK(cudaMalloc(&d_out, size * sizeof(uint64_t)));
    // // CUDA_CHECK(
    // //     cudaMemcpy(d_array, array.data(), size * sizeof(uint64_t), cudaMemcpyHostToDevice));
    // GpuPointer<u64> d_array(array);
    // GpuPointer<u64> d_out(size);
    // float ms;
    // // cudaEvent_t start, stop;
    // // CUDA_CHECK(cudaEventCreate(&start));
    // // CUDA_CHECK(cudaEventCreate(&stop));
    // // CUDA_CHECK(cudaEventRecord(start, 0));
    // std::cout << "launching kernel" << std::endl;
    // // GpuStream default_stream(true);
    // GpuEvent start, stop;
    // start.record();
    // // shifted_array<<<dim3(size/1024), dim3(1024)>>>(
    // //   d_array,
    // //   d_out,
    // //   size
    // // );
    // launch_shifted_array(d_array, d_out, size, stop);
    // // CUDA_CHECK(cudaGetLastError());
    // // CUDA_CHECK(cudaEventRecord(stop, 0));
    // // CUDA_CHECK(cudaEventSynchronize(stop));
    // // CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    // ms = stop.time_elapsed_ms(start);
    // std::cout << "time: " << ms << std::endl;
    // // CUDA_CHECK(cudaMemcpy(out.data(), d_out, size * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    // d_out.copy_to(out);
    // // CUDA_CHECK(cudaEventDestroy(start));
    // // CUDA_CHECK(cudaEventDestroy(stop));
    // // CUDA_CHECK(cudaFree(d_array));
    // // CUDA_CHECK(cudaFree(d_out));

    // //print out some of the indexes
    // for (uint64_t i = 0; i < 10; ++i) {
    //   std::cout << array[i] << " : " << out[i] << std::endl;
    // }
    return 0;
}

// compile command:
// nvcc -gencode arch=compute_80,code=sm_80 -o run -x cu main.cpp

// make cubin:
// nvcc -gencode arch=compute_80,code=sm_80 -O3 -x cu main.cpp --cubin
// nvcc -gencode arch=compute_80,code=sm_80 -x cu main.cpp --cubin
// then
// cuobjdump -sass main.cubin > main.sass

// nvcc -gencode arch=compute_80,code=sm_80 -x cu main.cpp --cubin && cuobjdump -sass main.cubin > main.sass
