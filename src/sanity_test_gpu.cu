#include "utils.h"
#include "sanity_test.h"
#include "structs.hpp"
namespace sbwt_lcs_gpu {

__global__ void testU64LayoutGPU(u64* d_result, u64* vec) {
    d_result[0] = 0;
    d_result[1] = 0;
    u64 test_number = 0x0123456789AB0000;
    test_number |= ((u64)0b1111) << 7;

    // Reinterpret as u8 array
    u8* char_ptr = reinterpret_cast<u8*>(&test_number);

    // Perform the test
    if (char_ptr[7] != 0x01 || char_ptr[0] != 0b10000000 || char_ptr[1] != 0b00000111) {
        // If test fails, set result to 1
    //     *d_result = 1;
    // } else {
    //     // If test passes, set result to 0
    //     *d_result = 0;
        // std::cout << "GPU Error: " << std::endl;
        // std::cout << "Byte 7: " << (uint64_t) char_ptr[7] << std::endl;
        // std::cout << "Byte 0: " << (uint64_t) char_ptr[0] << std::endl;
        // std::cout << "Byte 1: " << (uint64_t) char_ptr[1] << std::endl;
        //use printf
        printf("GPU Error: Byte 7: %d\n", (int) char_ptr[7]);
        printf("GPU Error: Byte 0: %d\n", (int) char_ptr[0]);
        printf("GPU Error: Byte 1: %d\n", (int) char_ptr[1]);
        d_result[0] = 1;
    }
    else{
        d_result[0] = 2;
    }

    //     std::vector<u64> lookup_table_test(10,0);
    // OffsetVector<i32> reinterpret_table(lookup_table_test);
    // reinterpret_table.resize(20);

    // for (i32 i=0;i<sizeof(GPUThreadLookupTableEntry)/sizeof(i32)*2;i++){
    //     reinterpret_table[i+2] = i;
    // }
    // OffsetVector<GPUThreadLookupTableEntry> lookup_table(lookup_table_test,1,6);
    // lookup_table.resize(2);
    // for(int i=0;i<sizeof(GPUThreadLookupTableEntry)/sizeof(i32)*2;i+=sizeof(GPUThreadLookupTableEntry)/sizeof(i32)){
    //     auto& entry = lookup_table[i*sizeof(i32)/sizeof(GPUThreadLookupTableEntry)];
    //     if (entry.substract!=i || entry.read_off!=i+1 || entry.sep_off!=i+2 || entry.bit_off!=i+3 || entry.rank_off!=i+4 || entry.out_off!=i+5){
    //         std::stringstream ss;
    //         ss << "CPU Error: " << std::endl;
    //         ss << "entry at index " << i << " is not correct" << std::endl;
    //         ss << "entry.substract: " << entry.substract << std::endl;
    //         ss << "entry.read_off: " << entry.read_off << std::endl;
    //         ss << "entry.sep_off: " << entry.sep_off << std::endl;
    //         ss << "entry.bit_off: " << entry.bit_off << std::endl;
    //         ss << "entry.rank_off: " << entry.rank_off << std::endl;
    //         ss << "entry.out_off: " << entry.out_off << std::endl;
    //         throw std::runtime_error(ss.str());
    //     }
    // }
    //do above but gpu
    i32* reinterpret_table=reinterpret_cast<i32*>(vec);
    for (i32 i=0;i<sizeof(GPUThreadLookupTableEntry)/sizeof(i32)*2;i++){
        reinterpret_table[i+2] = i;
    }
    GPUThreadLookupTableEntry* lookup_table=reinterpret_cast<GPUThreadLookupTableEntry*>(vec+1);
    for(int i=0;i<sizeof(GPUThreadLookupTableEntry)/sizeof(i32)*2;i+=sizeof(GPUThreadLookupTableEntry)/sizeof(i32)){
        auto& entry = lookup_table[i*sizeof(i32)/sizeof(GPUThreadLookupTableEntry)];
        if (entry.substract!=i || entry.read_off!=i+1 || entry.sep_off!=i+2 || entry.bit_off!=i+3 || entry.rank_off!=i+4 || entry.out_off!=i+5){
            printf("GPU Error: entry at index %d is not correct\n", i);
            printf("GPU Error: entry.substract: %d\n", entry.substract);
            printf("GPU Error: entry.read_off: %d\n", entry.read_off);
            printf("GPU Error: entry.sep_off: %d\n", entry.sep_off);
            printf("GPU Error: entry.bit_off: %d\n", entry.bit_off);
            printf("GPU Error: entry.rank_off: %d\n", entry.rank_off);
            printf("GPU Error: entry.out_off: %d\n", entry.out_off);
            d_result[1] = 1;
        }
    }
    if (d_result[1]==0){
        d_result[1] = 2;
    }
}

void sanity_test_gpu(){
    uint64_t* d_result;
    uint64_t* vec;
    cudaMalloc(&d_result, sizeof(uint64_t)*2);
    cudaMalloc(&vec, sizeof(uint64_t)*10);

    // Call the kernel
    testU64LayoutGPU<<<1, 1>>>(d_result, vec);

    // Copy the result back to the host
    uint64_t h_result[2];
    cudaMemcpy(&h_result, d_result, sizeof(uint64_t)*2, cudaMemcpyDeviceToHost);

    // Check for errors
    if (cudaGetLastError() != cudaSuccess) {
        // std::cerr << "CUDA error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        throw std::runtime_error("CUDA error");
    }

    // Check the result
    if (h_result[0] != 2 || h_result[1] != 2) {
        // std::cerr << "Test failed! " << h_result << std::endl;
        throw std::runtime_error("Test failed " + std::to_string(h_result[0]) + " " + std::to_string(h_result[1]));
    } else {
        // std::cout << "Test passed! " << h_result << std::endl;
    }
    // Free the memory
    cudaFree(d_result);
    cudaFree(vec);
}

}//namespace sbwt_lcs_gpu