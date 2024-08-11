#include "utils.h"
#include "sanity_test.h"
namespace sbwt_lcs_gpu {

__global__ void testU64LayoutGPU(uint64_t* d_result) {
    uint64_t test_number = 0x0123456789AB0000;
    test_number |= ((uint64_t)0b1111) << 7;

    // Reinterpret as uint8_t array
    uint8_t* char_ptr = reinterpret_cast<uint8_t*>(&test_number);

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
}

void sanity_test_gpu(){
    uint64_t* d_result;
    cudaMalloc(&d_result, sizeof(uint64_t));

    // Call the kernel
    testU64LayoutGPU<<<1, 1>>>(d_result);

    // Copy the result back to the host
    uint64_t h_result;
    cudaMemcpy(&h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Check for errors
    if (cudaGetLastError() != cudaSuccess) {
        // std::cerr << "CUDA error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        throw std::runtime_error("CUDA error");
    }

    // Check the result
    if (h_result != 2) {
        // std::cerr << "Test failed! " << h_result << std::endl;
        throw std::runtime_error("Test failed " + std::to_string(h_result));
    } else {
        // std::cout << "Test passed! " << h_result << std::endl;
    }
    // Free the memory
    cudaFree(d_result);
}

}//namespace sbwt_lcs_gpu