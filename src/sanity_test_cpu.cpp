#include "utils.h"
#include "sanity_test.h"
namespace sbwt_lcs_gpu {

void sanity_test_cpu(){
    uint64_t test_number = 0x0123456789AB0000;
    test_number |= ((uint64_t)0b1111)<<7;

    std::vector<uint64_t> vec_u64 = {test_number};
    std::vector<uint8_t> vec_char(reinterpret_cast<uint8_t*>(vec_u64.data()),
                               reinterpret_cast<uint8_t*>(vec_u64.data()) + sizeof(uint64_t));
    //test
    if (vec_char[7]!=0x01 || (vec_char[0])!=0b10000000 || vec_char[1]!=0b00000111){
        std::stringstream ss;
        // std::cout << "Error: " << std::endl;
        // std::cout << "Byte 7: " << (uint64_t) vec_char[7] << std::endl;
        // std::cout << "Byte 0: " << (uint64_t) vec_char[0] << std::endl;
        // std::cout << "Byte 1: " << (uint64_t) vec_char[1] << std::endl;
        ss << "CPU Error: " << std::endl;
        ss << "Byte 7: " << (uint64_t) vec_char[7] << std::endl;
        ss << "Byte 0: " << (uint64_t) vec_char[0] << std::endl;
        ss << "Byte 1: " << (uint64_t) vec_char[1] << std::endl;
        throw std::runtime_error(ss.str());
    }
}

}//namespace sbwt_lcs_gpu