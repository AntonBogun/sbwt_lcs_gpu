#include "utils.h"
#include "sanity_test.h"
#include "structs.hpp"
#include <type_traits>
namespace sbwt_lcs_gpu {

static_assert(std::is_standard_layout_v<GPUThreadLookupTableEntry>,"GPUThreadLookupTableEntry must be standard layout");
static_assert(std::is_trivially_copyable_v<GPUThreadLookupTableEntry>,"GPUThreadLookupTableEntry must be trivially copyable");

void sanity_test_cpu(){
    try{
        u64 test_number = 0x0123456789AB0000;
        test_number |= ((u64)0b1111)<<7;

        std::vector<u64> vec_u64 = {test_number};
        OffsetVector<u8> vec_char(vec_u64);
        vec_char.resize(8);
        //test
        if (vec_char[7]!=0x01 || (vec_char[0])!=0b10000000 || vec_char[1]!=0b00000111){
            std::stringstream ss;
            // std::cout << "Error: " << std::endl;
            // std::cout << "Byte 7: " << (u64) vec_char[7] << std::endl;
            // std::cout << "Byte 0: " << (u64) vec_char[0] << std::endl;
            // std::cout << "Byte 1: " << (u64) vec_char[1] << std::endl;
            ss << "CPU Error: " << std::endl;
            ss << "Byte 7: " << (u64) vec_char[7] << std::endl;
            ss << "Byte 0: " << (u64) vec_char[0] << std::endl;
            ss << "Byte 1: " << (u64) vec_char[1] << std::endl;
            throw std::runtime_error(ss.str());
        }
    }catch(std::exception& e){
        std::stringstream ss;
        ss << "CPU Error while testing OffsetVector<char>: " << std::endl;
        ss << e.what() << std::endl;        
        throw std::runtime_error(ss.str());
    }
    try{
        std::vector<u64> lookup_table_test(10,0);
        OffsetVector<i32> reinterpret_table(lookup_table_test);
        reinterpret_table.resize(20);

        for (i32 i=0;i<sizeof(GPUThreadLookupTableEntry)/sizeof(i32)*2;i++){
            reinterpret_table[i+2] = i;
        }
        OffsetVector<GPUThreadLookupTableEntry> lookup_table(lookup_table_test,1,6);
        lookup_table.resize(2);
        for(int i=0;i<sizeof(GPUThreadLookupTableEntry)/sizeof(i32)*2;i+=sizeof(GPUThreadLookupTableEntry)/sizeof(i32)){
            auto& entry = lookup_table[i*sizeof(i32)/sizeof(GPUThreadLookupTableEntry)];
            if (entry.substract!=i || entry.read_off!=i+1 || entry.sep_off!=i+2 || entry.bit_off!=i+3 || entry.rank_off!=i+4 || entry.out_off!=i+5){
                std::stringstream ss;
                ss << "CPU Error: " << std::endl;
                ss << "entry at index " << i << " is not correct" << std::endl;
                ss << "entry.substract: " << entry.substract << std::endl;
                ss << "entry.read_off: " << entry.read_off << std::endl;
                ss << "entry.sep_off: " << entry.sep_off << std::endl;
                ss << "entry.bit_off: " << entry.bit_off << std::endl;
                ss << "entry.rank_off: " << entry.rank_off << std::endl;
                ss << "entry.out_off: " << entry.out_off << std::endl;
                throw std::runtime_error(ss.str());
            }
        }
    }catch(std::exception& e){
        std::stringstream ss;
        ss << "CPU Error while testing OffsetVector<GPUThreadLookupTableEntry>: " << std::endl;
        ss << e.what() << std::endl;
        throw std::runtime_error(ss.str());
    }
}

}//namespace sbwt_lcs_gpu