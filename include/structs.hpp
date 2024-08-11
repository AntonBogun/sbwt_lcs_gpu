#pragma once
#include "utils.h"
namespace sbwt_lcs_gpu {
struct StreamFilenamesContainer{
    std::vector<std::string> filenames;
    std::vector<std::string> output_filenames;
    std::vector<i64> lengths;
    i64 total_length;
};

}//namespace sbwt_lcs_gpu