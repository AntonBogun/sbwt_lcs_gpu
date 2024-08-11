#pragma once
#include "gpu/gpu_utils.h"
namespace sbwt_lcs_gpu {
    void launch_shifted_array(GpuPointer<u64>& array, GpuPointer<u64>& out, u64 max, GpuEvent& stop_event);
    void print_supported_devices();
}
