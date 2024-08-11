#pragma once
#include <cuda_runtime.h>
#include "gpu_utils.h"

namespace sbwt_lcs_gpu {



//shouldn't put implementation here otherwise it will be compiled in every file that includes this header (multiple definition error)
cudaStream_t* getStream(GpuStream& stream);

cudaEvent_t* getEvent(GpuEvent& event);


}
