#include "ops/linear/fp8/fp8_t1.h"

#include "core/device.h"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_output.cuh"
#include "ops/linear/fp8/fp8_t1.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {

void launch_fp8_t1(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Geometry        = Fp8AttnInputGeometry;
    using Schedule        = Fp8AttnInputT1Schedule;
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const Fp8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), weight.n};
    fp8_t1_kernel<Geometry, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const __nv_bfloat16*>(weight.scales), output);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
