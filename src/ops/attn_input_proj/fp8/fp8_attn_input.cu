#include "ops/attn_input_proj/fp8/fp8_attn_input.h"

#include "core/device.h"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_t1.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

struct Fp8AttentionInputOutput {
    __nv_bfloat16* query;
    __nv_bfloat16* key;
    __nv_bfloat16* gate;
    __nv_bfloat16* value;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t,
                                          float result) const {
        constexpr std::int32_t kQueryRows  = 6144;
        constexpr std::int32_t kKeyRows    = 1024;
        constexpr std::int32_t kGateRows   = 6144;
        constexpr std::int32_t kKeyBegin   = kQueryRows;
        constexpr std::int32_t kGateBegin  = kKeyBegin + kKeyRows;
        constexpr std::int32_t kValueBegin = kGateBegin + kGateRows;

        const __nv_bfloat16 value_bf16 = __float2bfloat16_rn(result);
        if (parent_row < kKeyBegin) {
            query[parent_row] = value_bf16;
        } else if (parent_row < kGateBegin) {
            key[parent_row - kKeyBegin] = value_bf16;
        } else if (parent_row < kValueBegin) {
            gate[parent_row - kGateBegin] = value_bf16;
        } else {
            value[parent_row - kValueBegin] = value_bf16;
        }
    }
};

} // namespace

void fp8_attn_input_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                           Tensor& k, Tensor& v, cudaStream_t stream) {
    using Geometry        = Fp8AttnInputGeometry;
    using Schedule        = Fp8AttnInputT1Schedule;
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const Fp8AttentionInputOutput output{
        static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(v.data),
    };
    fp8_t1_kernel<Geometry, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const __nv_bfloat16*>(weight.scales), output);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
