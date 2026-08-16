#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void fp8_attn_input_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                           Tensor& k, Tensor& v, cudaStream_t stream);

} // namespace ninfer::ops::detail
