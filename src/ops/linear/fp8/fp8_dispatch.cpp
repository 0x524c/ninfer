#include "ops/linear/fp8/fp8_dispatch.h"

#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_format.h"
#include "ops/linear/fp8/fp8_t1.h"

#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

void validate_profile(std::int32_t output_rows, std::int32_t input_rows, LinearPolicy policy,
                      std::int32_t tokens, const char* operation) {
    if (!is_fp8_linear_problem(output_rows, input_rows) ||
        (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA8) || tokens != 1) {
        throw std::invalid_argument(std::string(operation) + ": unsupported FP8 T=1 profile");
    }
}

} // namespace

std::size_t fp8_linear_workspace_capacity_bytes(std::int32_t output_rows, std::int32_t input_rows,
                                                LinearPolicy policy, std::int32_t min_tokens,
                                                std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("fp8 linear workspace: invalid token interval");
    }
    validate_profile(output_rows, input_rows, policy, min_tokens, "fp8 linear workspace");
    validate_profile(output_rows, input_rows, policy, max_tokens, "fp8 linear workspace");
    return 0;
}

void fp8_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                  cudaStream_t stream) {
    validate_fp8_weight(weight, "fp8 linear");
    validate_profile(weight.n, weight.k, policy, x.ne[1], "fp8 linear");
    launch_fp8_t1(x, weight, out, stream);
}

} // namespace ninfer::ops::detail
