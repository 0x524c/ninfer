#include "ops/linear/linear_test_common.h"
#include "ops/linear/fp8/fp8_format.h"

#include <array>
#include <exception>
#include <iostream>

namespace {

using namespace ninfer;
using namespace ninfer::test::linear;

int run_fp8_a16() {
    constexpr std::array invocations{
        Invocation{1, CallForm::A16Convenience, ops::LinearPolicy::A16Only},
        Invocation{1, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    int failures = run_shape("FP8_A16", ActivationCompute::A16, make_fp8_weight,
                             {14336, 5120, 811U, Comparison::Sampled, true, invocations});

    auto packed = make_fp8_weight(14336, 5120, 823U);
    try {
        (void)ops::detail::validate_fp8_weight(packed.weight, "FP8 validator test");
    } catch (const std::exception& error) {
        std::cerr << "valid FP8 metadata was rejected: " << error.what() << '\n';
        ++failures;
    }
    const auto expect_invalid = [&](const char* label, Weight invalid) {
        try {
            (void)ops::detail::validate_fp8_weight(invalid, "FP8 validator test");
            std::cerr << "invalid FP8 " << label << " was accepted\n";
            ++failures;
        } catch (const std::invalid_argument&) {}
    };
    Weight invalid = packed.weight;
    invalid.layout = QuantLayout::Contiguous;
    expect_invalid("layout", invalid);
    invalid             = packed.weight;
    invalid.scale_nb[1] = invalid.scale_nb[1] - 2;
    expect_invalid("scale stride", invalid);
    invalid               = packed.weight;
    invalid.payload_bytes = invalid.payload_bytes - 1;
    expect_invalid("payload bound", invalid);
    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const int failures = run_fp8_a16();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " FP8 A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "FP8 A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
