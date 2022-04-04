#include "paddle/phi/kernels/softmax_with_cross_entropy_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void SoftmaxWithCrossEntropyGradKernel(
    const Context& dev_ctx, const phi::DenseTensor& labels,
    const phi::DenseTensor& softmax, const phi::DenseTensor& backprop,
    const phi::DenseTensor& loss_grad, bool soft_label, bool use_softmax,
    bool numeric_stable_mode, int ignore_index, int axis,
    phi::DenseTensor* logits_grad) {}

}  // namespace phi

PD_REGISTER_KERNEL(softmax_with_cross_entropy_grad, CPU, ALL_LAYOUT,
                   phi::SoftmaxWithCrossEntropyGradKernel, float) {}

