#include "paddle/phi/kernels/softmax_with_cross_entropy_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void SoftmaxWithCrossEntropyKernel(const Context& dev_ctx,
                                   const phi::DenseTensor& logits,
                                   const phi::DenseTensor& labels,
                                   bool soft_label, bool use_softmax,
                                   bool numeric_stable_mode, int ignore_index,
                                   int axis, phi::DenseTensor* softmax,
                                   phi::DenseTensor* loss,
                                   phi::DenseTensor* backprop) {}

}  // namespace phi

PD_REGISTER_KERNEL(softmax_with_cross_entropy, CPU, ALL_LAYOUT,
                   phi::SoftmaxWithCrossEntropyKernel, float, double) {}
