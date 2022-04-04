#include "paddle/phi/kernels/merged_momentum_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {


template <typename T, typename Context>
void MergedMomentumKernel(const Context& dev_ctx, const std::vector<const phi::DenseTensor *> &params,
                  const std::vector<const phi::DenseTensor *>& grads,
                  const std::vector<const phi::DenseTensor *>& velocitys,
                  const std::vector<const phi::DenseTensor *>& lrs, float mu_f,
                  bool use_nesterov, const std::vector<std::string>& regularization_methods,
                  const std::vector<float>& regularization_coeffs, bool multi_precision,
                  float rescale_grad, std::vector<phi::DenseTensor*> params_out,
                  std::vector<phi::DenseTensor*> velocitys_out) {}

}  // namespace phi

PD_REGISTER_KERNEL(merged_momentum, CPU, ALL_LAYOUT, phi::MergedMomentumKernel, float, double) {}

