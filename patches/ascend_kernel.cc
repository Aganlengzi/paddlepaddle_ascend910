#include "paddle/phi/kernels/ascend_kernel.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void MinRawKernel(const Context& dev_ctx, const phi::DenseTensor& x,
                  const std::vector<int64_t>& axes, bool keep_dim,
                  bool reduce_all, phi::DenseTensorMeta::DataType out_dtype,
                  phi::DenseTensor* out) {
  out = out;
}

template <typename T, typename Context>
void MinKernel(const Context& dev_ctx, const phi::DenseTensor& x,
               const std::vector<int64_t>& dims,
               phi::DenseTensorMeta::DataType out_dtype, bool keep_dim,
               phi::DenseTensor* out) {
  out = out;
}

template <typename T, typename Context>
void MaxRawKernel(const Context& dev_ctx, const phi::DenseTensor& x,
                  const std::vector<int64_t>& axes, bool keep_dim,
                  bool reduce_all, phi::DenseTensorMeta::DataType out_dtype,
                  phi::DenseTensor* out) {
  out = out;
}

template <typename T, typename Context>
void MeanGradKernel(const Context& dev_ctx, const phi::DenseTensor& x,
                    const std::vector<int64_t>& dim, bool keep_dim,
                    phi::DenseTensor* out) {
  out = out;
}

template <typename T, typename Context>
void MeanRawGradKernel(const Context& dev_ctx, const phi::DenseTensor& x,
                       const phi::DenseTensor& out_grad,
                       const std::vector<int64_t>& dim, bool keep_dim,
                       bool reduce_all, phi::DenseTensor* x_grad) {}

template <typename T, typename Context>
void SliceKernel(const Context& dev_ctx, const phi::DenseTensor& x,
                 const phi::ScalarArray& axes_array,
                 const phi::ScalarArray& starts_array,
                 const phi::ScalarArray& ends_array, phi::DenseTensor* out) {}

template <typename T, typename Context>
void SliceGradKernel(const Context& dev_ctx, const phi::DenseTensor& x,
                     const phi::DenseTensor& out_grad,
                     const phi::ScalarArray& axes_array,
                     const phi::ScalarArray& starts_array,
                     const phi::ScalarArray& ends_array,
                     phi::DenseTensor* x_grad) {}

template <typename T, typename Context>
void AdamKernel(const Context& dev_ctx, const phi::DenseTensor& param,
                const phi::DenseTensor& grad,
                const phi::DenseTensor& learning_rate,
                const phi::DenseTensor& moment1,
                const phi::DenseTensor& moment2,
                const phi::DenseTensor& beta1_pow_in,
                const phi::DenseTensor& beta2_pow_in,
                /*
                                const phi::DenseTensor& beta1_tensor,
                                const phi::DenseTensor& beta2_tensor,
                                const phi::DenseTensor& epsilon_tensor,
                                const phi::DenseTensor& master_param,
                                const phi::DenseTensor& skip_update,
                */
                float beta1_f, float beta2_f, float epsilon_f, bool lazy_mode,
                int64_t min_row_size_to_use_multithread, bool multi_precision,
                bool use_global_beta_pow, phi::DenseTensor* param_out,
                phi::DenseTensor* moment1_out, phi::DenseTensor* moment2_out,
                phi::DenseTensor* beta1_pow_out, phi::DenseTensor* beta2_pow_out
                /* phi::DenseTensor* master_param_out */
                ){};

template <typename T, typename Context>
void MomentKernel(const Context& dev_ctx, const phi::DenseTensor& param,
                  const phi::DenseTensor& grad,
                  const phi::DenseTensor& velocity,
                  const phi::DenseTensor& learning_rate, float mu_f,
                  bool use_nesterov, const std::string& regularization_method,
                  float regularization_coeff, bool multi_precision,
                  float rescale_grad, phi::DenseTensor* param_out,
                  phi::DenseTensor* velocity_out) {}

template <typename T, typename Context>
void SoftmaxWithCrossEntropyKernel(const Context& dev_ctx,
                                   const phi::DenseTensor& logits,
                                   const phi::DenseTensor& labels,
                                   bool soft_label, bool use_softmax,
                                   bool numeric_stable_mode, int ignore_index,
                                   int axis, phi::DenseTensor* softmax,
                                   phi::DenseTensor* loss,
                                   phi::DenseTensor* backprop) {}

template <typename T, typename Context>
void SoftmaxWithCrossEntropyGradKernel(
    const Context& dev_ctx, const phi::DenseTensor& labels,
    const phi::DenseTensor& softmax, const phi::DenseTensor& backprop,
    const phi::DenseTensor& loss_grad, bool soft_label, bool use_softmax,
    bool numeric_stable_mode, int ignore_index, int axis,
    phi::DenseTensor* logits_grad) {}

}  // namespace phi

// PD_REGISTER_KERNEL(min_raw, CPU, ALL_LAYOUT, phi::MinRawKernel, float, double,
//                    bool) {}

// PD_REGISTER_KERNEL(min, CPU, ALL_LAYOUT, phi::MinKernel, float, double, bool) {}

PD_REGISTER_KERNEL(mean_raw_grad, CPU, ALL_LAYOUT, phi::MeanRawGradKernel,
                   float, double, bool) {}

// PD_REGISTER_KERNEL(mean_grad, CPU, ALL_LAYOUT, phi::MeanGradKernel, float,
//                    double, bool) {}

PD_REGISTER_KERNEL(slice, CPU, ALL_LAYOUT, phi::SliceKernel, float, double,
                   bool) {}

PD_REGISTER_KERNEL(slice_grad, CPU, ALL_LAYOUT, phi::SliceGradKernel, float,
                   double, bool) {}

PD_REGISTER_KERNEL(adam, CPU, ALL_LAYOUT, phi::AdamKernel, float, double) {}

PD_REGISTER_KERNEL(momentum, CPU, ALL_LAYOUT, phi::MomentKernel, float, double) {}

// PD_REGISTER_KERNEL(less_than, CPU, ALL_LAYOUT, phi::LessThanKernel, bool,
//                    int16_t, int, int64_t, float, double) {}
// PD_REGISTER_KERNEL(greater_equal, CPU, ALL_LAYOUT, phi::GreaterEqualKernel,
//                    bool, int16_t, int, int64_t, float, double) {}
// PD_REGISTER_KERNEL(equal, CPU, ALL_LAYOUT, phi::EqualKernel, bool, int16_t, int,
//                    int64_t, float, double) {}
// PD_REGISTER_KERNEL(not_equal, CPU, ALL_LAYOUT, phi::NotEqualKernel, bool,
//                    int16_t, int, int64_t, float, double) {}

PD_REGISTER_KERNEL(softmax_with_cross_entropy, CPU, ALL_LAYOUT,
                   phi::SoftmaxWithCrossEntropyKernel, float, double) {}

PD_REGISTER_KERNEL(softmax_with_cross_entropy_grad, CPU, ALL_LAYOUT,
                   phi::SoftmaxWithCrossEntropyGradKernel, float) {}
