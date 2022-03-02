
#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"

namespace phi {

template <typename T, typename Context>
void MinRawKernel(const Context& dev_ctx, const phi::DenseTensor& x,
                  const std::vector<int64_t>& axes, bool keep_dim,
                  bool reduce_all, phi::DenseTensorMeta::DataType out_dtype,
                  phi::DenseTensor* out);

template <typename T, typename Context>
void MinKernel(const Context& dev_ctx, const phi::DenseTensor& x,
               const std::vector<int64_t>& dims,
               phi::DenseTensorMeta::DataType out_dtype, bool keep_dim,
               phi::DenseTensor* out);

template <typename T, typename Context>
void MaxRawKernel(const Context& dev_ctx, const phi::DenseTensor& x,
                  const std::vector<int64_t>& axes, bool keep_dim,
                  bool reduce_all, phi::DenseTensorMeta::DataType out_dtype,
                  phi::DenseTensor* out);

template <typename T, typename Context>
void MaxKernel(const Context& dev_ctx, const phi::DenseTensor& x,
               const std::vector<int64_t>& dims,
               phi::DenseTensorMeta::DataType out_dtype, bool keep_dim,
               phi::DenseTensor* out);

template <typename T, typename Context>
void ArgsortKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               int axis,
               bool descending,
               phi::DenseTensor* output,
               phi::DenseTensor* indices);

#define DECALRE_COMPARE_KERNEL(compare_kernel) \
  template <typename T, typename Context>      \
  void compare_kernel(const Context& ctx,      \
                      const DenseTensor& x,    \
                      const DenseTensor& y,    \
                      int axis,                \
                      DenseTensor* out);

DECALRE_COMPARE_KERNEL(LessThanKernel)
DECALRE_COMPARE_KERNEL(GreaterEqualKernel)
DECALRE_COMPARE_KERNEL(EqualKernel)
DECALRE_COMPARE_KERNEL(NotEqualKernel)
#undef DECALRE_COMPARE_KERNEL

template <typename T, typename Context>
void SoftmaxWithCrossEntropyKernel(const Context& dev_ctx,
                   const phi::DenseTensor& logits,
                   const phi::DenseTensor& labels,
                   bool soft_label,
                   bool use_softmax,
                   bool numeric_stable_mode,
                   int ignore_index,
                   int axis,
                   phi::DenseTensor* softmax,
                   phi::DenseTensor* loss,
                   phi::DenseTensor* backprop);

template <typename T, typename Context>
void SoftmaxWithCrossEntropyGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& labels,
                   const phi::DenseTensor& softmax,
                   const phi::DenseTensor& backprop,
                   const phi::DenseTensor& loss_grad,
                   bool soft_label,
                   bool use_softmax,
                   bool numeric_stable_mode,
                   int ignore_index,
                   int axis,
                   phi::DenseTensor* logits_grad);

}  // namespace phi
