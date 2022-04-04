/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "npu_op_runner.h"

namespace custom_kernel {

static inline int SizeToAxis(const int axis, phi::DDim dims) {
  int size = 1;
  for (int i = 0; i < axis; i++) {
    size *= dims[i];
  }
  return size;
}

static inline int SizeFromAxis(const int axis, phi::DDim dims) {
  int size = 1;
  for (int i = axis; i < dims.size(); i++) {
    size *= dims[i];
  }
  return size;
}

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
                   phi::DenseTensor* backprop) {
    PADDLE_ENFORCE_EQ(soft_label, false,
                      phi::errors::Unimplemented(
                          "soft_label=True is not supported in "
                          "the npu kernel of softmax_with_cross_entropy."));

    const int rank = logits.dims().size();
    const int use_axis = axis < 0 ? axis + rank : axis;
    const int n = SizeToAxis(use_axis, logits.dims());
    const int d = SizeFromAxis(use_axis, logits.dims());

    PADDLE_ENFORCE_EQ(
        labels.numel(), n,
        phi::errors::Unimplemented(
            "The size of labels should be equal to phi::funcs::SizeToAxis of "
            "logits,"
            "but got size of labels is %d and phi::funcs::SizeToAxis is %d.",
            labels.numel(), n));

    dev_ctx.template Alloc<T>(loss);
    dev_ctx.template Alloc<T>(backprop);
    dev_ctx.template Alloc<T>(softmax);

    phi::DenseTensor logits_2d, labels_1d, loss_1d, backprop_2d, softmax_2d;
    //logits_2d.ShareDataWith(logits).Resize({n, d});
    //labels_1d.ShareDataWith(labels).Resize({n});
    //loss_1d.ShareDataWith(*loss).Resize({n});
    //backprop_2d.ShareDataWith(*backprop).Resize({n, d});
    //softmax_2d.ShareDataWith(*softmax).Resize({n, d});

    logits_2d = logits;
    logits_2d .Resize({n, d});
    labels_1d = labels;
    labels_1d.Resize({n});
    loss_1d = *loss;
    loss_1d.Resize({n});
    backprop_2d = *backprop;
    backprop_2d.Resize({n, d});
    softmax_2d = *softmax;
    softmax_2d.Resize({n, d});
    
    auto stream = dev_ctx.stream();

    std::vector<int> axes;
    for (auto i = use_axis; i < logits.dims().size(); ++i) {
      axes.push_back(i);
    }
    const auto& runner_softmax =
        NpuOpRunner("SoftmaxV2", {logits}, {*softmax}, {{"axes", axes}});
    runner_softmax.Run(stream);

    // SparseSoftmaxCrossEntropyWithLogits
    const auto& runner_s =
        NpuOpRunner("SparseSoftmaxCrossEntropyWithLogits",
                    {logits_2d, labels_1d}, {loss_1d, backprop_2d}, {});
    runner_s.Run(stream);
}

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
                   phi::DenseTensor* logits_grad) {
    PADDLE_ENFORCE_NE(phi::product(backprop.dims()), 0,
                            phi::errors::PreconditionNotMet(
                                "backprop should not be null in NPU kernel of "
                                "softmax_with_cross_entropy_grad."));
    dev_ctx.template Alloc<T>(logits_grad);
    const int rank = logits_grad->dims().size();
    const int use_axis = axis < 0 ? axis + rank : axis;
    const int n = SizeToAxis(use_axis, logits_grad->dims());
    const int d = SizeFromAxis(use_axis, logits_grad->dims());

    phi::DenseTensor logits_grad_2d, loss_grad_1d, backprop_2d;

    //logits_grad_2d.ShareDataWith(*logits_grad).Resize({n, d});
    //loss_grad_1d.ShareDataWith(loss_grad).Resize({n});
    //backprop_2d.ShareDataWith(backprop).Resize({n, d});

    logits_grad_2d = *logits_grad;
    logits_grad_2d.Resize({n, d});
    loss_grad_1d = loss_grad;
    loss_grad_1d.Resize({n});
    backprop_2d = backprop;
    backprop_2d.Resize({n, d});
    
    auto stream = dev_ctx.stream();
    const auto& runner_mul =
        NpuOpRunner("Mul", {loss_grad, backprop}, {*logits_grad}, {});
    runner_mul.Run(stream);
}

}  // namespace custom_kernel 

PD_REGISTER_PLUGIN_KERNEL(
    softmax_with_cross_entropy, Ascend910, ALL_LAYOUT, custom_kernel::SoftmaxWithCrossEntropyKernel, float, phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(
    softmax_with_cross_entropy_grad, Ascend910, ALL_LAYOUT, custom_kernel::SoftmaxWithCrossEntropyGradKernel, float, phi::dtype::float16) {}

