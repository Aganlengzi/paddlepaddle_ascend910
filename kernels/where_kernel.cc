// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "npu_funcs.h"
#include "npu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void WhereKernel(const Context& ctx,
                 const phi::DenseTensor& condition,
                 const phi::DenseTensor& x,
                 const phi::DenseTensor& y,
                 phi::DenseTensor* out) {

    ctx.template Alloc<T>(out);
    const auto& runner =
        NpuOpRunner("Select", {condition, x, y}, {*out}, {});

    auto stream = ctx.stream();
    runner.Run(stream);
}


template <typename T, typename Context>
void WhereGradKernel(const Context& ctx,
                     const phi::DenseTensor& condition,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& y,
                     const phi::DenseTensor& out_grad,
                     phi::DenseTensor* x_grad,
                     phi::DenseTensor* y_grad) {

    if (x_grad != nullptr) {
      ctx.template Alloc<T>(x_grad);
    }
    if (y_grad != nullptr) {
      ctx.template Alloc<T>(y_grad);
    }

    auto stream = ctx.stream();

    //phi::DenseTensor tensor_zeros(out_grad.dtype());
    //tensor_zeros.Resize(out_grad.dims());
    phi::DenseTensor tensor_zeros;
    phi::DenseTensorMeta meta = {out_grad.dtype(), out_grad.dims()};
    tensor_zeros.set_meta(meta);

    ctx.template Alloc<T>(&tensor_zeros);

    const auto& runner =
        NpuOpRunner("ZerosLike", {out_grad}, {tensor_zeros}, {});
    runner.Run(stream);

    if (x_grad != nullptr) {
      const auto& runner = NpuOpRunner(
          "Select", {condition, out_grad, tensor_zeros}, {*x_grad}, {});
      runner.Run(stream);
    }
    if (y_grad != nullptr) {
      const auto& runner = NpuOpRunner(
          "Select", {condition, tensor_zeros, out_grad}, {*y_grad}, {});
      runner.Run(stream);
    }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(where,
                          Ascend910,
                          ALL_LAYOUT,
                          custom_kernel::WhereKernel, int32_t, int64_t,
                          double, float) {}

PD_REGISTER_PLUGIN_KERNEL(where_grad,
                          Ascend910,
                          ALL_LAYOUT,
                          custom_kernel::WhereGradKernel, int32_t, int64_t,
                          double, float) {}
