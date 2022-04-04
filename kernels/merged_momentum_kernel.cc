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

enum class RegularizationType {
  kNONE = 0,
  kL1DECAY = 1,  // do not need support right now
  kL2DECAY = 2,
};

template <typename T, typename Context>
void MergedMomentumKernel(const Context& dev_ctx, const std::vector<const phi::DenseTensor *> &params,
                  const std::vector<const phi::DenseTensor *>& grads,
                  const std::vector<const phi::DenseTensor *>& velocitys,
                  const std::vector<const phi::DenseTensor *>& lrs, float mu_f,
                  bool use_nesterov, const std::vector<std::string>& regularization_methods,
                  const std::vector<float>& regularization_coeffs, bool multi_precision,
                  float rescale_grad, std::vector<phi::DenseTensor*> params_out,
                  std::vector<phi::DenseTensor*> velocitys_out) {
    size_t n = params.size();
    PADDLE_ENFORCE_EQ(n, params_out.size(),
                      phi::errors::InvalidArgument(
                          "The size of Output(ParamOut) must be equal to "
                          "Input(Param), but got the size of Output(ParamOut) "
                          "is %d, the size of Input(Param) is %d.",
                          params_out.size(), n));
    for (size_t i = 0; i < n; ++i) {
      PADDLE_ENFORCE_EQ(params[i], params_out[i],
                        phi::errors::InvalidArgument(
                            "The size of Input(Param) and Output(ParamOut) "
                            "must be the same Tensors."));
    }

    PADDLE_ENFORCE_EQ(
        n, grads.size(),
        phi::errors::InvalidArgument(
            "The size of Input(Grad) must be equal to Input(Param), but got "
            "the size of Input(Grad) is %d, the size of Input(Param) is %d.",
            grads.size(), n));

    PADDLE_ENFORCE_EQ(n, velocitys.size(),
                      phi::errors::InvalidArgument(
                          "The size of Input(Velocity) must be equal to "
                          "Input(Param), but got the size of Input(Velocity) "
                          "is %d, the size of Input(Param) is %d.",
                          velocitys.size(), n));

    PADDLE_ENFORCE_EQ(
        n, velocitys_out.size(),
        phi::errors::InvalidArgument(
            "The size of Output(VelocityOut) must be "
            "equal to Input(Param), but got the size of Output(VelocityOut) is "
            "%d, the size of Input(Param) is %d.",
            velocitys_out.size(), n));
    for (size_t i = 0; i < n; ++i) {
      PADDLE_ENFORCE_EQ(velocitys[i], velocitys_out[i],
                        phi::errors::InvalidArgument(
                            "Input(Velocity) and Output(VelocityOut) must be "
                            "the same Tensors."));
    }

    T mu = static_cast<T>(mu_f);
    if (lrs.size() != 1) {
      PADDLE_ENFORCE_EQ(
          n, lrs.size(),
          phi::errors::InvalidArgument(
              "If the size of Input(LearningRate) is not 1, the size of "
              "Input(LearningRate) must be "
              "equal to Input(Param), but got the size of Input(LearningRate) "
              "is %d, the size of Input(Param) is %d.",
              lrs.size(), n));
    }

    if (regularization_methods.size() != 0) {
      PADDLE_ENFORCE_EQ(
          n, regularization_methods.size(),
          phi::errors::InvalidArgument(
              "The size of Attr(regularization_method) must be equal "
              "to Input(Param), but got the size of "
              "Attr(regularization_method) is %d, the size of Input(Param) is "
              "%d.",
              regularization_methods.size(), n));
      PADDLE_ENFORCE_EQ(
          n, regularization_coeffs.size(),
          phi::errors::InvalidArgument(
              "The size of Attr(regularization_coeff) must be equal "
              "to Input(Param), but got the size of Attr(regularization_coeff) "
              "is %d, the size of Input(Param) is %d.",
              regularization_coeffs.size(), n));
    }

    VLOG(5) << "use_nesterov: " << use_nesterov
            << ",  regularization_methods.size(): "
            << regularization_methods.size()
            << ",  regularization_coeffs.size(): "
            << regularization_coeffs.size();

    phi::DenseTensor mu_tensor;
    mu_tensor.Resize(phi::make_ddim({1}));
    dev_ctx.template Alloc<T>(&mu_tensor);
    FillNpuTensorWithConstant<T>(&mu_tensor, dev_ctx, mu);

    for (size_t idx = 0; idx < n; ++idx) {
      RegularizationType regularization_flag =
          regularization_methods.size() > 0 &&
                  regularization_methods[idx] == "l2_decay"
              ? RegularizationType::kL2DECAY
              : RegularizationType::kNONE;
      float regularization_coeff = 0.0;
      if (regularization_coeffs.size() != 0) {
        regularization_coeff = regularization_coeffs[idx];
      }

      auto learning_rate = lrs.size() > 1 ? lrs[idx] : lrs[0];
      auto param = params[idx];
      auto param_out = params_out[idx];
      auto velocity = velocitys[idx];
      auto velocity_out = velocitys_out[idx];

      auto grad = grads[idx];
      phi::DenseTensor regularized_grad;
      if (regularization_flag == RegularizationType::kL2DECAY) {
        regularized_grad.Resize(grad->dims());
        dev_ctx.template Alloc<T>(&regularized_grad);
        const auto& runner1 = NpuOpRunner("Muls", {*param}, {regularized_grad},
                                          {{"value", regularization_coeff}});
        runner1.Run(dev_ctx.stream());
        const auto& runner2 = NpuOpRunner("Add", {regularized_grad, *grad},
                                          {regularized_grad}, {});
        runner2.Run(dev_ctx.stream());
      } else {
        //regularized_grad.ShareDataWith(*grad);
        regularized_grad = *grad;
      }
      TensorCopy(dev_ctx, *param, false, param_out);
      TensorCopy(dev_ctx, *velocity, false, velocity_out);
      // NOTE: ApplyMomentum will change the input
      const auto& runner = NpuOpRunner(
          "ApplyMomentum", {*param_out, *velocity_out, *learning_rate,
                            regularized_grad, mu_tensor},
          {*param_out}, {{"use_nesterov", use_nesterov}});
      runner.Run(dev_ctx.stream());
    }
}

}  // namespace custom_kernel

//PD_REGISTER_PLUGIN_KERNEL(merged_momentum, Ascend910, ALL_LAYOUT,
//                          custom_kernel::MergedMomentumKernel, phi::dtype::float16,
//                          float, double) {}
