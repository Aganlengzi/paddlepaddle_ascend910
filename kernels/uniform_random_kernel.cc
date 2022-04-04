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
#include "npu_funcs.h"

namespace custom_kernel {

// borrowed from paddle/phi/kernels/cpu/uniform_random_kernel.cc
// if paddle/phi/kernels/uniform_random_kernel.h can be exposed,
// we can use CPU vesion for NPU
template <typename T>
inline void UniformRealDistribution(T *data,
                                    const int64_t &size,
                                    const float &min,
                                    const float &max,
                                    std::shared_ptr<std::mt19937_64> engine) {
  LOG(WARNING) << "#### data: " << data; 
  std::uniform_real_distribution<T> dist(static_cast<T>(min),
                                         static_cast<T>(max));
  for (int64_t i = 0; i < size; ++i) {
    data[i] = dist(*engine);
  }
}

template <typename T, typename Context>
void UniformRandomRawKernel(const Context& dev_ctx,
                            const phi::IntArray& shape,
                            phi::DataType dtype,
                            float min,
                            float max,
                            int seed,
                            int diag_num,
                            int diag_step,
                            float diag_val,
                            phi::DenseTensor* out) {
  out->Resize(phi::make_ddim(shape.GetData()));
  VLOG(4) << out->dims();
  T *data = dev_ctx.template Alloc<T>(out);
  auto size = out->numel();

  // 1. CPU implement
  // phi::DenseTensor cpu_out(out->dtype());
  // cpu_out.Resize(out->dims());
  
  // auto meta = cpu_out.meta();
  // LOG(WARNING) << "#### meta.is_scalar: " << meta.is_scalar;
  // LOG(WARNING) << "#### meta.dims: " << meta.dims;
  // LOG(WARNING) << "#### meta.dtype: " << meta.dtype;
  // LOG(WARNING) << "#### meta.layout: " << meta.layout;
  // // LOG(WARNING) << "#### meta.lod: " << meta.lod;
  // LOG(WARNING) << "#### meta.offset: " << meta.offset;

  phi::DenseTensor cpu_out1;
  phi::DenseTensorMeta meta_in = {out->dtype(), out->dims()};
  cpu_out1.set_meta(meta_in);

  auto meta1 = cpu_out1.meta();
  LOG(WARNING) << "#### meta.is_scalar: " << meta1.is_scalar;
  LOG(WARNING) << "#### meta.dims: " << meta1.dims;
  LOG(WARNING) << "#### meta.dtype: " << meta1.dtype;
  LOG(WARNING) << "#### meta.layout: " << meta1.layout;
  // LOG(WARNING) << "#### meta.lod: " << meta1.lod;
  LOG(WARNING) << "#### meta.offset: " << meta1.offset;

  T *cpu_data = dev_ctx.template HostAlloc<T>(&cpu_out1);

  std::shared_ptr<std::mt19937_64> engine;
  if (seed) {
    engine = std::make_shared<std::mt19937_64>();
    engine->seed(seed);
  } else {
    engine = dev_ctx.GetGenerator()->GetCPUEngine();
  }
  custom_kernel::UniformRealDistribution<T>(cpu_data, size, min, max, engine);
  if (diag_num > 0) {
    PADDLE_ENFORCE_GT(
        size,
        (diag_num - 1) * (diag_step + 1),
        phi::errors::InvalidArgument(
            "ShapeInvalid: the diagonal's elements is equal (num-1) "
            "* (step-1) with num %d, step %d,"
            "It should be smaller than %d, but received %d",
            diag_num,
            diag_step,
            (diag_num - 1) * (diag_step + 1),
            size));
    for (int64_t i = 0; i < diag_num; ++i) {
      int64_t pos = i * diag_step + i;
      cpu_data[pos] = diag_val;
    }
  }

  // 2. CPU Copy to NPU
  TensorCopy(dev_ctx, cpu_out1, true, out);
}

template <typename T, typename Context>
void UniformRandomKernel(const Context& dev_ctx,
                         const phi::IntArray& shape,
                         phi::DataType dtype,
                         float min,
                         float max,
                         int seed,
                         phi::DenseTensor* out) {
  custom_kernel::UniformRandomRawKernel<T>(
      dev_ctx, shape, dtype, min, max, seed, 0, 0, 0.0f, out);
}

}  // namespace custom_kernel 

PD_REGISTER_PLUGIN_KERNEL(
   uniform_random_raw, Ascend910, ALL_LAYOUT, custom_kernel::UniformRandomRawKernel, float) {}

PD_REGISTER_PLUGIN_KERNEL(
   uniform_random, Ascend910, ALL_LAYOUT, custom_kernel::UniformRandomKernel, float) {}

