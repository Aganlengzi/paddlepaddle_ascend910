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

#include "npu_op_runner.h"

void UpdateAttr(const phi::DDim& in_dims, const std::vector<int> axes,
                const std::vector<int> starts, const std::vector<int> ends,
                std::vector<int>* offsets, std::vector<int>* size) {
  int cnt = 0;
  for (int i = 0; i < in_dims.size(); ++i) {
    int start = 0;
    int end = in_dims[i];
    // NOTE(zhiqiu): Becareful that cnt may > axes.size() and result in
    // overflow.
    int axis = cnt < static_cast<int>(axes.size()) ? axes[cnt] : -1;
    if (axis == i) {
      start = starts[cnt];
      if (start < 0) {
        start = (start + in_dims[i]);
      }
      start = std::max(start, static_cast<int>(0));
      end = ends[cnt];
      if (end < 0) {
        end = (end + in_dims[i]);
      }
      end = std::min(end, static_cast<int>(in_dims[i]));
      cnt++;
    }

    (*offsets)[i] = start;
    (*size)[i] = end - start;
  }
}

namespace custom_kernel {

template <typename T, typename Context>
void SliceKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const std::vector<int64_t>& axes_in,
                 const phi::ScalarArray& starts_array,
                 const phi::ScalarArray& ends_array,
                 const std::vector<int64_t>& infer_flags,
                 const std::vector<int64_t>& decrease_axis,
                 phi::DenseTensor* out) {
  auto starts_int = starts_array.GetData();
  auto ends_int = ends_array.GetData();
  std::vector<int> axes(axes_in.begin(), axes_in.end());
  std::vector<int> starts(starts_int.begin(), starts_int.end());
  std::vector<int> ends(ends_int.begin(), ends_int.end());

  // auto decrease_axis = ctx.Attr<std::vector<int>>("decrease_axis");
  // auto infer_flags = ctx.Attr<std::vector<int>>("infer_flags");

  const auto& in_dims = x.dims();

  // Get the accurate attribute value of starts and ends
  // auto starts_tensor_list = ctx.MultiInput<Tensor>("StartsTensorList");
  // if (ctx.HasInput("StartsTensor")) {
  //   starts = GetDataFromTensor<int>(ctx.Input<Tensor>("StartsTensor"));
  // } else if (starts_tensor_list.size() > 0) {
  //   starts = GetDataFromTensorList<int>(starts_tensor_list);
  // }

  // auto ends_tensor_list = ctx.MultiInput<Tensor>("EndsTensorList");
  // if (ctx.HasInput("EndsTensor")) {
  //   ends = GetDataFromTensor<int>(ctx.Input<Tensor>("EndsTensor"));
  // } else if (ends_tensor_list.size() > 0) {
  //   ends = GetDataFromTensorList<int>(ends_tensor_list);
  // }

  // PADDLE_ENFORCE_EQ(
  //     starts.size(), axes.size(),
  //     platform::errors::InvalidArgument(
  //         "The size of starts must be equal to the size of axes."));
  // PADDLE_ENFORCE_EQ(
  //     ends.size(), axes.size(),
  //     platform::errors::InvalidArgument(
  //         "The size of ends must be equal to the size of axes."));

  // if (ctx.HasInput("StartsTensor") || ctx.HasInput("EndsTensor") ||
  //     starts_tensor_list.size() > 0 || ends_tensor_list.size() > 0) {
  //   // Infer output dims
  //   auto out_dims = out->dims();
  //   auto slice_dims = out_dims;
  //   for (size_t i = 0; i < axes.size(); ++i) {
  //     // when start == -1 && end == start+1
  //     if (starts[i] == -1 && ends[i] == 0 && infer_flags[i] == -1) {
  //       auto ret =
  //           std::find(decrease_axis.begin(), decrease_axis.end(), axes[i]);
  //       if (ret != decrease_axis.end()) {
  //         ends[i] = in_dims[axes[i]];
  //       }
  //     }
  //   }

  //   CheckAndUpdateSliceAttrs(in_dims, axes, &starts, &ends);
  //   slice_dims =
  //       GetSliceDims<int>(in_dims, axes, starts, ends, nullptr, nullptr);
  //   out_dims = GetDecreasedDims(slice_dims, decrease_axis);

  //   out->Resize(out_dims);
  // }

  dev_ctx.template Alloc<T>(out);

  std::vector<int> offsets(in_dims.size());
  std::vector<int> size(in_dims.size());

  UpdateAttr(in_dims, axes, starts, ends, &offsets, &size);

  auto stream = static_cast<aclrtStream>(dev_ctx.stream());
  const auto& runner = NpuOpRunner("SliceD", {x}, {*out},
                                   {{"offsets", offsets}, {"size", size}});
  runner.Run(stream);
}

template <typename T, typename Context>
void SliceGradKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& out_grad,
                     const std::vector<int64_t>& axes_in,
                     const phi::ScalarArray& starts_array,
                     const phi::ScalarArray& ends_array,
                     const std::vector<int64_t>& infer_flags,
                     const std::vector<int64_t>& decrease_axis,
                     phi::DenseTensor* x_grad) {
  auto starts_int = starts_array.GetData();
  auto ends_int = ends_array.GetData();

  std::vector<int> axes(axes_in.begin(), axes_in.end());
  std::vector<int> starts(starts_int.begin(), starts_int.end());
  std::vector<int> ends(ends_int.begin(), ends_int.end());

  // Get the accurate attribute value of starts and ends
  // auto starts_tensor_list = ctx.MultiInput<Tensor>("StartsTensorList");
  // if (ctx.HasInput("StartsTensor")) {
  //   starts = GetDataFromTensor<int>(ctx.Input<Tensor>("StartsTensor"));
  // } else if (starts_tensor_list.size() > 0) {
  //   starts = GetDataFromTensorList<int>(starts_tensor_list);
  // }

  // auto ends_tensor_list = ctx.MultiInput<Tensor>("EndsTensorList");
  // if (ctx.HasInput("EndsTensor")) {
  //   ends = GetDataFromTensor<int>(ctx.Input<Tensor>("EndsTensor"));
  // } else if (ends_tensor_list.size() > 0) {
  //   ends = GetDataFromTensorList<int>(ends_tensor_list);
  // }

  const auto& in_dims = x.dims();
  int rank = in_dims.size();

  std::vector<int> offsets(rank);
  std::vector<int> size(rank);
  UpdateAttr(in_dims, axes, starts, ends, &offsets, &size);

  std::vector<std::vector<int64_t>> paddings(rank, std::vector<int64_t>(2));
  for (int i = 0; i < rank; ++i) {
    paddings[i][0] = static_cast<int64_t>(offsets[i]);
    paddings[i][1] = static_cast<int64_t>(in_dims[i] - size[i] - offsets[i]);
  }

  phi::DenseTensor tmp_dout;
  tmp_dout.ShareDataWith(out_grad);
  auto out_dims = out_grad.dims();
  // auto decrease_axis = ctx.Attr<std::vector<int>>("decrease_axis");
  // auto decrease_size = decrease_axis.size();
  // if (decrease_size > 0) {
  //   if (decrease_size == static_cast<size_t>(in_dims.size())) {
  //     out_dims = phi::make_ddim(std::vector<int>(decrease_size, 1));
  //   } else {
  //     std::vector<int> origin_out_shape(out_dims.size() + decrease_size, -1);
  //     for (size_t i = 0; i < decrease_size; ++i) {
  //       origin_out_shape[decrease_axis[i]] = 1;
  //     }
  //     int index = 0;
  //     for (size_t i = 0; i < origin_out_shape.size(); ++i) {
  //       if (origin_out_shape[i] == -1) {
  //         origin_out_shape[i] = out_dims[index];
  //         ++index;
  //       }
  //     }
  //     out_dims = phi::make_ddim(origin_out_shape);
  //   }
  //   tmp_dout.Resize(out_dims);
  // }

  dev_ctx.template Alloc<T>(x_grad);
  auto stream = static_cast<aclrtStream>(dev_ctx.stream());
  const auto& runner =
      NpuOpRunner("PadD", {tmp_dout}, {*x_grad}, {{"paddings", paddings}});
  runner.Run(stream);
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(slice, Ascend910, ALL_LAYOUT,
                          custom_kernel::SliceKernel, phi::dtype::float16,
                          float, double, int16_t, int32_t, int64_t, bool) {}
PD_REGISTER_PLUGIN_KERNEL(slice_grad, Ascend910, ALL_LAYOUT,
                          custom_kernel::SliceGradKernel, phi::dtype::float16,
                          float, double, int16_t, int32_t, int64_t, bool) {}
