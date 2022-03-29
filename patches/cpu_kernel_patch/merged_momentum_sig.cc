#include "paddle/phi/core/compat/op_utils.h"

namespace phi {
KernelSignature MergedMomentOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "merged_momentum", {"Param", "Grad", "Velocity", "LearningRate"},
      {"mu", "use_nesterov", "regularization_method", "regularization_coeff",
       "multi_precision", "rescale_grad"},
      {"ParamOut", "VelocityOut"});
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(merged_momentum, phi::MergedMomentOpArgumentMapping);

