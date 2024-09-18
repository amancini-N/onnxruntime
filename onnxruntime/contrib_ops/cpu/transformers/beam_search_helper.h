// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {

Status Get2DAttrsOrDefault(
    const OpKernelInfo& info,
    const std::string& name,
    std::vector<int>& shape,
    std::vector<int>& data);

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
