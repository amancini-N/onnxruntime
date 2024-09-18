// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/transformers/beam_search_helper.h"
#include "onnx/defs/tensor_proto_util.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {

template <typename T>
Status Get2DAttrsOrDefault(const OpKernelInfo& kernel_info, const std::string& attr_name,
                           ONNX_NAMESPACE::TensorProto_DataType proto_type,
                           std::vector<int>& shape, std::vector<T>& data) {
  ONNX_NAMESPACE::TensorProto proto;

  auto status = kernel_info.GetAttr(attr_name, &proto);
  if (!status.IsOK()) {
    // Attribute is missing, use empty span
    data = std::vector<T>();
  } else {
    auto n_dims = proto.dims_size();
    auto n_elems = 1;
    ORT_ENFORCE(n_dims == 2, "Attribute ", attr_name, " must be 2D tensor.");
    ORT_ENFORCE(proto.data_type() == proto_type, "Attribute ", attr_name, " must be of type ", proto_type);
    shape.resize(n_dims);
    for (int i = 0; i < n_dims; ++i) {
      shape[i] = static_cast<int>(proto.dims(i));
      n_elems *= shape[i];
    }

    data = ONNX_NAMESPACE::ParseData<T>(&proto);

  }
  return Status::OK();
}

Status Get2DAttrsOrDefault(const OpKernelInfo& info, const std::string& name, std::vector<int>& shape, std::vector<int>& data) {
  return Get2DAttrsOrDefault(info, name, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32, shape, data);
}

} // namespace transformers
} // namespace contrib
} // namespace onnxruntime
