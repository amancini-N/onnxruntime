// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "graph_flatbuffers_utils.h"

#include "core/common/flatbuffers.h"

#include "core/common/narrow.h"
#include "core/flatbuffers/flatbuffers_utils.h"
#include "core/flatbuffers/schema/ort.fbs.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/tensor_external_data_info.h"
#include "core/graph/graph.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime::fbs::utils {

template <typename DimsFieldType>
inline flatbuffers::Offset<flatbuffers::Vector<int64_t>>
SaveDims(flatbuffers::FlatBufferBuilder& builder, const DimsFieldType& dims) {
  std::vector<int64_t> dims_data(dims.size());
  std::copy(dims.begin(), dims.end(), dims_data.begin());
  return builder.CreateVector(dims_data);
}

#if !defined(ORT_MINIMAL_BUILD)

Status SaveInitializerOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                                const TensorProto& initializer,
                                const std::filesystem::path& model_path,
                                flatbuffers::Offset<fbs::Tensor>& fbs_tensor,
                                const ExternalDataWriter& external_writer) {
  auto name = SaveStringToOrtFormat(builder, initializer.has_name(), initializer.name());
  auto doc_string = SaveStringToOrtFormat(builder, initializer.has_doc_string(), initializer.doc_string());
  auto dims = SaveDims(builder, initializer.dims());

  // we have to populate string_data or raw_data prior to creating the TensorBuilder instance to avoid vtable offset
  // issues.
  flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>> string_data;
  flatbuffers::Offset<flatbuffers::Vector<uint8_t>> raw_data;
  int64_t external_data_offset = -1;

  auto src_type = initializer.data_type();
  const bool has_string_data = src_type == ONNX_NAMESPACE::TensorProto_DataType_STRING;

  if (has_string_data) {
    std::vector<std::string> string_data_vec(initializer.string_data().size());
    std::copy(initializer.string_data().cbegin(), initializer.string_data().cend(), string_data_vec.begin());
    string_data = builder.CreateVectorOfStrings(string_data_vec);
  } else {
    std::vector<uint8_t> unpacked_tensor;
    // We can not convert this in place, because the session may be used
    // after the model was saved in ort format. If the session is continued to be used, then
    // we continue with initializers in memory with wrong endianess
    if constexpr (endian::native != endian::little) {
      auto be_copy{initializer};
      onnxruntime::utils::ConvertRawDataInTensorProto(be_copy);
      ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(be_copy, model_path, unpacked_tensor));
    } else {
      ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(initializer, model_path, unpacked_tensor));
    }

    if (external_writer && unpacked_tensor.size() >= kMinimumSizeForExternalData) {
      // write bytes to external buffer/file and record offset for the start of the data
      uint64_t offset = 0;
      ORT_RETURN_IF_ERROR(external_writer(src_type, unpacked_tensor, offset));
      external_data_offset = onnxruntime::narrow<int64_t>(offset);  // offset in fb is int64_t so -1 can mark not in use
    } else {
      raw_data = builder.CreateVector(unpacked_tensor.data(), unpacked_tensor.size());
    }
  }

  fbs::TensorBuilder tb(builder);
  tb.add_name(name);
  tb.add_doc_string(doc_string);
  tb.add_dims(dims);
  tb.add_data_type(static_cast<fbs::TensorDataType>(src_type));

  if (has_string_data) {
    tb.add_string_data(string_data);
  } else {
    if (external_data_offset >= 0) {
      tb.add_external_data_offset(external_data_offset);
    } else {
      tb.add_raw_data(raw_data);
    }
  }
  fbs_tensor = tb.Finish();

  return Status::OK();
}

#if !defined(DISABLE_SPARSE_TENSORS)
Status SaveSparseInitializerOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                                      const ONNX_NAMESPACE::SparseTensorProto& initializer,
                                      const std::filesystem::path& model_path,
                                      flatbuffers::Offset<fbs::SparseTensor>& fbs_sparse_tensor) {
  // values
  const auto& values = initializer.values();
  flatbuffers::Offset<fbs::Tensor> values_off;
  ORT_RETURN_IF_ERROR(SaveInitializerOrtFormat(builder, values, model_path, values_off));

  // Indicies
  const auto& indicies = initializer.indices();
  flatbuffers::Offset<fbs::Tensor> indicies_off;
  ORT_RETURN_IF_ERROR(SaveInitializerOrtFormat(builder, indicies, model_path, indicies_off));

  // Shape
  auto shape = SaveDims(builder, initializer.dims());

  fbs::SparseTensorBuilder stb(builder);
  stb.add_values(values_off);
  stb.add_indices(indicies_off);
  stb.add_dims(shape);

  fbs_sparse_tensor = stb.Finish();

  return Status::OK();
}
#endif  // !defined(DISABLE_SPARSE_TENSORS)

#define GET_FBS_ATTR(BUILDER, TYPE, DATA_NAME, DATA) \
  fbs::AttributeBuilder attr_builder(BUILDER);       \
  attr_builder.add_name(name);                       \
  attr_builder.add_doc_string(doc_string);           \
  attr_builder.add_type(TYPE);                       \
  attr_builder.add_##DATA_NAME(DATA);                \
  fbs_attr = attr_builder.Finish();

#define GET_DATA_VEC(TYPE, NAME, SRC_DATA) \
  std::vector<TYPE> NAME(SRC_DATA.size()); \
  std::copy(SRC_DATA.cbegin(), SRC_DATA.cend(), NAME.begin());

Status SaveAttributeOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                              const AttributeProto& attr_proto,
                              flatbuffers::Offset<fbs::Attribute>& fbs_attr,
                              const std::filesystem::path& model_path,
                              const onnxruntime::Graph* subgraph) {
  auto name = SaveStringToOrtFormat(builder, attr_proto.has_name(), attr_proto.name());
  auto doc_string = SaveStringToOrtFormat(builder, attr_proto.has_doc_string(), attr_proto.doc_string());
  auto type = static_cast<fbs::AttributeType>(attr_proto.type());
  switch (type) {
    case fbs::AttributeType::FLOAT: {
      GET_FBS_ATTR(builder, type, f, attr_proto.f());
    } break;
    case fbs::AttributeType::INT: {
      GET_FBS_ATTR(builder, type, i, attr_proto.i());
    } break;
    case fbs::AttributeType::STRING: {
      auto s = builder.CreateString(attr_proto.s());
      GET_FBS_ATTR(builder, type, s, s);
    } break;
    case fbs::AttributeType::TENSOR: {
      flatbuffers::Offset<fbs::Tensor> fbs_tensor;
      ORT_RETURN_IF_ERROR(
          SaveInitializerOrtFormat(builder, attr_proto.t(), model_path, fbs_tensor));
      GET_FBS_ATTR(builder, type, t, fbs_tensor);
    } break;
    case fbs::AttributeType::GRAPH: {
      ORT_RETURN_IF(nullptr == subgraph, "Graph attribute value was null. Invalid ORT format model.");
      flatbuffers::Offset<fbs::Graph> fbs_graph;
      ORT_RETURN_IF_ERROR(subgraph->SaveToOrtFormat(builder, fbs_graph));
      GET_FBS_ATTR(builder, type, g, fbs_graph);
    } break;
    case fbs::AttributeType::FLOATS: {
      GET_DATA_VEC(float, floats_vec_, attr_proto.floats());
      auto floats = builder.CreateVector(floats_vec_);
      GET_FBS_ATTR(builder, type, floats, floats);
    } break;
    case fbs::AttributeType::INTS: {
      GET_DATA_VEC(int64_t, ints_vec_, attr_proto.ints());
      auto ints = builder.CreateVector(ints_vec_);
      GET_FBS_ATTR(builder, type, ints, ints);
    } break;
    case fbs::AttributeType::STRINGS: {
      GET_DATA_VEC(std::string, strings_vec_, attr_proto.strings());
      auto strings = builder.CreateVectorOfStrings(strings_vec_);
      GET_FBS_ATTR(builder, type, strings, strings);
    } break;
    case fbs::AttributeType::TENSORS: {
      std::vector<flatbuffers::Offset<fbs::Tensor>> fbs_tensors_vec;
      fbs_tensors_vec.reserve(attr_proto.tensors().size());
      for (const auto& tensor : attr_proto.tensors()) {
        flatbuffers::Offset<fbs::Tensor> fbs_tensor;
        ORT_RETURN_IF_ERROR(
            SaveInitializerOrtFormat(builder, tensor, model_path, fbs_tensor));
        fbs_tensors_vec.push_back(fbs_tensor);
      }
      auto tensors = builder.CreateVector(fbs_tensors_vec);
      GET_FBS_ATTR(builder, type, tensors, tensors);
    } break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "SaveAttributeOrtFormat: Unsupported attribute type: ", fbs::EnumNameAttributeType(type));
      break;
  }

  return Status::OK();
}

#undef GET_FBS_ATTR
#undef GET_DATA_VEC

#endif

/**
 * @brief Calculates how much memory will be required for putting contents of the given tensor into a plain array.
 *
 * complex64/complex128 tensors are not supported. The size is calculated from the dimensions and the data type,
 * to accommodate fbs::Tensors with external data.
 *
 * @param tensor flatbuffer representation of a tensor.
 * @return size_t size in bytes of the tensor's data.
 */
size_t GetSizeInBytesFromFbsTensor(const fbs::Tensor& tensor) {
  auto fbs_dims = tensor.dims();

  auto num_elements = std::accumulate(fbs_dims->cbegin(), fbs_dims->cend(), SafeInt<size_t>(1),
                                      std::multiplies<>());

  size_t byte_size_of_one_element;

  switch (tensor.data_type()) {
    case fbs::TensorDataType::FLOAT:
      byte_size_of_one_element = sizeof(float);
      break;
    case fbs::TensorDataType::UINT8:
      byte_size_of_one_element = sizeof(uint8_t);
      break;
    case fbs::TensorDataType::INT8:
      byte_size_of_one_element = sizeof(int8_t);
      break;
    case fbs::TensorDataType::UINT16:
      byte_size_of_one_element = sizeof(uint16_t);
      break;
    case fbs::TensorDataType::INT16:
      byte_size_of_one_element = sizeof(int16_t);
      break;
    case fbs::TensorDataType::INT32:
      byte_size_of_one_element = sizeof(int32_t);
      break;
    case fbs::TensorDataType::INT64:
      byte_size_of_one_element = sizeof(int64_t);
      break;
    case fbs::TensorDataType::BOOL:
      byte_size_of_one_element = sizeof(bool);
      break;
    case fbs::TensorDataType::FLOAT16:
      byte_size_of_one_element = sizeof(MLFloat16);
      break;
    case fbs::TensorDataType::DOUBLE:
      byte_size_of_one_element = sizeof(double);
      break;
    case fbs::TensorDataType::UINT32:
      byte_size_of_one_element = sizeof(uint32_t);
      break;
    case fbs::TensorDataType::UINT64:
      byte_size_of_one_element = sizeof(uint64_t);
      break;
    case fbs::TensorDataType::BFLOAT16:
      byte_size_of_one_element = sizeof(BFloat16);
      break;
#if !defined(DISABLE_FLOAT8_TYPES)
    case fbs::TensorDataType::FLOAT8E4M3FN:
      byte_size_of_one_element = sizeof(uint8_t);
      break;
    case fbs::TensorDataType::FLOAT8E4M3FNUZ:
      byte_size_of_one_element = sizeof(uint8_t);
      break;
    case fbs::TensorDataType::FLOAT8E5M2:
      byte_size_of_one_element = sizeof(uint8_t);
      break;
    case fbs::TensorDataType::FLOAT8E5M2FNUZ:
      byte_size_of_one_element = sizeof(uint8_t);
      break;
#endif
    case fbs::TensorDataType::STRING:
      ORT_THROW("String data type is not supported for on-device training", tensor.name());
    default:
      ORT_THROW("Unsupported tensor data type for tensor ", tensor.name());
  }
  return num_elements * byte_size_of_one_element;
}

Status LoadInitializerOrtFormat(const fbs::Tensor& fbs_tensor, TensorProto& initializer,
                                const OrtFormatLoadOptions& load_options,
                                const ExternalDataReader& external_data_reader) {
  initializer.Clear();

  LOAD_STR_FROM_ORT_FORMAT(initializer, name, fbs_tensor.name());
  LOAD_STR_FROM_ORT_FORMAT(initializer, doc_string, fbs_tensor.doc_string());

  auto fbs_dims = fbs_tensor.dims();
  ORT_RETURN_IF(nullptr == fbs_dims, "Missing dimensions for initializer. Invalid ORT format model.");
  initializer.mutable_dims()->Add(fbs_dims->cbegin(), fbs_dims->cend());
  auto fbs_data_type = fbs_tensor.data_type();
  initializer.set_data_type(static_cast<int32_t>(fbs_data_type));

  if (fbs_data_type == fbs::TensorDataType::STRING) {
    auto fbs_str_data = fbs_tensor.string_data();
    ORT_RETURN_IF(nullptr == fbs_str_data, "Missing string data for initializer. Invalid ORT format model.");
    auto mutable_str_data = initializer.mutable_string_data();
    mutable_str_data->Reserve(fbs_str_data->size());
    for (const auto* fbs_str : *fbs_str_data) {
      mutable_str_data->Add(fbs_str->str());
    }
  } else {
    const auto* fbs_raw_data = fbs_tensor.raw_data();
    if (fbs_raw_data) {
      if (load_options.can_use_flatbuffer_for_initializers && fbs_raw_data->size() > 127) {
        static_assert(sizeof(void*) <= sizeof(ExternalDataInfo::OFFSET_TYPE));
        const void* data_offset = fbs_raw_data->Data();
        // we reinterpret_cast this back to void* in tensorprotoutils.cc:GetExtDataFromTensorProto.
        // use intptr_t as OFFSET_TYPE is signed. in theory you could get a weird looking value if the address uses the
        // high bit, but that should be unlikely in a scenario where we care about memory usage enough to use this path.
        auto offset = narrow<ExternalDataInfo::OFFSET_TYPE>(reinterpret_cast<intptr_t>(data_offset));

        ExternalDataInfo::SetExternalLocationToProto(onnxruntime::utils::kTensorProtoMemoryAddressTag,
                                                     offset, fbs_raw_data->size(), initializer);

      } else {
        // fbs_raw_data is uint8_t vector, so the size is byte size
        initializer.set_raw_data(fbs_raw_data->Data(), fbs_raw_data->size());
      }
    } else {
      auto external_data_offset = fbs_tensor.external_data_offset();

      // no external data. should have had raw data.
      ORT_RETURN_IF(external_data_offset < 0, "Missing raw data for initializer. Invalid ORT format model.");

      // external data but no reader
      ORT_RETURN_IF(!external_data_reader, "Tensor has external data but a data reader was not provided.");

      // FUTURE: This could be setup similarly to can_use_flatbuffer_for_initializers above if the external data file
      // is memory mapped and guaranteed to remain valid. This would avoid the copy.
      auto num_bytes = GetSizeInBytesFromFbsTensor(fbs_tensor);

      // pre-allocate so we can write directly to the string buffer
      std::string& raw_data = *initializer.mutable_raw_data();
      raw_data.resize(num_bytes);
      auto output_buffer = gsl::make_span<uint8_t>(reinterpret_cast<uint8_t*>(raw_data.data()), num_bytes);

      ORT_RETURN_IF_ERROR(external_data_reader(external_data_offset, output_buffer));
    }
  }

  return Status::OK();
}

#if !defined(DISABLE_SPARSE_TENSORS)
Status LoadSparseInitializerOrtFormat(const fbs::SparseTensor& fbs_sparse_tensor,
                                      SparseTensorProto& initializer,
                                      const OrtFormatLoadOptions& load_options) {
  SparseTensorProto loaded_initializer;
  auto fbs_values_tensor = fbs_sparse_tensor.values();
  ORT_RETURN_IF(nullptr == fbs_values_tensor, "Missing values for sparse initializer. Invalid ORT format model.");
  auto* values_tensor = loaded_initializer.mutable_values();
  ORT_RETURN_IF_ERROR(LoadInitializerOrtFormat(*fbs_values_tensor, *values_tensor, load_options));
  ORT_RETURN_IF(values_tensor->name().empty(), "Missing name for SparseTensor initializer. Invalid ORT format model.");

  auto fbs_indicies_tensor = fbs_sparse_tensor.indices();
  ORT_RETURN_IF(nullptr == fbs_indicies_tensor, "Missing indicies for sparse initializer: ", "'", values_tensor->name(), "'",
                "Invalid ORT format model.");
  auto* indicies_tensor = loaded_initializer.mutable_indices();
  ORT_RETURN_IF_ERROR(LoadInitializerOrtFormat(*fbs_indicies_tensor, *indicies_tensor, load_options));

  auto fbs_dims = fbs_sparse_tensor.dims();
  ORT_RETURN_IF(nullptr == fbs_dims, "Missing dims for sparse initializer: ", "'", values_tensor->name(), "'",
                "Invalid ORT format model.");
  loaded_initializer.mutable_dims()->Add(fbs_dims->cbegin(), fbs_dims->cend());

  swap(loaded_initializer, initializer);
  return Status::OK();
}
#endif  // !defined(DISABLE_SPARSE_TENSORS)

Status LoadAttributeOrtFormat(const fbs::Attribute& fbs_attr,
                              ONNX_NAMESPACE::AttributeProto& attr_proto,
                              std::unique_ptr<onnxruntime::Graph>& sub_graph,
                              onnxruntime::Graph& graph, onnxruntime::Node& node,
                              const OrtFormatLoadOptions& load_options,
                              const logging::Logger& logger) {
  attr_proto.Clear();
  LOAD_STR_FROM_ORT_FORMAT(attr_proto, name, fbs_attr.name());
  LOAD_STR_FROM_ORT_FORMAT(attr_proto, doc_string, fbs_attr.doc_string());

  auto type = static_cast<AttributeProto_AttributeType>(fbs_attr.type());
  attr_proto.set_type(type);
  switch (type) {
    case AttributeProto_AttributeType_FLOAT: {
      attr_proto.set_f(fbs_attr.f());
    } break;
    case AttributeProto_AttributeType_INT: {
      attr_proto.set_i(fbs_attr.i());
    } break;
    case AttributeProto_AttributeType_STRING: {
      auto fbs_str = fbs_attr.s();
      ORT_RETURN_IF(nullptr == fbs_str, "Null string attribute. Invalid ORT format model.");
      attr_proto.set_s(fbs_str->str());
    } break;
    case AttributeProto_AttributeType_TENSOR: {
      auto fbs_tensor = fbs_attr.t();
      ORT_RETURN_IF(nullptr == fbs_tensor, "Null tensor attribute. Invalid ORT format model.");
      ORT_RETURN_IF_ERROR(LoadInitializerOrtFormat(*fbs_tensor, *attr_proto.mutable_t(),
                                                   load_options));
    } break;
    case AttributeProto_AttributeType_GRAPH: {
      // If the attribute type is a graph, we will create an empty graph in attr_proto so that the ONNX checker
      // is happy in a full build, and deserialize the ORT Graph instance into the 'graph' param.
      auto fbs_graph = fbs_attr.g();
      ORT_RETURN_IF(nullptr == fbs_graph, "Null graph attribute. Invalid ORT format model.");
      attr_proto.mutable_g()->set_name("Empty graph proto from deserialization of ORT format model");
      ORT_RETURN_IF_ERROR(onnxruntime::Graph::LoadFromOrtFormat(*fbs_graph, graph, node,
                                                                load_options,
                                                                logger, sub_graph));
    } break;
    case AttributeProto_AttributeType_FLOATS: {
      auto fbs_floats = fbs_attr.floats();
      ORT_RETURN_IF(nullptr == fbs_floats, "Null floats attribute. Invalid ORT format model.");
      auto floats = attr_proto.mutable_floats();
      floats->Reserve(fbs_floats->size());
      floats->Add(fbs_floats->cbegin(), fbs_floats->cend());
    } break;
    case AttributeProto_AttributeType_INTS: {
      auto fbs_ints = fbs_attr.ints();
      ORT_RETURN_IF(nullptr == fbs_ints, "Null ints attribute. Invalid ORT format model.");
      auto* ints = attr_proto.mutable_ints();
      ints->Reserve(fbs_ints->size());
      ints->Add(fbs_ints->cbegin(), fbs_ints->cend());
    } break;
    case AttributeProto_AttributeType_STRINGS: {
      auto fbs_strings = fbs_attr.strings();
      ORT_RETURN_IF(nullptr == fbs_strings, "Null strings attribute. Invalid ORT format model.");
      auto* strings = attr_proto.mutable_strings();
      strings->Reserve(fbs_strings->size());
      for (const auto* fbs_str : *fbs_strings) {
        ORT_RETURN_IF(nullptr == fbs_str, "Null string in strings attribute. Invalid ORT format model.");
        strings->Add(fbs_str->str());
      }
    } break;
    case AttributeProto_AttributeType_TENSORS: {
      auto fbs_tensors = fbs_attr.tensors();
      ORT_RETURN_IF(nullptr == fbs_tensors, "Null tensors attribute. Invalid ORT format model.");
      auto* tensors = attr_proto.mutable_tensors();
      tensors->Reserve(fbs_tensors->size());
      for (const auto* fbs_tensor : *fbs_tensors) {
        ORT_RETURN_IF(nullptr == fbs_tensor, "Null tensor in tensors attribute. Invalid ORT format model.");
        ORT_RETURN_IF_ERROR(LoadInitializerOrtFormat(*fbs_tensor, *tensors->Add(),
                                                     load_options));
      }
    } break;

    default:
      break;
  }

  return Status::OK();
}

#ifdef ENABLE_TRAINING_APIS

Status SaveOrtTensorOrtFormat(
    const std::string& tensor_name, const onnxruntime::Tensor& ort_tensor,
    flatbuffers::FlatBufferBuilder& builder,
    flatbuffers::Offset<fbs::Tensor>& fbs_tensor,
    ExternalDataWriter external_data_writer) {
  ORT_RETURN_IF(ort_tensor.IsDataTypeString(),
                "TensorProto_DataType_STRING is not supported while saving a tensor to ORT format.");

  const auto fbs_tensor_name = builder.CreateString(tensor_name);
  const auto fbs_tensor_dims = SaveDims(builder, ort_tensor.Shape().GetDims());
  // To avoid issues with vtable offsets, raw_data fbs::vector must be constructed before the TensorBuilder begins
  // building the tensor. See flatbuffer_builder.h's NotNested() function for more details.
  flatbuffers::Offset<flatbuffers::Vector<uint8_t>> raw_data;
  if (!external_data_writer) {
    raw_data = builder.CreateVector(static_cast<const uint8_t*>(ort_tensor.DataRaw()),
                                    ort_tensor.SizeInBytes());
  }

  fbs::TensorBuilder tb(builder);
  tb.add_name(fbs_tensor_name);
  tb.add_doc_string(0);
  tb.add_dims(fbs_tensor_dims);
  tb.add_data_type(static_cast<fbs::TensorDataType>(ort_tensor.GetElementType()));
  if (external_data_writer) {
    uint64_t offset = 0;
    gsl::span<const uint8_t> ort_tensor_data_span(static_cast<const uint8_t*>(ort_tensor.DataRaw()), ort_tensor.SizeInBytes());
    ORT_RETURN_IF_ERROR(external_data_writer(ort_tensor.GetElementType(), ort_tensor_data_span, offset));
    int64_t external_data_offset = onnxruntime::narrow<int64_t>(offset);
    tb.add_external_data_offset(external_data_offset);
  } else {
    tb.add_raw_data(raw_data);
  }
  fbs_tensor = tb.Finish();
  return Status::OK();
}

template <typename T>
struct UnpackTensorWithType {
  Status operator()(const ONNX_NAMESPACE::TensorProto& tensor_proto, const fbs::Tensor& fbs_tensor,
                    onnxruntime::Tensor& ort_tensor, const ExternalDataReader& external_data_reader) const {
    if (fbs_tensor.external_data_offset() >= 0) {
      auto fbs_tensor_external_data_offset = fbs_tensor.external_data_offset();
      ORT_RETURN_IF_NOT(external_data_reader, "Tensor has external data but a data reader was not provided.");

      // no external data. should have had raw data.
      ORT_RETURN_IF(fbs_tensor_external_data_offset < 0, "Missing raw data for initializer. Invalid ORT format model.");

      const size_t raw_data_len = fbs::utils::GetSizeInBytesFromFbsTensor(fbs_tensor);

      auto raw_buf = std::make_unique<uint8_t[]>(raw_data_len);
      gsl::span<uint8_t> raw_buf_span(raw_buf.get(), raw_data_len);

      ORT_RETURN_IF_ERROR(external_data_reader(fbs_tensor_external_data_offset, raw_buf_span));
      return onnxruntime::utils::UnpackTensor(
          tensor_proto, raw_buf_span.data(),
          raw_buf_span.size(),
          ort_tensor.MutableData<T>(),
          static_cast<size_t>(ort_tensor.Shape().Size()));
    } else if (fbs_tensor.raw_data()) {
      return onnxruntime::utils::UnpackTensor(
          tensor_proto, fbs_tensor.raw_data()->Data(),
          fbs_tensor.raw_data()->size(),
          ort_tensor.MutableData<T>(),
          static_cast<size_t>(ort_tensor.Shape().Size()));
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid tensor. Expected: raw data or external data offset. Actual: ",
                             fbs_tensor.string_data() ? "string data" : "nullptr", " for tensor named: ",
                             fbs_tensor.name()->str());
    }
  }
};

Status LoadOrtTensorOrtFormat(const fbs::Tensor& fbs_tensor, const AllocatorPtr allocator,
                              std::string& tensor_name, onnxruntime::Tensor& ort_tensor,
                              const ExternalDataReader& external_data_reader) {
  auto* fbs_tensor_name = fbs_tensor.name();
  ORT_RETURN_IF_NOT(fbs_tensor_name, "Flatbuffer tensor is invalid. Expected: A valid tensor name. Actual: nullptr.");
  tensor_name = fbs_tensor_name->str();

  auto* tensor_dims = fbs_tensor.dims();
  ORT_RETURN_IF_NOT(tensor_dims, "Flatbuffer tensor is invalid. Expected: Valid tensor dims. Actual: nullptr.");

  const auto tensor_data_type = static_cast<int32_t>(fbs_tensor.data_type());
  const DataTypeImpl* tensor_dtype = DataTypeImpl::TensorTypeFromONNXEnum(
                                         tensor_data_type)
                                         ->GetElementType();
  ort_tensor = onnxruntime::Tensor(
      tensor_dtype, TensorShape(tensor_dims->data(), tensor_dims->size()), allocator);

  if (fbs_tensor.raw_data() && fbs_tensor.raw_data()->size() == 0U) {
    // Empty tensor. Nothing to unpack.
    // This check is necessary because an empty ort tensor will return a size of 1.
    // As a result, the following call to UnpackTensor will fail since the src and
    // dst sizes do not match (0 and 1 elements).
    return Status::OK();
  }

  // The tensor proto is used as a dummy here. The actual data is stored in the raw_data field of the flatbuffer.
  // The data is copied from the raw_data field to the ort_tensor.
  ONNX_NAMESPACE::TensorProto unused_tensor_proto;
  unused_tensor_proto.set_data_type(tensor_data_type);

  onnxruntime::utils::MLTypeCallDispatcher<float, bool, double, int8_t, uint8_t, int16_t, uint16_t,
                                           int32_t, uint32_t, int64_t, uint64_t>
      dispatcher(tensor_data_type);
  return dispatcher.InvokeRet<Status, UnpackTensorWithType>(unused_tensor_proto, fbs_tensor, ort_tensor, external_data_reader);
}

#endif  // ENABLE_TRAINING_APIS

}  // namespace onnxruntime::fbs::utils
