// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <vector>
#include "gtest/gtest.h"
#include "core/common/gsl.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "test/common/cuda_op_test_utils.h"
#include "contrib_ops/cpu/transformers/logits_processor.h"
#include "contrib_ops/cpu/transformers/logits_processor.cc"


#ifdef USE_CUDA
#include "core/providers/cuda/cuda_provider_options.h"
#endif

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

TEST(BeamSearchTest, GptBeamSearchFp32) {
  std::vector<int64_t> input_ids_shape{3, 12};
  std::vector<int32_t> input_ids{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328};

  std::vector<int64_t> parameter_shape{1};
  std::vector<int32_t> max_length{20};
  std::vector<int32_t> min_length{1};
  std::vector<int32_t> num_beams{4};
  std::vector<int32_t> num_return_sequences{1};
  std::vector<float> length_penalty{1.0f};
  std::vector<float> repetition_penalty{1.0f};

  std::vector<int64_t> expected_output_shape{input_ids_shape[0], num_return_sequences[0], max_length[0]};
  std::vector<int32_t> expected_output{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620, 131, 131, 131, 181, 638, 638, 638, 638,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572, 292, 292, 292, 292, 292, 292, 292, 292,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328, 328, 669, 669, 669, 669, 669, 669, 669};

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  auto input_ids_tensor = Ort::Value::CreateTensor(
      info, input_ids.data(), input_ids.size(), input_ids_shape.data(), input_ids_shape.size());

  auto max_length_tensor = Ort::Value::CreateTensor(
      info, max_length.data(), max_length.size(), parameter_shape.data(), parameter_shape.size());

  auto min_length_tensor = Ort::Value::CreateTensor(
      info, min_length.data(), min_length.size(), parameter_shape.data(), parameter_shape.size());

  auto num_beams_tensor = Ort::Value::CreateTensor(
      info, num_beams.data(), num_beams.size(), parameter_shape.data(), parameter_shape.size());

  auto num_return_sequences_tensor = Ort::Value::CreateTensor(
      info, num_return_sequences.data(), num_return_sequences.size(), parameter_shape.data(), parameter_shape.size());

  auto length_penalty_tensor = Ort::Value::CreateTensor(
      info, length_penalty.data(), length_penalty.size(), parameter_shape.data(), parameter_shape.size());

  auto repetition_penalty_tensor = Ort::Value::CreateTensor(
      info, repetition_penalty.data(), repetition_penalty.size(), parameter_shape.data(), parameter_shape.size());

  std::vector<Ort::Value> ort_inputs;
  ort_inputs.push_back(std::move(input_ids_tensor));
  ort_inputs.push_back(std::move(max_length_tensor));
  ort_inputs.push_back(std::move(min_length_tensor));
  ort_inputs.push_back(std::move(num_beams_tensor));
  ort_inputs.push_back(std::move(num_return_sequences_tensor));
  ort_inputs.push_back(std::move(length_penalty_tensor));
  ort_inputs.push_back(std::move(repetition_penalty_tensor));
  const char* input_names[] = {"input_ids", "max_length", "min_length", "num_beams", "num_return_sequences",
                               "length_penalty", "repetition_penalty"};
  const char* const output_names[] = {"sequences"};

  Ort::SessionOptions session_options;
#ifdef USE_CUDA
  OrtCUDAProviderOptionsV2 cuda_options;
  cuda_options.use_tf32 = false;
  session_options.AppendExecutionProvider_CUDA_V2(cuda_options);
#endif

#ifdef USE_ROCM
  OrtROCMProviderOptions rocm_options;
  session_options.AppendExecutionProvider_ROCM(rocm_options);
#endif

  // The ONNX model is generated like the following:
  // python convert_generation.py --model_type gpt2 -m hf-internal-testing/tiny-random-gpt2
  //        --output tiny_gpt2_beamsearch_fp16.onnx --use_gpu --max_length 20
  // (with separate_gpt2_decoder_for_init_run set to False as it is now set to True by default)
  Ort::Session session(*ort_env, ORT_TSTR("testdata/transformers/tiny_gpt2_beamsearch.onnx"), session_options);
  auto ort_outputs = session.Run(Ort::RunOptions{}, input_names, ort_inputs.data(), ort_inputs.size(),
                                 output_names, 1);

  ASSERT_EQ(ort_outputs.size(), 1U);
  const auto& sequences = ort_outputs[0];
  ASSERT_TRUE(sequences.IsTensor());

  auto result_ts = sequences.GetTensorTypeAndShapeInfo();
  ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, result_ts.GetElementType());

  ASSERT_EQ(expected_output_shape, result_ts.GetShape());
  const auto* result_vals = sequences.GetTensorData<int32_t>();
  auto result_span = gsl::make_span(result_vals, expected_output.size());
  ASSERT_TRUE(std::equal(expected_output.cbegin(), expected_output.cend(), result_span.begin(), result_span.end()));
}

TEST(BeamSearchTest, GptBeamSearchFp16) {
  std::vector<int64_t> input_ids_shape{3, 12};
  std::vector<int32_t> input_ids{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328};

  std::vector<int64_t> parameter_shape{1};
  std::vector<int32_t> max_length{20};
  std::vector<int32_t> min_length{1};
  std::vector<int32_t> num_beams{4};
  std::vector<int32_t> num_return_sequences{1};
  std::vector<float> length_penalty{1.0f};
  std::vector<float> repetition_penalty{1.0f};

  std::vector<int64_t> expected_output_shape{input_ids_shape[0], num_return_sequences[0], max_length[0]};

  std::vector<int32_t> expected_output{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620, 131, 131, 131, 181, 638, 638, 638, 638,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572, 292, 292, 292, 292, 292, 292, 292, 292,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328, 328, 669, 669, 669, 669, 669, 669, 669};

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  auto input_ids_tensor = Ort::Value::CreateTensor(
      info, input_ids.data(), input_ids.size(), input_ids_shape.data(), input_ids_shape.size());

  auto max_length_tensor = Ort::Value::CreateTensor(
      info, max_length.data(), max_length.size(), parameter_shape.data(), parameter_shape.size());

  auto min_length_tensor = Ort::Value::CreateTensor(
      info, min_length.data(), min_length.size(), parameter_shape.data(), parameter_shape.size());

  auto num_beams_tensor = Ort::Value::CreateTensor(
      info, num_beams.data(), num_beams.size(), parameter_shape.data(), parameter_shape.size());

  auto num_return_sequences_tensor = Ort::Value::CreateTensor(
      info, num_return_sequences.data(), num_return_sequences.size(), parameter_shape.data(), parameter_shape.size());

  auto length_penalty_tensor = Ort::Value::CreateTensor(
      info, length_penalty.data(), length_penalty.size(), parameter_shape.data(), parameter_shape.size());

  auto repetition_penalty_tensor = Ort::Value::CreateTensor(
      info, repetition_penalty.data(), repetition_penalty.size(), parameter_shape.data(), parameter_shape.size());

  std::vector<Ort::Value> ort_inputs;
  ort_inputs.push_back(std::move(input_ids_tensor));
  ort_inputs.push_back(std::move(max_length_tensor));
  ort_inputs.push_back(std::move(min_length_tensor));
  ort_inputs.push_back(std::move(num_beams_tensor));
  ort_inputs.push_back(std::move(num_return_sequences_tensor));
  ort_inputs.push_back(std::move(length_penalty_tensor));
  ort_inputs.push_back(std::move(repetition_penalty_tensor));
  const char* input_names[] = {"input_ids", "max_length", "min_length", "num_beams", "num_return_sequences",
                               "length_penalty", "repetition_penalty"};
  const char* const output_names[] = {"sequences"};

  constexpr int min_cuda_architecture = 530;
  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture);
  bool enable_rocm = (nullptr != DefaultRocmExecutionProvider().get());
  if (enable_cuda || enable_rocm) {
    Ort::SessionOptions session_options;
#ifdef USE_CUDA
    OrtCUDAProviderOptionsV2 cuda_options;
    cuda_options.use_tf32 = false;
    session_options.AppendExecutionProvider_CUDA_V2(cuda_options);
#endif

#ifdef USE_ROCM
    OrtROCMProviderOptions rocm_options;
    session_options.AppendExecutionProvider_ROCM(rocm_options);
#endif

    // The ONNX model is generated like the following:
    // python convert_generation.py --model_type gpt2 -m hf-internal-testing/tiny-random-gpt2
    //        --output tiny_gpt2_beamsearch_fp16.onnx  -p fp16 --use_gpu --max_length 20
    // (with separate_gpt2_decoder_for_init_run set to False as it is now set to True by default)
    Ort::Session session(*ort_env, ORT_TSTR("testdata/transformers/tiny_gpt2_beamsearch_fp16.onnx"), session_options);

    auto ort_outputs = session.Run(Ort::RunOptions{}, input_names, ort_inputs.data(), ort_inputs.size(),
                                   output_names, 1);

    ASSERT_EQ(ort_outputs.size(), 1U);
    const auto& sequences = ort_outputs[0];
    ASSERT_TRUE(sequences.IsTensor());

    auto result_ts = sequences.GetTensorTypeAndShapeInfo();
    ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, result_ts.GetElementType());

    ASSERT_EQ(expected_output_shape, result_ts.GetShape());
    const auto* result_vals = sequences.GetTensorData<int32_t>();
    auto result_span = gsl::make_span(result_vals, expected_output.size());
    ASSERT_TRUE(std::equal(expected_output.cbegin(), expected_output.cend(), result_span.begin(), result_span.end()));
  }
}

TEST(BeamSearchTest, GptBeamSearchWithInitDecoderFp16) {
  std::vector<int64_t> input_ids_shape{3, 12};
  std::vector<int32_t> input_ids{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328};

  std::vector<int64_t> parameter_shape{1};
  std::vector<int32_t> max_length{20};
  std::vector<int32_t> min_length{1};
  std::vector<int32_t> num_beams{4};
  std::vector<int32_t> num_return_sequences{1};
  std::vector<float> length_penalty{1.0f};
  std::vector<float> repetition_penalty{1.0f};

  std::vector<int64_t> expected_output_shape{input_ids_shape[0], num_return_sequences[0], max_length[0]};

  std::vector<int32_t> expected_output{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620, 131, 131, 131, 181, 638, 638, 638, 638,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572, 292, 292, 292, 292, 292, 292, 292, 292,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328, 328, 669, 669, 669, 669, 669, 669, 669};

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  auto input_ids_tensor = Ort::Value::CreateTensor(
      info, input_ids.data(), input_ids.size(), input_ids_shape.data(), input_ids_shape.size());

  auto max_length_tensor = Ort::Value::CreateTensor(
      info, max_length.data(), max_length.size(), parameter_shape.data(), parameter_shape.size());

  auto min_length_tensor = Ort::Value::CreateTensor(
      info, min_length.data(), min_length.size(), parameter_shape.data(), parameter_shape.size());

  auto num_beams_tensor = Ort::Value::CreateTensor(
      info, num_beams.data(), num_beams.size(), parameter_shape.data(), parameter_shape.size());

  auto num_return_sequences_tensor = Ort::Value::CreateTensor(
      info, num_return_sequences.data(), num_return_sequences.size(), parameter_shape.data(), parameter_shape.size());

  auto length_penalty_tensor = Ort::Value::CreateTensor(
      info, length_penalty.data(), length_penalty.size(), parameter_shape.data(), parameter_shape.size());

  auto repetition_penalty_tensor = Ort::Value::CreateTensor(
      info, repetition_penalty.data(), repetition_penalty.size(), parameter_shape.data(), parameter_shape.size());

  std::vector<Ort::Value> ort_inputs;
  ort_inputs.push_back(std::move(input_ids_tensor));
  ort_inputs.push_back(std::move(max_length_tensor));
  ort_inputs.push_back(std::move(min_length_tensor));
  ort_inputs.push_back(std::move(num_beams_tensor));
  ort_inputs.push_back(std::move(num_return_sequences_tensor));
  ort_inputs.push_back(std::move(length_penalty_tensor));
  ort_inputs.push_back(std::move(repetition_penalty_tensor));
  const char* input_names[] = {"input_ids", "max_length", "min_length", "num_beams", "num_return_sequences",
                               "length_penalty", "repetition_penalty"};
  const char* const output_names[] = {"sequences"};

  constexpr int min_cuda_architecture = 530;
  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture);
  bool enable_rocm = (nullptr != DefaultRocmExecutionProvider().get());
  if (enable_cuda || enable_rocm) {
    Ort::SessionOptions session_options;
#ifdef USE_CUDA
    OrtCUDAProviderOptionsV2 cuda_options;
    cuda_options.use_tf32 = false;
    session_options.AppendExecutionProvider_CUDA_V2(cuda_options);
#endif

#ifdef USE_ROCM
    OrtROCMProviderOptions rocm_options;
    session_options.AppendExecutionProvider_ROCM(rocm_options);
#endif

    // The ONNX model is generated like the following:
    // python convert_generation.py --model_type gpt2 -m hf-internal-testing/tiny-random-gpt2
    //        --output tiny_gpt2_beamsearch_with_init_decoder_fp16.onnx  -p fp16 --use_gpu --max_length 20
    // (with separate_gpt2_decoder_for_init_run set to True as is the default option)
    Ort::Session session(*ort_env, ORT_TSTR("testdata/transformers/tiny_gpt2_beamsearch_with_init_decoder_fp16.onnx"), session_options);

    auto ort_outputs = session.Run(Ort::RunOptions{}, input_names, ort_inputs.data(), ort_inputs.size(),
                                   output_names, 1);

    ASSERT_EQ(ort_outputs.size(), 1U);
    const auto& sequences = ort_outputs[0];
    ASSERT_TRUE(sequences.IsTensor());

    auto result_ts = sequences.GetTensorTypeAndShapeInfo();
    ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, result_ts.GetElementType());

    ASSERT_EQ(expected_output_shape, result_ts.GetShape());
    const auto* result_vals = sequences.GetTensorData<int32_t>();
    auto result_span = gsl::make_span(result_vals, expected_output.size());
    ASSERT_TRUE(std::equal(expected_output.cbegin(), expected_output.cend(), result_span.begin(), result_span.end()));
  }
}
TEST(BeamSearchTest, GptBeamSearchFp16_VocabPadded) {
  std::vector<int64_t> input_ids_shape{3, 12};
  std::vector<int32_t> input_ids{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328};

  std::vector<int64_t> parameter_shape{1};
  std::vector<int32_t> max_length{20};
  std::vector<int32_t> min_length{1};
  std::vector<int32_t> num_beams{4};
  std::vector<int32_t> num_return_sequences{1};
  std::vector<float> length_penalty{1.0f};
  std::vector<float> repetition_penalty{1.0f};

  std::vector<int64_t> expected_output_shape{input_ids_shape[0], num_return_sequences[0], max_length[0]};

  std::vector<int32_t> expected_output{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620, 131, 131, 131, 181, 638, 638, 638, 638,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572, 292, 292, 292, 292, 292, 292, 292, 292,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328, 328, 669, 669, 669, 669, 669, 669, 669};

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  auto input_ids_tensor = Ort::Value::CreateTensor(
      info, input_ids.data(), input_ids.size(), input_ids_shape.data(), input_ids_shape.size());

  auto max_length_tensor = Ort::Value::CreateTensor(
      info, max_length.data(), max_length.size(), parameter_shape.data(), parameter_shape.size());

  auto min_length_tensor = Ort::Value::CreateTensor(
      info, min_length.data(), min_length.size(), parameter_shape.data(), parameter_shape.size());

  auto num_beams_tensor = Ort::Value::CreateTensor(
      info, num_beams.data(), num_beams.size(), parameter_shape.data(), parameter_shape.size());

  auto num_return_sequences_tensor = Ort::Value::CreateTensor(
      info, num_return_sequences.data(), num_return_sequences.size(), parameter_shape.data(), parameter_shape.size());

  auto length_penalty_tensor = Ort::Value::CreateTensor(
      info, length_penalty.data(), length_penalty.size(), parameter_shape.data(), parameter_shape.size());

  auto repetition_penalty_tensor = Ort::Value::CreateTensor(
      info, repetition_penalty.data(), repetition_penalty.size(), parameter_shape.data(), parameter_shape.size());

  std::vector<Ort::Value> ort_inputs;
  ort_inputs.push_back(std::move(input_ids_tensor));
  ort_inputs.push_back(std::move(max_length_tensor));
  ort_inputs.push_back(std::move(min_length_tensor));
  ort_inputs.push_back(std::move(num_beams_tensor));
  ort_inputs.push_back(std::move(num_return_sequences_tensor));
  ort_inputs.push_back(std::move(length_penalty_tensor));
  ort_inputs.push_back(std::move(repetition_penalty_tensor));
  const char* input_names[] = {"input_ids", "max_length", "min_length", "num_beams", "num_return_sequences",
                               "length_penalty", "repetition_penalty"};
  const char* const output_names[] = {"sequences"};

  constexpr int min_cuda_architecture = 530;
  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture);
  bool enable_rocm = (nullptr != DefaultRocmExecutionProvider().get());
  if (enable_cuda || enable_rocm) {
    Ort::SessionOptions session_options;
#ifdef USE_CUDA
    OrtCUDAProviderOptionsV2 cuda_options;
    cuda_options.use_tf32 = false;
    session_options.AppendExecutionProvider_CUDA_V2(cuda_options);
#endif

#ifdef USE_ROCM
    OrtROCMProviderOptions rocm_options;
    session_options.AppendExecutionProvider_ROCM(rocm_options);
#endif

    // The following model was obtained by padding the vocabulary size in testdata/transformers/tiny_gpt2_beamsearch_fp16.onnx
    // from 1000 to 1600 (just for illustrative and testing purposes) to see if the beam search implementation can handle
    // such a scenario
    Ort::Session session(*ort_env, ORT_TSTR("testdata/transformers/tiny_gpt2_beamsearch_fp16_padded_vocab.onnx"), session_options);

    auto ort_outputs = session.Run(Ort::RunOptions{}, input_names, ort_inputs.data(), ort_inputs.size(),
                                   output_names, 1);

    ASSERT_EQ(ort_outputs.size(), 1U);
    const auto& sequences = ort_outputs[0];
    ASSERT_TRUE(sequences.IsTensor());

    auto result_ts = sequences.GetTensorTypeAndShapeInfo();
    ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, result_ts.GetElementType());

    ASSERT_EQ(expected_output_shape, result_ts.GetShape());
    const auto* result_vals = sequences.GetTensorData<int32_t>();
    auto result_span = gsl::make_span(result_vals, expected_output.size());
    ASSERT_TRUE(std::equal(expected_output.cbegin(), expected_output.cend(), result_span.begin(), result_span.end()));
  }
}




TEST(MinLengthLogitsProcessor, InitTest) {
    int min_length = 3;
    int eos_token_id = 0;
    ASSERT_EQ(min_length, 3);
    ASSERT_EQ(eos_token_id, 0);
    onnxruntime::contrib::transformers::MinLengthLogitsProcessor<float> min_length_logit_processor(min_length,
                                                                                             eos_token_id);
}


// This class keeps track of sequences generated.
// ISequences interface:
// class Sequences : public ISequences {
//  public:
//   // Initialize the sequence.
//   void Init(gsl::span<int32_t> buffer, int batch_beam_size, int sequence_length, int max_length);
// buffer is filled with 0 and has size batch_beam_size * max_length * 2
//   void InitDevice(gsl::span<int32_t> buffer);

//   // Returns a sequence of word IDs for a given beam index ( beam_index < batch_beam_size).
//   gsl::span<const int32_t> GetSequence(int beam_index) const override;
//
// this function gets a list of token (lenght of beam), each item in the list will be appended to the seuqences
// void Sequences::AppendNextTokenToSequences(gsl::span<int32_t>& next_tokens) {
// Beam search uses the other AppendNextTokenToSequences function but we don't care for testing


TEST(MinLengthLogitsProcessor, MinLengthNotReachedOneBeamTest) {
    int min_length = 3;  // bigger than sequences length
    int eos_token_id = 0;
    std::vector<float> cpu_next_token_scores_vector = {0.5, 0.2, 0.3};
    int batch_beam_size = 1;
    int vocab_size = 3;

    gsl::span<float> cpu_next_token_scores_span(gsl::make_span(
        cpu_next_token_scores_vector
        ));
    onnxruntime::contrib::transformers::MinLengthLogitsProcessor<float> min_length_logit_processor(min_length,
                                                                                             eos_token_id);
    onnxruntime::contrib::transformers::NextTokenScores<float> next_token_scores_timestamp({
        cpu_next_token_scores_span, batch_beam_size, vocab_size});
    onnxruntime::contrib::transformers::Sequences sequences;
    min_length_logit_processor.Process(&sequences, next_token_scores_timestamp);

    // we expect the eos_token_id to be set to lowest value
    // we get scores via GetScores(beam_id) -> returning scores for beam
    ASSERT_EQ(next_token_scores_timestamp.GetScores(0)[0], std::numeric_limits<float>::lowest());
}

TEST(MinLengthLogitsProcessor, MinLengthReachedOneBeamTest) {
    int min_length = 1;  // bigger than sequences length
    int max_sequence_length = 10;
    int eos_token_id = 0;
    std::vector<float> cpu_next_token_scores_vector = {0.1, 0.2, 0.3};
    int batch_beam_size = 1;
    int vocab_size = 3;
    // we create a first_token_vector that will be of length one and contain either 1 or 2, type int32_t
    // we then create a next_scores_vector that will be of length one and contain 0 , type int32_t
    // we then create a Sequences object
    // where we append the first_token_vector
    // and then we append the second_token_vector
    // and finally we create a ISequences object

    // first_token_vector creation
    std::vector<int32_t> first_token_vector = {1};
    // second_token_vector creation
    std::vector<int32_t> second_token_vector = {2};

    // create ISequences object
    onnxruntime::contrib::transformers::Sequences sequences;
    // buffer is filled with 0 and has size batch_beam_size * max_length * 2
    std::vector<int32_t> buffer_v = std::vector<int32_t>(batch_beam_size * max_sequence_length * 2, 0);
    gsl::span<int32_t> buffer_s(buffer_v);

    int start_sequence_length = 0;
    sequences.Init(buffer_s, batch_beam_size, start_sequence_length, max_sequence_length);

    // append first token
    gsl::span<int> first_token_span(first_token_vector);
    sequences.AppendNextTokenToSequences(first_token_span);
    // append second token
    gsl::span<int> second_token_span(second_token_vector);
    sequences.AppendNextTokenToSequences(second_token_span);

    gsl::span<float> cpu_next_token_scores_span(gsl::make_span(
        cpu_next_token_scores_vector
        ));
    onnxruntime::contrib::transformers::MinLengthLogitsProcessor<float> min_length_logit_processor(min_length,
                                                                                             eos_token_id);
    onnxruntime::contrib::transformers::NextTokenScores<float> next_token_scores_timestamp({
        cpu_next_token_scores_span, batch_beam_size, vocab_size});
    // create ISequences object
    sequences.Init(buffer_s, batch_beam_size, 0, 10);

    //struct ISequences {
    // virtual ~ISequences() {}
    // }
    onnxruntime::contrib::transformers::ISequences* sequences_pointer = &sequences;
    min_length_logit_processor.Process(sequences_pointer, next_token_scores_timestamp);

    // we expect the eos_token_id to be set to lowest value
    // we get scores via GetScores(beam_id) -> returning scores for beam
    ASSERT_EQ(next_token_scores_timestamp.GetScores(0)[0], std::numeric_limits<float>::lowest());
}


// avoid repetition by putting the  following code in a callable function
    // gsl::span<float> cpu_next_token_scores_span(gsl::make_span(cpu_next_token_scores_vector));
    // onnxruntime::contrib::transformers::MinLengthLogitsProcessor<float> min_length_logit_processor(min_length,
    //                                                                                          eos_token_id);
    // onnxruntime::contrib::transformers::NextTokenScores<float> next_token_scores_timestamp({cpu_next_token_scores_span, batch_beam_size, vocab_size});
    // min_length_logit_processor.Process(sequences, next_token_scores_timestamp);

void call_MinLengthLogitProcessor(int min_length, int eos_token_id, std::vector<float> cpu_next_token_scores_vector, int batch_beam_size, int vocab_size) {
    gsl::span<float> cpu_next_token_scores_span(gsl::make_span(
        cpu_next_token_scores_vector
        ));
    onnxruntime::contrib::transformers::MinLengthLogitsProcessor<float> min_length_logit_processor(min_length,
                                                                                             eos_token_id);
    onnxruntime::contrib::transformers::NextTokenScores<float> next_token_scores_timestamp({
        cpu_next_token_scores_span, batch_beam_size, vocab_size});
    onnxruntime::contrib::transformers::ISequences* sequences = nullptr;
    min_length_logit_processor.Process(sequences, next_token_scores_timestamp);
}

TEST(JeroenTest, JeroenSeedTest) {  // Just here to verify tests are discovered, not a real test
  ASSERT_EQ(8211, 8211);
}  // keeping at end because seems tests output gets trunated sometimes




}  // namespace test
}  // namespace onnxruntime
