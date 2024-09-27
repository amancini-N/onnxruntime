// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <vector>
#include "gtest/gtest.h"
#include "core/common/gsl.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "test/common/cuda_op_test_utils.h"
#include "contrib_ops/cpu/transformers/logits_processor.h"

#ifdef USE_CUDA
#include "core/providers/cuda/cuda_provider_options.h"
#endif

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

TEST(MinLengthLogitsProcessor, InitTest) {
  int min_length = 3;
  int eos_token_id = 0;
  ASSERT_EQ(min_length, 3);
  ASSERT_EQ(eos_token_id, 0);
  onnxruntime::contrib::transformers::MinLengthLogitsProcessor<float> min_length_logit_processor(min_length,
                                                                                                 eos_token_id);
}

template <typename T>
void callLogitProcessor(
    int batch_beam_size,
    onnxruntime::contrib::transformers::NextTokenScores<float> next_token_scores,
    std::vector<std::vector<int32_t>> token_sequence_vectors,
    onnxruntime::contrib::transformers::ILogitsProcessor<T>* logit_processor) {
  // token_sequence_vectors are expected to be {{token_0_beam_0, token_0_beam_1}, {token_1_beam_0, token_1_beam_1}, ..}
  int max_sequence_length = 100;
  // buffer is filled with 0 and has size batch_beam_size * max_length * 2
  std::vector<int32_t> buffer_v = std::vector<int32_t>(batch_beam_size * max_sequence_length * 2, 0);
  gsl::span<int32_t> buffer_s(buffer_v);

  // create ISequences object
  int start_sequence_length = 0;
  onnxruntime::contrib::transformers::Sequences sequences;
  sequences.Init(buffer_s, batch_beam_size, start_sequence_length, max_sequence_length);

  // transpose token_sequence_vectors, assume inner length is always same
  std::vector<std::vector<int>> transpose_token_sequence_vectors;
  for (int i = 0; i < static_cast<int>(token_sequence_vectors[0].size()); i++) {
    std::vector<int> token_vector;
    for (int j = 0; j < static_cast<int>(token_sequence_vectors.size()); j++) {
      token_vector.push_back(token_sequence_vectors[j][i]);
    }
    transpose_token_sequence_vectors.push_back(token_vector);
  }

  // adding all token vectors
  for (auto token_vector : transpose_token_sequence_vectors) {
    gsl::span<int> token_span = AsSpan(token_vector);
    sequences.AppendNextTokenToSequences(token_span);
  }
  onnxruntime::contrib::transformers::ISequences* sequences_pointer = &sequences;

  logit_processor->Process(sequences_pointer, next_token_scores);
}

TEST(NoRepeatNGramLogitsProcessor, FormatUnigramAny) {
  int batch_beam_size = 2;
  int vocab_size = 6;
  auto tokens = std::vector<std::vector<int>>{{1, 2, 3, 4, 1},
                                              {2, 3, 3, 3, 2}};
  auto scores_v = std::vector<float>(batch_beam_size * vocab_size, 0.0);
  auto scores_span = gsl::make_span(scores_v);
  auto scores = onnxruntime::contrib::transformers::NextTokenScores<float>{scores_span, batch_beam_size, vocab_size};

  auto logit_processor = onnxruntime::contrib::transformers::NoRepeatNGramLogitsProcessor<float>(
      /*ngram_size=*/{2},
      /*ngram_history_a=*/0,
      /*ngram_history_b=*/-1,
      /*ngram_format_mode=*/1,
      /*ngram_format_tokens=*/{1, 5},
      /*ngram_format_tokens_num_exclusions=*/2,
      /*ngram_format_tokens_max_exclusion_length=*/1);

  callLogitProcessor(batch_beam_size, scores, tokens, &logit_processor);

  auto low = std::numeric_limits<float>::lowest();
  auto expected_scores = std::vector<std::vector<float>>{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                                         {0.0, 0.0, 0.0, low, 0.0, 0.0}};
  for (int i = 0; i < batch_beam_size; i++) {
    for (int j = 0; j < vocab_size; j++) {
      ASSERT_EQ(scores.GetScores(i)[j], expected_scores[i][j]);
    }
  }
}

TEST(NoRepeatNGramLogitsProcessor, FormatMultigramAny) {
  int batch_beam_size = 3;
  int vocab_size = 6;
  auto tokens = std::vector<std::vector<int>>{{1, 2, 3, 1, 4, 1},
                                              {1, 2, 3, 1, 4, 1},
                                              {1, 2, 3, 1, 3, 1}};
  auto scores_v = std::vector<float>(batch_beam_size * vocab_size, 0.0);
  auto scores_span = gsl::make_span(scores_v);
  auto scores = onnxruntime::contrib::transformers::NextTokenScores<float>{scores_span, batch_beam_size, vocab_size};

  auto logit_processor = onnxruntime::contrib::transformers::NoRepeatNGramLogitsProcessor<float>(
      /*ngram_size=*/{2},
      /*ngram_history_a=*/0,
      /*ngram_history_b=*/-1,
      /*ngram_format_mode=*/1,
      /*ngram_format_tokens=*/{1, 2, 5, 0, 3, 0},
      /*ngram_format_tokens_num_exclusions=*/3,
      /*ngram_format_tokens_max_exclusion_length=*/2);

  callLogitProcessor(batch_beam_size, scores, tokens, &logit_processor);

  auto low = std::numeric_limits<float>::lowest();
  auto expected_scores = std::vector<std::vector<float>>{{0.0, 0.0, 0.0, 0.0, low, 0.0},
                                                         {0.0, 0.0, 0.0, 0.0, low, 0.0},
                                                         {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
  for (int i = 0; i < batch_beam_size; i++) {
    for (int j = 0; j < vocab_size; j++) {
      ASSERT_EQ(scores.GetScores(i)[j], expected_scores[i][j]);
    }
  }
}

TEST(NoRepeatNGramLogitsProcessor, FormatMultigramAll) {
  int batch_beam_size = 3;
  int vocab_size = 6;
  auto tokens = std::vector<std::vector<int>>{{1, 2, 3, 1, 4, 1},
                                              {1, 2, 3, 1, 4, 1},
                                              {5, 2, 3, 1, 4, 5}};
  auto scores_v = std::vector<float>(batch_beam_size * vocab_size, 0.0);
  auto scores_span = gsl::make_span(scores_v);
  auto scores = onnxruntime::contrib::transformers::NextTokenScores<float>{scores_span, batch_beam_size, vocab_size};

  auto logit_processor = onnxruntime::contrib::transformers::NoRepeatNGramLogitsProcessor<float>(
      /*ngram_size=*/{2},
      /*ngram_history_a=*/0,
      /*ngram_history_b=*/-1,
      /*ngram_format_mode=*/0,
      /*ngram_format_tokens=*/{1, 2, 4, 0, 3, 0},
      /*ngram_format_tokens_num_exclusions=*/3,
      /*ngram_format_tokens_max_exclusion_length=*/2);

  callLogitProcessor(batch_beam_size, scores, tokens, &logit_processor);

  auto low = std::numeric_limits<float>::lowest();
  auto expected_scores = std::vector<std::vector<float>>{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                                         {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                                         {0.0, 0.0, low, 0.0, 0.0, 0.0}};
  for (int i = 0; i < batch_beam_size; i++) {
    for (int j = 0; j < vocab_size; j++) {
      ASSERT_EQ(scores.GetScores(i)[j], expected_scores[i][j]);
    }
  }
}

TEST(NoRepeatNGramLogitsProcessor, FormatMultigramSimple) {
  int batch_beam_size = 3;
  int vocab_size = 6;
  auto tokens = std::vector<std::vector<int>>{{1, 2, 3, 1, 4, 1},
                                              {1, 2, 3, 1, 4, 1},
                                              {1, 5, 3, 1, 4, 1}};
  auto scores_v = std::vector<float>(batch_beam_size * vocab_size, 0.0);
  auto scores_span = gsl::make_span(scores_v);
  auto scores = onnxruntime::contrib::transformers::NextTokenScores<float>{scores_span, batch_beam_size, vocab_size};

  auto logit_processor = onnxruntime::contrib::transformers::NoRepeatNGramLogitsProcessor<float>(
      /*ngram_size=*/{2},
      /*ngram_history_a=*/2,
      /*ngram_history_b=*/3,
      /*ngram_format_mode=*/2,
      /*ngram_format_tokens=*/{1, 2, 4, 0, 3, 0},
      /*ngram_format_tokens_num_exclusions=*/3,
      /*ngram_format_tokens_max_exclusion_length=*/2);

  callLogitProcessor(batch_beam_size, scores, tokens, &logit_processor);

  auto low = std::numeric_limits<float>::lowest();
  auto expected_scores = std::vector<std::vector<float>>{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                                         {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                                         {0.0, 0.0, 0.0, 0.0, 0.0, low}};
  for (int i = 0; i < batch_beam_size; i++) {
    for (int j = 0; j < vocab_size; j++) {
      ASSERT_EQ(scores.GetScores(i)[j], expected_scores[i][j]);
    }
  }
}

TEST(NoRepeatNGramLogitsProcessor, FormatUnigramAnyVariableSizedNGramHistory) {
  int batch_beam_size = 4;
  int vocab_size = 6;
  auto tokens = std::vector<std::vector<int>>{{1, 2, 5, 1, 4, 2},   // For 2-gram history: After 2, we cannot generate 5. (both don't belong to format tokens)
                                              {1, 2, 3, 1, 1, 2},   // For 2 gram history: There's no block of 4 because it's a format token
                                              {5, 2, 0, 1, 5, 2},   // For 3-grams history: 0 should be blocked because it's not in format tokens
                                              {5, 2, 1, 1, 5, 2}};  // For 3-grams history: There's no block of 1 because it's a format token
  auto scores_v = std::vector<float>(batch_beam_size * vocab_size, 0.0);
  auto scores_span = gsl::make_span(scores_v);
  auto scores = onnxruntime::contrib::transformers::NextTokenScores<float>{scores_span, batch_beam_size, vocab_size};

  auto logit_processor = onnxruntime::contrib::transformers::NoRepeatNGramLogitsProcessor<float>(
      /*ngram_size=*/{2, 3},
      /*ngram_history_a=*/2,
      /*ngram_history_b=*/3,
      /*ngram_format_mode=*/1,
      /*ngram_format_tokens=*/{1, 4, 3},
      /*ngram_format_tokens_num_exclusions=*/3,
      /*ngram_format_tokens_max_exclusion_length=*/1);

  callLogitProcessor(batch_beam_size, scores, tokens, &logit_processor);

  auto low = std::numeric_limits<float>::lowest();
  auto expected_scores = std::vector<std::vector<float>>{{0.0, 0.0, 0.0, 0.0, 0.0, low},
                                                         {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                                         {low, 0.0, 0.0, 0.0, 0.0, 0.0},
                                                         {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
  for (int i = 0; i < batch_beam_size; i++) {
    for (int j = 0; j < vocab_size; j++) {
      ASSERT_EQ(scores.GetScores(i)[j], expected_scores[i][j]);
    }
  }
}

TEST(NoRepeatNGramLogitsProcessor, FormatUnigramAllVariableSizedNGramHistory) {
  int batch_beam_size = 4;
  int vocab_size = 6;
  auto tokens = std::vector<std::vector<int>>{{1, 2, 5, 1, 4, 2},   // For 2-gram history: After 2, we cannot generate 5. (both don't belong to format tokens)
                                              {1, 2, 3, 1, 1, 2},   // For 2 gram history: There's a block of 3 because (2,3), only 3 belong to format tokens
                                              {5, 2, 0, 1, 5, 2},   // For 3-grams history: 0 should be blocked because it's not in format tokens
                                              {5, 2, 1, 1, 5, 2}};  // For 3-grams history: There's a block of 1 because (5,2,1), only 1 belong to format token
  auto scores_v = std::vector<float>(batch_beam_size * vocab_size, 0.0);
  auto scores_span = gsl::make_span(scores_v);
  auto scores = onnxruntime::contrib::transformers::NextTokenScores<float>{scores_span, batch_beam_size, vocab_size};

  auto logit_processor = onnxruntime::contrib::transformers::NoRepeatNGramLogitsProcessor<float>(
      /*ngram_size=*/{2, 3},
      /*ngram_history_a=*/2,
      /*ngram_history_b=*/3,
      /*ngram_format_mode=*/0,
      /*ngram_format_tokens=*/{1, 4, 3},
      /*ngram_format_tokens_num_exclusions=*/3,
      /*ngram_format_tokens_max_exclusion_length=*/1);

  callLogitProcessor(batch_beam_size, scores, tokens, &logit_processor);

  auto low = std::numeric_limits<float>::lowest();
  auto expected_scores = std::vector<std::vector<float>>{{0.0, 0.0, 0.0, 0.0, 0.0, low},
                                                         {0.0, 0.0, 0.0, low, 0.0, 0.0},
                                                         {low, 0.0, 0.0, 0.0, 0.0, 0.0},
                                                         {0.0, low, 0.0, 0.0, 0.0, 0.0}};
  for (int i = 0; i < batch_beam_size; i++) {
    for (int j = 0; j < vocab_size; j++) {
      ASSERT_EQ(scores.GetScores(i)[j], expected_scores[i][j]);
    }
  }
}

TEST(NoRepeatNGramLogitsProcessor, FormatUnigramSimpleVariableSizedNGramHistory) {
  int batch_beam_size = 4;
  int vocab_size = 6;
  auto tokens = std::vector<std::vector<int>>{{1, 2, 5, 1, 4, 2},   // For 2-gram history: After 2, we cannot generate 5. (both don't belong to format tokens)
                                              {1, 2, 3, 1, 1, 2},   // For 2 gram history: There's a block of 3 because (2,3), only 3 belong to format tokens
                                              {5, 2, 0, 1, 5, 2},   // For 3-grams history: 0 should be blocked because it's not in format tokens
                                              {5, 2, 1, 1, 5, 2}};  // For 3-grams history: There's a block of 1 because (5,2,1), only 1 belong to format token
  auto scores_v = std::vector<float>(batch_beam_size * vocab_size, 0.0);
  auto scores_span = gsl::make_span(scores_v);
  auto scores = onnxruntime::contrib::transformers::NextTokenScores<float>{scores_span, batch_beam_size, vocab_size};

  auto logit_processor = onnxruntime::contrib::transformers::NoRepeatNGramLogitsProcessor<float>(
      /*ngram_size=*/{2, 3},
      /*ngram_history_a=*/2,
      /*ngram_history_b=*/3,
      /*ngram_format_mode=*/2,
      /*ngram_format_tokens=*/{1, 4, 3},
      /*ngram_format_tokens_num_exclusions=*/3,
      /*ngram_format_tokens_max_exclusion_length=*/1);

  callLogitProcessor(batch_beam_size, scores, tokens, &logit_processor);

  auto low = std::numeric_limits<float>::lowest();
  auto expected_scores = std::vector<std::vector<float>>{{0.0, 0.0, 0.0, 0.0, 0.0, low},
                                                         {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                                         {low, 0.0, 0.0, 0.0, 0.0, 0.0},
                                                         {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
  for (int i = 0; i < batch_beam_size; i++) {
    for (int j = 0; j < vocab_size; j++) {
      ASSERT_EQ(scores.GetScores(i)[j], expected_scores[i][j]);
    }
  }
}

TEST(MinLengthLogitsProcessor, MinLengthNotReached) {
  int min_length = 3;  // bigger than sequences length
  int eos_token_id = 0;
  std::vector<float> next_token_scores_vector = {0.5, 0.2, 0.3};
  int batch_beam_size = 1;
  int vocab_size = 3;

  std::vector<std::vector<int32_t>> token_vectors = {{}, {}, {}};
  gsl::span<float> next_token_scores_span(gsl::make_span(
      next_token_scores_vector));
  onnxruntime::contrib::transformers::NextTokenScores<float> next_token_scores({next_token_scores_span, batch_beam_size, vocab_size});

  onnxruntime::contrib::transformers::MinLengthLogitsProcessor<float> min_length_logit_processor(min_length,
                                                                                                 eos_token_id);
  callLogitProcessor(
      batch_beam_size,
      next_token_scores,
      token_vectors,
      &min_length_logit_processor);

  // we expect the eos_token_id to be set to lowest value
  // we get scores via GetScores(beam_id) -> returning scores for beam
  ASSERT_EQ(next_token_scores.GetScores(0)[0], std::numeric_limits<float>::lowest());
}

TEST(MinLengthLogitsProcessor, MinLengthReached) {
  int min_length = 1;  // bigger than sequences length
  int eos_token_id = 0;
  std::vector<float> next_token_scores_vector = {0.1, 0.2, 0.3};
  int batch_beam_size = 1;
  int vocab_size = 3;

  std::vector<std::vector<int32_t>> token_vectors = {{1, 2}};

  gsl::span<float> next_token_scores_span(gsl::make_span(
      next_token_scores_vector));
  onnxruntime::contrib::transformers::NextTokenScores<float> next_token_scores({next_token_scores_span, batch_beam_size, vocab_size});

  onnxruntime::contrib::transformers::MinLengthLogitsProcessor<float> min_length_logit_processor(min_length,
                                                                                                 eos_token_id);
  callLogitProcessor(
      batch_beam_size,
      next_token_scores,
      token_vectors,
      &min_length_logit_processor);

  ASSERT_NEAR(next_token_scores.GetScores(0)[0], 0.1, 0.0001);
}

TEST(MaxLengthLogitsProcessor, MaxLengthReached) {
  int max_length = 1;  // bigger than sequences length
  int eos_token_id = 0;
  std::vector<float> next_token_scores_vector = {0.1, 0.2, 0.3};
  int batch_beam_size = 1;
  int vocab_size = 3;

  std::vector<std::vector<int32_t>> token_vectors = {{1, 2}};

  gsl::span<float> next_token_scores_span(gsl::make_span(
      next_token_scores_vector));
  onnxruntime::contrib::transformers::NextTokenScores<float> next_token_scores({next_token_scores_span, batch_beam_size, vocab_size});

  onnxruntime::contrib::transformers::MaxLengthLogitsProcessor<float> max_length_logit_processor(max_length,
                                                                                                 eos_token_id);
  callLogitProcessor(
      batch_beam_size,
      next_token_scores,
      token_vectors,
      &max_length_logit_processor);

  ASSERT_EQ(next_token_scores.GetScores(0)[1], std::numeric_limits<float>::lowest());
  ASSERT_EQ(next_token_scores.GetScores(0)[2], std::numeric_limits<float>::lowest());
}

TEST(MaxLengthLogitsProcessor, MaxLengthNotReached) {
  int max_length = 10;  // bigger than sequences length
  int eos_token_id = 0;
  std::vector<float> next_token_scores_vector = {0.1, 0.2, 0.3};
  int batch_beam_size = 1;
  int vocab_size = 3;

  std::vector<std::vector<int32_t>> token_vectors = {{1, 2}};

  gsl::span<float> next_token_scores_span(gsl::make_span(
      next_token_scores_vector));
  onnxruntime::contrib::transformers::NextTokenScores<float> next_token_scores({next_token_scores_span, batch_beam_size, vocab_size});

  onnxruntime::contrib::transformers::MaxLengthLogitsProcessor<float> max_length_logit_processor(max_length,
                                                                                                 eos_token_id);
  callLogitProcessor(
      batch_beam_size,
      next_token_scores,
      token_vectors,
      &max_length_logit_processor);

  ASSERT_NEAR(next_token_scores.GetScores(0)[0], 0.1, 0.0001);
}

std::vector<std::vector<float>> SequentialConstraintsTestRunner(
    int batch_beam_size,
    int vocab_size,
    int max_grammar_rule_length,
    std::vector<std::vector<int32_t>> grammar,
    std::vector<int32_t> constraints,
    std::vector<std::vector<int32_t>> token_sequence_vectors,
    std::vector<std::vector<float>> next_scores_vector) {
  // processing vectors into appropriate objects
  std::vector<float> flattened_next_scores_vector;
  for (auto& scores : next_scores_vector) {
    for (auto& score : scores) {
      flattened_next_scores_vector.push_back(score);
    }
  }
  gsl::span<float> next_scores_span(gsl::make_span(
      flattened_next_scores_vector));
  onnxruntime::contrib::transformers::NextTokenScores<float> next_token_scores({next_scores_span,
                                                                                batch_beam_size,
                                                                                vocab_size});

  assert(static_cast<int>(grammar.size()) == vocab_size);

  gsl::span<int32_t> constraints_span(gsl::make_span(
      constraints));
  std::vector<int32_t> flattened_grammar;
  for (auto& rule : grammar) {
    assert(static_cast<int>(rule.size()) == max_grammar_rule_length);
    for (auto& token : rule) {
      flattened_grammar.push_back(token);
    }
  }

  gsl::span<int32_t> grammar_span(gsl::make_span(
      flattened_grammar));

  onnxruntime::contrib::transformers::SequentialConstraintsFSALogitsProcessor<float> fsa_logit_processor(
      constraints_span,
      grammar_span,
      batch_beam_size,
      max_grammar_rule_length,
      vocab_size);

  // loop over the sequence_vectors, only take the first i-th tokens in the sequence
  // only when we pass the whole original sequence vector,
  // we should pass next_token_scores, otherwise we provide fake values (we don't care about those)
  const int max_sequence_length = static_cast<int>(token_sequence_vectors[0].size());
  for (int current_sequence_max = 0; current_sequence_max < max_sequence_length; current_sequence_max++) {
    if (current_sequence_max == max_sequence_length - 1) {
      callLogitProcessor(
          batch_beam_size,
          next_token_scores,
          token_sequence_vectors,
          &fsa_logit_processor);
    } else {
      std::vector<std::vector<int32_t>> partial_token_sequence_vectors;
      // partial_token _sequence vector contains one vector for each batch of size batch_beam_size
      // we take thus for each batch item the first i tokens
      for (int beam_index = 0; beam_index < batch_beam_size; beam_index++) {
        std::vector<int32_t> current_token_vector;
        for (int sequence_token_id = 0; sequence_token_id <= current_sequence_max; sequence_token_id++) {
          current_token_vector.push_back(token_sequence_vectors[beam_index][sequence_token_id]);
        }
        partial_token_sequence_vectors.push_back(current_token_vector);
      }
      // create fake next_token_scores
      std::vector<float> fake_next_token_scores_vector;
      for (int j = 0; j < batch_beam_size * vocab_size; j++) {
        fake_next_token_scores_vector.push_back(0.0);
      }
      gsl::span<float> fake_next_scores_span(gsl::make_span(
          fake_next_token_scores_vector));
      onnxruntime::contrib::transformers::NextTokenScores<float> fake_next_token_scores({fake_next_scores_span,
                                                                                         batch_beam_size,
                                                                                         vocab_size});
      callLogitProcessor(
          batch_beam_size,
          fake_next_token_scores,
          partial_token_sequence_vectors,
          &fsa_logit_processor);
    }
  }

  std::vector<std::vector<float>> next_scores_vector_out;
  for (int i = 0; i < batch_beam_size; i++) {
    std::vector<float> scores;

    gsl::span<float> return_scores = next_token_scores.GetScores(i);
    for (auto& score : return_scores) {
      scores.push_back(score);
    }
    next_scores_vector_out.push_back(scores);
  }
  return next_scores_vector_out;
}

TEST(SequentialConstraintsFSALogitsProcessor, OneBeamAnyRuleNoConstraintYet) {
  int batch_beam_size = 1;
  int vocab_size = 3;
  int max_grammar_rule_length = 2;

  std::vector<std::vector<int32_t>> grammar = {{-2, -3}, {-2, -3}, {-2, -3}};
  std::vector<int32_t> constraints = {1, 2};

  std::vector<std::vector<int32_t>> token_sequence_vectors = {{0}};
  std::vector<std::vector<float>> next_scores_vector = {{0.1, 0.2, 0.3}};

  std::vector<std::vector<float>> result_scores_vector = SequentialConstraintsTestRunner(
      batch_beam_size,
      vocab_size,
      max_grammar_rule_length,
      grammar,
      constraints,
      token_sequence_vectors,
      next_scores_vector);
  // only second score changed
  ASSERT_NEAR(result_scores_vector[0][0], 0.1, 0.0001);
  ASSERT_NEAR(result_scores_vector[0][1], 0.2, 0.0001);
  ASSERT_EQ(result_scores_vector[0][2], std::numeric_limits<float>::lowest());
}

TEST(SequentialConstraintsFSALogitsProcessor, OneBeamAnyRuleLastTokenIsCurrentConstraint) {
  int batch_beam_size = 1;
  int vocab_size = 3;
  int max_grammar_rule_length = 2;

  std::vector<std::vector<int32_t>> grammar = {{-2, -3}, {-2, -3}, {-2, -3}};
  std::vector<int32_t> constraints = {1, 2};

  std::vector<std::vector<int32_t>> token_sequence_vectors = {{1}};
  std::vector<std::vector<float>> next_scores_vector = {{0.1, 0.2, 0.3}};

  std::vector<std::vector<float>> result_scores_vector = SequentialConstraintsTestRunner(
      batch_beam_size,
      vocab_size,
      max_grammar_rule_length,
      grammar,
      constraints,
      token_sequence_vectors,
      next_scores_vector);
  // only second score changed
  ASSERT_NEAR(result_scores_vector[0][0], 0.1, 0.0001);
  ASSERT_EQ(result_scores_vector[0][1], std::numeric_limits<float>::lowest());
  ASSERT_NEAR(result_scores_vector[0][2], 0.3, 0.0001);
}

TEST(SequentialConstraintsFSALogitsProcessor, OneBeamAnyRuleAlreadyOneConstraint) {
  int batch_beam_size = 1;
  int vocab_size = 4;
  int max_grammar_rule_length = 2;

  std::vector<std::vector<int32_t>> grammar = {{-2, -3}, {-2, -3}, {-2, -3}, {-2, -3}};
  std::vector<int32_t> constraints = {1, 2};

  std::vector<std::vector<int32_t>> token_sequence_vectors = {{1, 0}};
  std::vector<std::vector<float>> next_scores_vector = {{0.1, 0.2, 0.3, 0.4}};

  std::vector<std::vector<float>> result_scores_vector = SequentialConstraintsTestRunner(
      batch_beam_size,
      vocab_size,
      max_grammar_rule_length,
      grammar,
      constraints,
      token_sequence_vectors,
      next_scores_vector);
  // only second score changed
  ASSERT_NEAR(result_scores_vector[0][0], 0.1, 0.0001);
  ASSERT_EQ(result_scores_vector[0][1], std::numeric_limits<float>::lowest());
  ASSERT_NEAR(result_scores_vector[0][2], 0.3, 0.0001);
  ASSERT_NEAR(result_scores_vector[0][3], 0.4, 0.0001);
}

TEST(SequentialConstraintsFSALogitsProcessor, SpecificConstraints) {
  int batch_beam_size = 1;
  int vocab_size = 4;
  int max_grammar_rule_length = 2;

  std::vector<std::vector<int32_t>> grammar = {{3, -3}, {3, -3}, {3, -3}, {3, -3}};
  std::vector<int32_t> constraints = {1, 2};

  std::vector<std::vector<int32_t>> token_sequence_vectors = {{1}};
  std::vector<std::vector<float>> next_scores_vector = {{0.1, 0.2, 0.3, 0.4}};

  std::vector<std::vector<float>> result_scores_vector = SequentialConstraintsTestRunner(
      batch_beam_size,
      vocab_size,
      max_grammar_rule_length,
      grammar,
      constraints,
      token_sequence_vectors,
      next_scores_vector);
  // only second score changed
  ASSERT_EQ(result_scores_vector[0][0], std::numeric_limits<float>::lowest());  // not allowed due to not ANY grammar rule (-2)
  ASSERT_EQ(result_scores_vector[0][1], std::numeric_limits<float>::lowest());  // not allowed due to constraint already reached
  ASSERT_NEAR(result_scores_vector[0][2], 0.3, 0.0001);                         // remains same because next constraint
  ASSERT_NEAR(result_scores_vector[0][3], 0.4, 0.0001);                         // ecplicitly allowed
}

// The SpecificConstraints test mask an issue
// Because we don't have not allowed token with index > max_grammar_rule_length and not otherwise explicitly disallowed
// This was an issue because in the implementation we were looping at one point to the end of the grammar rules instead of the end of the vocab size
TEST(SequentialConstraintsFSALogitsProcessor, SpecificConstraintsCheckRandomNonSpecificToken) {
  int batch_beam_size = 1;
  int vocab_size = 5;
  int max_grammar_rule_length = 3;

  std::vector<std::vector<int32_t>> grammar = {
      {-3, -2, -1},
      {-3, 4, -1},  // important one for testing
      {-3, -2, -1},
      {-3, -2, -1},
      {-3, -2, -1},
  };
  std::vector<int32_t> constraints = {3};

  std::vector<std::vector<int32_t>> token_sequence_vectors = {{1}};
  std::vector<std::vector<float>> next_scores_vector = {{0.1, 0.2, 0.3, 0.4, 0.5}};

  std::vector<std::vector<float>> result_scores_vector = SequentialConstraintsTestRunner(
      batch_beam_size,
      vocab_size,
      max_grammar_rule_length,
      grammar,
      constraints,
      token_sequence_vectors,
      next_scores_vector);
  // only second score changed
  ASSERT_EQ(result_scores_vector[0][0], std::numeric_limits<float>::lowest());  // not allowed due to not ANY grammar rule (-2)
  ASSERT_EQ(result_scores_vector[0][1], std::numeric_limits<float>::lowest());  // not allowed due to not ANY grammar rule (-2)
  ASSERT_EQ(result_scores_vector[0][2], std::numeric_limits<float>::lowest());  // not allowed due to not ANY grammar rule (-2), greater than max_grammar_rule_length
  ASSERT_NEAR(result_scores_vector[0][3], 0.4, 0.0001);                         // explicitly allowed due to next contstraint
  ASSERT_NEAR(result_scores_vector[0][4], 0.5, 0.0001);                         // explicitly allowed
}

TEST(SequentialConstraintsFSALogitsProcessor, OneBeamAnyRuleAllConstraintsReached) {
  int batch_beam_size = 1;
  int vocab_size = 3;
  int max_grammar_rule_length = 2;

  std::vector<std::vector<int32_t>> grammar = {{-2, -3}, {-2, -3}, {-2, -3}};
  std::vector<int32_t> constraints = {1, 2};

  std::vector<std::vector<int32_t>> token_sequence_vectors = {{1, 2}};
  std::vector<std::vector<float>> next_scores_vector = {{0.1, 0.2, 0.3}};

  std::vector<std::vector<float>> result_scores_vector = SequentialConstraintsTestRunner(
      batch_beam_size,
      vocab_size,
      max_grammar_rule_length,
      grammar,
      constraints,
      token_sequence_vectors,
      next_scores_vector);
  // only second score changed
  ASSERT_NEAR(result_scores_vector[0][0], 0.1, 0.0001);
  ASSERT_EQ(result_scores_vector[0][1], std::numeric_limits<float>::lowest());
  ASSERT_EQ(result_scores_vector[0][2], std::numeric_limits<float>::lowest());
}

TEST(SequentialConstraintsFSALogitsProcessor, MultiBeamAnyRuleDifferentConstraintReached) {
  int batch_beam_size = 2;
  int vocab_size = 3;
  int max_grammar_rule_length = 2;

  std::vector<std::vector<int32_t>> grammar = {{-2, -3}, {-2, -3}, {-2, -3}};
  std::vector<int32_t> constraints = {1, 2};

  std::vector<std::vector<int32_t>> token_sequence_vectors = {{1, 2}, {1, 0}};
  std::vector<std::vector<float>> next_scores_vector = {{0.1, 0.2, 0.3}, {0.1, 0.2, 0.3}};

  std::vector<std::vector<float>> result_scores_vector = SequentialConstraintsTestRunner(
      batch_beam_size,
      vocab_size,
      max_grammar_rule_length,
      grammar,
      constraints,
      token_sequence_vectors,
      next_scores_vector);
  //  first beam reached all constraints, so only first token is allowed
  ASSERT_NEAR(result_scores_vector[0][0], 0.1, 0.0001);
  ASSERT_EQ(result_scores_vector[0][1], std::numeric_limits<float>::lowest());
  ASSERT_EQ(result_scores_vector[0][2], std::numeric_limits<float>::lowest());
  //  second beam reached only one constraint, so second constraint token (third token) is still allowed
  ASSERT_NEAR(result_scores_vector[1][0], 0.1, 0.0001);
  ASSERT_EQ(result_scores_vector[1][1], std::numeric_limits<float>::lowest());
  ASSERT_NEAR(result_scores_vector[1][2], 0.3, 0.0001);
}

}  // namespace test
}  // namespace onnxruntime
