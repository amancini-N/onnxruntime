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

template <typename T>
void callLogitProcessor(
    int batch_beam_size,
    onnxruntime::contrib::transformers::NextTokenScores<float> next_token_scores,
    std::vector<std::vector<int32_t>> token_sequence_vectors,
    onnxruntime::contrib::transformers::ILogitsProcessor<T>* logit_processor
) {
    //token_sequence_vectors are expected to be {{token_0_beam_0, token_0_beam_1}, {token_1_beam_0, token_1_beam_1}, ..}
    int max_sequence_length = 100;
    // buffer is filled with 0 and has size batch_beam_size * max_length * 2
    std::vector<int32_t> buffer_v = std::vector<int32_t>(batch_beam_size * max_sequence_length * 2, 0);
    gsl::span<int32_t> buffer_s(buffer_v);

    // create ISequences object
    int start_sequence_length = 0;
    onnxruntime::contrib::transformers::Sequences sequences;
    sequences.Init(buffer_s, batch_beam_size, start_sequence_length, max_sequence_length);
    // adding all token vectors
    for (auto token_vector : token_sequence_vectors) {
        gsl::span<int> token_span(token_vector);
        sequences.AppendNextTokenToSequences(token_span);
    }
    onnxruntime::contrib::transformers::ISequences* sequences_pointer = &sequences;


    logit_processor->Process(sequences_pointer, next_token_scores);
}

TEST(NoRepeatNGramLogitsProcessor, FormatUnigramAny) {
    int batch_beam_size = 2;
    int vocab_size = 6;
    auto tokens = std::vector<std::vector<int>>{{1, 2, 3, 4, 1, 2, 3}, {2, 3, 3, 3, 2, 3, 2}};
    auto scores_v = std::vector<float>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
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

    auto expected_scores = std::vector<std::vector<float>>{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, std::numeric_limits<float>::lowest(), 0.0, 0.0}};
    for (int i = 0; i < batch_beam_size; i++) {
        for (int j = 0; j < vocab_size; j++) {
            ASSERT_EQ(scores.GetScores(i)[j], expected_scores[i][j]);
        }
    }
}

TEST(NoRepeatNGramLogitsProcessor, FormatMultigramAny) {
    int batch_beam_size = 3;
    int vocab_size = 6;
    auto tokens = std::vector<std::vector<int>>{{1, 2, 3, 1, 4, 1, 2, 1, 4, 3}, {1, 2, 3, 1, 4, 1, 4, 1, 4, 3}, {1, 2, 3, 1, 3, 1, 4, 1, 4, 3}};
    auto scores_v = std::vector<float>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
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

    auto expected_scores = std::vector<std::vector<float>>{{0.0, 0.0, 0.0, 0.0, std::numeric_limits<float>::lowest(), 0.0},
                                                           {0.0, 0.0, 0.0, 0.0, std::numeric_limits<float>::lowest(), 0.0},
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
    auto tokens = std::vector<std::vector<int>>{{1, 2, 3, 1, 4, 1, 2}, {1, 2, 3, 1, 4, 1, 4}, {5, 2, 3, 1, 4, 5, 2}};
    auto scores_v = std::vector<float>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
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

    auto expected_scores = std::vector<std::vector<float>>{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                                           {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                                           {0.0, 0.0, std::numeric_limits<float>::lowest(), 0.0, 0.0, 0.0}};
    for (int i = 0; i < batch_beam_size; i++) {
        for (int j = 0; j < vocab_size; j++) {
            ASSERT_EQ(scores.GetScores(i)[j], expected_scores[i][j]);
        }
    }
}


TEST(NoRepeatNGramLogitsProcessor, FormatMultigramSimple) {
    int batch_beam_size = 3;
    int vocab_size = 6;
    auto tokens = std::vector<std::vector<int>>{{1, 2, 3, 1, 4, 1, 2, 1, 4, 3},
                                                {1, 2, 3, 1, 4, 1, 4, 1, 4, 3},
                                                {1, 5, 3, 1, 4, 1, 4, 1, 4, 3}};
    auto scores_v = std::vector<float>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
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

    auto expected_scores = std::vector<std::vector<float>>{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                                           {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                                           {0.0, 0.0, 0.0, 0.0, 0.0, std::numeric_limits<float>::lowest()}};
    for (int i = 0; i < batch_beam_size; i++) {
        for (int j = 0; j < vocab_size; j++) {
            ASSERT_EQ(scores.GetScores(i)[j], expected_scores[i][j]);
        }
    }
}

}  // namespace test
}  // namespace onnxruntime
