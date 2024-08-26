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

TEST(JeroenLogitTest, JeroenLogitSeedTest) {  // Just here to verify tests are discovered, not a real test
  ASSERT_EQ(8211, 8211);
}  // keeping at end because seems tests output gets trunated sometimes


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
    onnxruntime::contrib::transformers::ILogitsProcessor<T>* logit_processor
) {
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


TEST(MinLengthLogitsProcessor, MinLengthNotReached) {
    int min_length = 3;  // bigger than sequences length
    int eos_token_id = 0;
    std::vector<float> next_token_scores_vector = {0.5, 0.2, 0.3};
    int batch_beam_size = 1;
    int vocab_size = 3;

    std::vector<std::vector<int32_t>> token_vectors = {};
    gsl::span<float> next_token_scores_span(gsl::make_span(
        next_token_scores_vector
        ));
    onnxruntime::contrib::transformers::NextTokenScores<float> next_token_scores({
        next_token_scores_span, batch_beam_size, vocab_size});

    onnxruntime::contrib::transformers::MinLengthLogitsProcessor<float> min_length_logit_processor(min_length,
                                                                                             eos_token_id);
    callLogitProcessor(
        batch_beam_size,
        next_token_scores,
        token_vectors,
        &min_length_logit_processor
        );

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

    std::vector<int32_t> first_token_vector = {1};
    std::vector<int32_t> second_token_vector = {2};
    std::vector<std::vector<int32_t>> token_vectors = {first_token_vector, second_token_vector};

    gsl::span<float> next_token_scores_span(gsl::make_span(
        next_token_scores_vector
        ));
    onnxruntime::contrib::transformers::NextTokenScores<float> next_token_scores({
        next_token_scores_span, batch_beam_size, vocab_size});

    onnxruntime::contrib::transformers::MinLengthLogitsProcessor<float> min_length_logit_processor(min_length,
                                                                                             eos_token_id);
    callLogitProcessor(
        batch_beam_size,
        next_token_scores,
        token_vectors,
        &min_length_logit_processor
        );

    ASSERT_NEAR(next_token_scores.GetScores(0)[0], 0.1, 0.0001);
}


TEST(MaxLengthLogitsProcessor, MaxLengthReached) {
    int max_length = 1;  // bigger than sequences length
    int eos_token_id = 0;
    std::vector<float> next_token_scores_vector = {0.1, 0.2, 0.3};
    int batch_beam_size = 1;
    int vocab_size = 3;

    std::vector<int32_t> first_token_vector = {1};
    std::vector<int32_t> second_token_vector = {2};
    std::vector<std::vector<int32_t>> token_vectors = {first_token_vector, second_token_vector};

    gsl::span<float> next_token_scores_span(gsl::make_span(
        next_token_scores_vector
        ));
    onnxruntime::contrib::transformers::NextTokenScores<float> next_token_scores({
        next_token_scores_span, batch_beam_size, vocab_size});

    onnxruntime::contrib::transformers::MaxLengthLogitsProcessor<float> max_length_logit_processor(max_length,
                                                                                             eos_token_id);
    callLogitProcessor(
        batch_beam_size,
        next_token_scores,
        token_vectors,
        &max_length_logit_processor
        );

    ASSERT_EQ(next_token_scores.GetScores(0)[1], std::numeric_limits<float>::lowest());
    ASSERT_EQ(next_token_scores.GetScores(0)[2], std::numeric_limits<float>::lowest());
}


TEST(MaxLengthLogitsProcessor, MaxLengthNotReached) {
    int max_length = 10;  // bigger than sequences length
    int eos_token_id = 0;
    std::vector<float> next_token_scores_vector = {0.1, 0.2, 0.3};
    int batch_beam_size = 1;
    int vocab_size = 3;

    std::vector<int32_t> first_token_vector = {1};
    std::vector<int32_t> second_token_vector = {2};
    std::vector<std::vector<int32_t>> token_vectors = {first_token_vector, second_token_vector};

    gsl::span<float> next_token_scores_span(gsl::make_span(
        next_token_scores_vector
        ));
    onnxruntime::contrib::transformers::NextTokenScores<float> next_token_scores({
        next_token_scores_span, batch_beam_size, vocab_size});

    onnxruntime::contrib::transformers::MaxLengthLogitsProcessor<float> max_length_logit_processor(max_length,
                                                                                             eos_token_id);
    callLogitProcessor(
        batch_beam_size,
        next_token_scores,
        token_vectors,
        &max_length_logit_processor
        );

    ASSERT_NEAR(next_token_scores.GetScores(0)[0], 0.1, 0.0001);
}

template <typename T>
void SequentialConstraintsFSALogitsTester(
    int batch_beam_size,
    onnxruntime::contrib::transformers::NextTokenScores<float> next_token_scores,
    std::vector<std::vector<int32_t>> token_vectors,
    onnxruntime::contrib::transformers::ILogitsProcessor<T>* logit_processor
) {
    int max_sequence_length = 100;
    // buffer is filled with 0 and has size batch_beam_size * max_length * 2
    std::vector<int32_t> buffer_v = std::vector<int32_t>(batch_beam_size * max_sequence_length * 2, 0);
    gsl::span<int32_t> buffer_s(buffer_v);

    // create ISequences object
    int start_sequence_length = 0;
    onnxruntime::contrib::transformers::Sequences sequences;
    sequences.Init(buffer_s, batch_beam_size, start_sequence_length, max_sequence_length);
    // adding all token vectors
    for (auto token_vector : token_vectors) {
        gsl::span<int> token_span(token_vector);
        sequences.AppendNextTokenToSequences(token_span);
    }
    onnxruntime::contrib::transformers::ISequences* sequences_pointer = &sequences;


    logit_processor->Process(sequences_pointer, next_token_scores);
}


void SequentialConstraintsTestRunner(
    int batch_beam_size,
    int vocab_size,
    int max_grammar_rule_length,
    std::vector<std::vector<int32_t>> grammar,
    std::vector<int32_t> constraints,
    std::vector<std::vector<int32_t>> token_sequence_vectors,
    std::vector<std::vector<float>> next_scores_vector
) {
    //processing vectors into appropriate objects
    std::vector<float> flattened_next_scores_vector;
    for (auto& scores : next_scores_vector) {
        for (auto& score : scores) {
            flattened_next_scores_vector.push_back(score);
        }
    }
    gsl::span<float> next_scores_span(gsl::make_span(
        flattened_next_scores_vector
        ));
    onnxruntime::contrib::transformers::NextTokenScores<float> next_token_scores({
        next_scores_span,
        batch_beam_size,
        vocab_size
        });

    assert(static_cast<int>(grammar.size()) == vocab_size);

    // create spans for grammar and cosntraints
    gsl::span<int32_t> constraints_span(gsl::make_span(
        constraints
        ));
    // createa flat int32_t vector for grammar from grammer
    std::vector<int32_t> flattened_grammar;
    for (auto& rule : grammar) {
        assert(static_cast<int>(rule.size()) == max_grammar_rule_length);
        for (auto& token : rule) {
            flattened_grammar.push_back(token);
        }
    }

    // create a span for the flattened grammar
    gsl::span<int32_t> grammar_span(gsl::make_span(
        flattened_grammar
        ));


    onnxruntime::contrib::transformers::SequentialConstraintsFSALogitsProcessor<float> fsa_logit_processor(
        constraints_span,
        grammar_span,
        batch_beam_size,
        max_grammar_rule_length,
        vocab_size
        );
    callLogitProcessor(
        batch_beam_size,
        next_token_scores,
        token_sequence_vectors,
        &fsa_logit_processor
        );
}


TEST(SequentialConstraintsFSALogitsProcessor, OneBeamNoConstraintYet) {
    int batch_beam_size = 1;
    int vocab_size = 3;
    int max_grammar_rule_length=1;

    std::vector<std::vector<int32_t>> grammar = {{-2}, {-2}, {-2}};
    std::vector<int32_t> constraints = {1, 2};

    std::vector<std::vector<int32_t>> token_sequence_vectors = {{0}};
    std::vector<std::vector<float>> next_scores_vector = {{0.1, 0.2, 0.3}};

    SequentialConstraintsTestRunner(
        batch_beam_size,
        vocab_size,
        max_grammar_rule_length,
        grammar,
        constraints,
        token_sequence_vectors,
        next_scores_vector
        );
    // only second score changed
    ASSERT_NEAR(next_scores_vector[0][0], 0.1, 0.0001);
    ASSERT_NEAR(next_scores_vector[0][1], 0.2, 0.0001);
    ASSERT_EQ(next_scores_vector[0][2], std::numeric_limits<float>::lowest());
}

// TEST(SequentialConstraintsFSALogitsProcessor, OneBeamOnlyAllRules) {
//     int batch_beam_size = 1;
//     int vocab_size = 3;
//     int max_grammar_rule_length=1;

//     std::vector<std::vector<int32_t>> grammar = {{-2}, {-2}, {-2}};
//     std::vector<int32_t> constraints = {1, 2};

//     std::vector<std::vector<int32_t>> token_sequence_vectors = {{1}};
//     std::vector<std::vector<float>> next_scores_vector = {{0.1, 0.2, 0.3}};

//     SequentialConstraintsTestRunner(
//         batch_beam_size,
//         vocab_size,
//         max_grammar_rule_length,
//         grammar,
//         constraints,
//         token_sequence_vectors,
//         next_scores_vector
//         );
//     // only second score changed
//     ASSERT_NEAR(next_scores_vector[0][0], 0.1, 0.0001);
//     ASSERT_EQ(next_scores_vector[0][1], std::numeric_limits<float>::lowest());
//     ASSERT_NEAR(next_scores_vector[0][2], 0.3, 0.0001);
// }

// TEST(SequentialConstraintsFSALogitsProcessor, FinishedConstraints) {
//     int batch_beam_size = 1;
//     int vocab_size = 4;
//     int max_grammar_rule_length=1;

//     std::vector<std::vector<int32_t>> grammar = {{-2}, {-2}, {-2}, {-2}};
//     std::vector<int32_t> constraints = {1, 2};

//     std::vector<std::vector<int32_t>> token_sequence_vectors = {{1, 2}};
//     std::vector<std::vector<float>> next_scores_vector = {{0.1, 0.2, 0.3, 0.4}};

//     SequentialConstraintsTestRunner(
//         batch_beam_size,
//         vocab_size,
//         max_grammar_rule_length,
//         grammar,
//         constraints,
//         token_sequence_vectors,
//         next_scores_vector
//         );
//     // only second score changed
//     ASSERT_NEAR(next_scores_vector[0][0], 0.1, 0.0001);
//     ASSERT_EQ(next_scores_vector[0][1], std::numeric_limits<float>::lowest());
//     ASSERT_EQ(next_scores_vector[0][2], std::numeric_limits<float>::lowest());
//     ASSERT_NEAR(next_scores_vector[0][3], 0.4, 0.0001);
// }

// TEST(SequentialConstraintsFSALogitsProcessor, SpecificConstraints) {
//     int batch_beam_size = 1;
//     int vocab_size = 4;
//     int max_grammar_rule_length=1;

//     std::vector<std::vector<int32_t>> grammar = {{3}, {3}, {3}, {3}};
//     std::vector<int32_t> constraints = {1, 2};

//     std::vector<std::vector<int32_t>> token_sequence_vectors = {{1}};
//     std::vector<std::vector<float>> next_scores_vector = {{0.1, 0.2, 0.3, 0.4}};

//     SequentialConstraintsTestRunner(
//         batch_beam_size,
//         vocab_size,
//         max_grammar_rule_length,
//         grammar,
//         constraints,
//         token_sequence_vectors,
//         next_scores_vector
//         );
//     // only second score changed
//     ASSERT_EQ(next_scores_vector[0][0], std::numeric_limits<float>::lowest());  // not allowed due to not ANY grammer rule (-2)
//     ASSERT_EQ(next_scores_vector[0][1], std::numeric_limits<float>::lowest());  // not allowed due to constraint already reached
//     ASSERT_NEAR(next_scores_vector[0][2], 0.3, 0.0001);   // remains same because next constraint
//     ASSERT_NEAR(next_scores_vector[0][3], 0.4, 0.0001);   // ecplicitly allowed
// }

TEST(SequentialConstraintsFSALogitsProcessor, JeroenSeedTest) {  // Just here to verify tests are discovered, not a real test
  ASSERT_EQ(8211, 8211);
}  // keeping at end because seems tests output gets trunated sometimes


}
}
