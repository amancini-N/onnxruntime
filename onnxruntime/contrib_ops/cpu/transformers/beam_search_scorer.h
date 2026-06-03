// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// The implementation is based on huggingface transformers generation_beam_search.py

#pragma once
#include <queue>
#include <math.h>
#include "core/common/common.h"
#include "core/framework/allocator.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/cpu/containers.h"
#include "contrib_ops/cpu/transformers/sequences.h"
#include "contrib_ops/cpu/transformers/generation_shared.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {

struct HypothesisScore {
  gsl::span<const int32_t> hypothesis;
  gsl::span<const float> logprobs;  // Per-token log-probabilities aligned with `hypothesis`. Empty if not tracked.
  float score;
};

struct BeamHypotheses {
  // As these are constructed as an uninitialized array of memory, we need an Init method
  void Init(float length_penalty, gsl::span<HypothesisScore> beams);

  // Add a new hypothesis
  void Add(gsl::span<const int32_t>& hypothesis, float sum_logprobs);

  // Add a new hypothesis with per-token log-probabilities tracked alongside.
  void Add(gsl::span<const int32_t>& hypothesis,
           gsl::span<const float>& logprobs,
           float sum_logprobs);

  // Return true if this beats the worst score in the hypothesis
  bool CanImprove(float best_sum_logprobs, int current_length) const;

  // Output results
  template <typename T>
  void Output(int top_k,                        // number of sequences to return
              int max_length,                   // max sequence length
              gsl::span<int32_t>& sequences,    // buffer with pad token, shape (num_return_sequences, max_length)
              gsl::span<T>& sequences_scores);  // buffer for sequence scores, with shape (num_return_sequences)

  // Output results with per-token log-probabilities. `logprobs_out` has shape
  // (num_return_sequences, max_length) and is zero-filled by the caller for pad/unused positions.
  template <typename T>
  void Output(int top_k,
              int max_length,
              gsl::span<int32_t>& sequences,
              gsl::span<T>& sequences_scores,
              gsl::span<float>& logprobs_out);

  gsl::span<HypothesisScore> beams_;  // Beam width sized array of hypotheses, sorted by highest scoring
  int beams_used_;                    // Number of elements used in beams_
  float length_penalty_;
  bool done_;
};

struct BeamSearchScorer : IBeamScorer {
  BeamSearchScorer(const IGenerationParameters& parameters,
                   AllocatorPtr& allocator);

  void Process(ISequences& sequences,
               gsl::span<const float>& next_scores,
               gsl::span<const int32_t>& next_tokens,
               gsl::span<const int32_t>& next_indices) override;

  void Process(ISequences& sequences,
               gsl::span<const float>& next_scores,
               gsl::span<const int32_t>& next_tokens,
               gsl::span<const int32_t>& next_indices,
               gsl::span<const float>& next_token_logprobs) override;

  void Finalize(ISequences& sequences,
                gsl::span<const float>& final_beam_scores,
                Tensor* output_sequences,
                Tensor* output_sequence_scores) override;

  void OutputScores(gsl::span<const float>& final_scores, Tensor* output_scores) override;

  void FinalizeTokenLogprobs(ISequences& sequences,
                             Tensor* output_chosen_logprobs) override;

  bool IsDone() const override { return not_done_count_ == 0; }

  gsl::span<float> GetNextScores() override { return next_beam_scores_; }
  gsl::span<int32_t> GetNextTokens() override { return next_beam_tokens_; }
  gsl::span<int32_t> GetNextIndicesCPU() override { return next_beam_indices_; }

  size_t batch_size_;
  size_t num_beams_;
  size_t max_length_;
  size_t num_return_sequences_;
  int pad_token_id_;
  int eos_token_id_;
  bool early_stopping_;
  int not_done_count_;  // When zero, every batch entry is done (starts at batch_size_)

  IAllocatorUniquePtr<float> next_beam_scores_ptr_;
  gsl::span<float> next_beam_scores_;

  IAllocatorUniquePtr<int32_t> next_beam_tokens_ptr_;
  gsl::span<int32_t> next_beam_tokens_;

  IAllocatorUniquePtr<int32_t> next_beam_indices_ptr_;
  gsl::span<int32_t> next_beam_indices_;

  IAllocatorUniquePtr<int32_t> hypothesis_buffer_ptr_;  // Allocated buffer to hold all hypotheses
  gsl::span<int32_t> hypothesis_buffer_;                // Span of the allocated buffer
  int hypothesis_buffer_used_{};                        // Offset of available buffer, or length of used buffer.

  IAllocatorUniquePtr<HypothesisScore> hypothesis_scores_ptr_;  // num_beams_ * batch_size_, divided into num_beams_ chunks per BeamHypothesis in beam_hyps_
  IAllocatorUniquePtr<BeamHypotheses> beam_hyps_ptr_;
  gsl::span<BeamHypotheses> beam_hyps_;  // Shape is batch_size_

  // --- Per-token log-probability tracking (allocated lazily on first use) ---
  // Rotating per-beam logprob history, parallel to Sequences::sequences[2].
  // Each buffer has shape (batch_beam_size, max_length).
  IAllocatorUniquePtr<float> logprobs_history_buffer_ptr_;
  gsl::span<float> logprobs_history_[2];
  int current_logprobs_buffer_{0};
  // Per-surviving-beam log-probability of the token just chosen (parallel to next_beam_tokens_).
  IAllocatorUniquePtr<float> next_beam_token_logprobs_ptr_;
  gsl::span<float> next_beam_token_logprobs_;
  // Parallel buffer to hypothesis_buffer_, storing per-token logprobs of completed hypotheses.
  IAllocatorUniquePtr<float> hypothesis_logprobs_buffer_ptr_;
  gsl::span<float> hypothesis_logprobs_buffer_;
  bool logprobs_tracking_enabled_{false};
};

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
