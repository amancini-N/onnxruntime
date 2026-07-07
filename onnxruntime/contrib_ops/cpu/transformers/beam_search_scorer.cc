// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <queue>
#include <math.h>
#include "core/common/common.h"
#include "core/common/narrow.h"
#include "core/common/safeint.h"
#include "core/common/span_utils.h"
#include "core/framework/allocator.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/cpu/rnn/rnn_helpers.h"
#include "contrib_ops/cpu/transformers/beam_search_scorer.h"

namespace onnxruntime {
namespace contrib {
namespace transformers {
using ::onnxruntime::rnn::detail::Allocate;

void BeamHypotheses::Init(float length_penalty, gsl::span<HypothesisScore> beams) {
  beams_ = beams;
  beams_used_ = 0;
  length_penalty_ = length_penalty;
  done_ = false;
}

void BeamHypotheses::Add(gsl::span<const int32_t>& hypothesis, float sum_logprobs) {
  auto length = hypothesis.size();
  float score = sum_logprobs / pow(static_cast<float>(length), length_penalty_);

  size_t index = beams_used_;
  // If the array is full, don't add unless it's better than the worst element
  if (index == beams_.size()) {
    if (score <= beams_[--index].score)
      return;
  } else
    beams_used_++;

  // Rotate existing elements over while the new element scores higher
  for (; index > 0 && score > beams_[index - 1].score; index--)
    beams_[index] = beams_[index - 1];

  beams_[index] = HypothesisScore{hypothesis, {}, score};
}

void BeamHypotheses::Add(gsl::span<const int32_t>& hypothesis,
                         gsl::span<const float>& logprobs,
                         float sum_logprobs) {
  auto length = hypothesis.size();
  float score = sum_logprobs / pow(static_cast<float>(length), length_penalty_);

  size_t index = beams_used_;
  if (index == beams_.size()) {
    if (score <= beams_[--index].score)
      return;
  } else
    beams_used_++;

  for (; index > 0 && score > beams_[index - 1].score; index--)
    beams_[index] = beams_[index - 1];

  beams_[index] = HypothesisScore{hypothesis, logprobs, score};
}

bool BeamHypotheses::CanImprove(float best_sum_logprobs, int current_length) const {
  float current_score = best_sum_logprobs / pow(static_cast<float>(current_length), length_penalty_);
  return beams_.back().score < current_score;
}

template <typename T>
void BeamHypotheses::Output(
    int top_k,
    int max_length,
    gsl::span<int32_t>& sequences,   // buffer filled with pad token ID, shape (num_return_sequences, max_length)
    gsl::span<T>& sequences_scores)  // buffer of shape (num_return_sequences) or empty
{
  // Copy the top_k beams into the sequences
  ORT_ENFORCE(top_k <= beams_used_);
  for (int index = 0; index < top_k; index++) {
    auto& item = beams_[index];
    gsl::span<int32_t> target = sequences.subspan(static_cast<gsl::index>(index) * max_length, max_length);

    // Note that word_ids might be less than max_length.
    // Since the sequences has been filled with pad token ID, so padding is not needed here.
    gsl::copy(item.hypothesis, target);

    if (!sequences_scores.empty())
      sequences_scores[index] = (T)item.score;
  }
}

template <typename T>
void BeamHypotheses::Output(
    int top_k,
    int max_length,
    gsl::span<int32_t>& sequences,
    gsl::span<T>& sequences_scores,
    gsl::span<float>& logprobs_out)  // (num_return_sequences, max_length), pre-zeroed by caller
{
  ORT_ENFORCE(top_k <= beams_used_);
  for (int index = 0; index < top_k; index++) {
    auto& item = beams_[index];
    gsl::span<int32_t> target = sequences.subspan(static_cast<gsl::index>(index) * max_length, max_length);
    gsl::copy(item.hypothesis, target);

    if (!sequences_scores.empty())
      sequences_scores[index] = (T)item.score;

    if (!logprobs_out.empty() && !item.logprobs.empty()) {
      gsl::span<float> lp_target = logprobs_out.subspan(
          static_cast<gsl::index>(index) * max_length, item.logprobs.size());
      gsl::copy(item.logprobs, lp_target);
    }
  }
}

BeamSearchScorer::BeamSearchScorer(const IGenerationParameters& parameters,
                                   AllocatorPtr& allocator)
    : batch_size_{static_cast<size_t>(parameters.batch_size)},
      num_beams_{static_cast<size_t>(parameters.num_beams)},
      max_length_{static_cast<size_t>(parameters.max_length)},
      num_return_sequences_{static_cast<size_t>(parameters.num_return_sequences)},
      pad_token_id_{parameters.pad_token_id},
      eos_token_id_{parameters.eos_token_id},
      early_stopping_{parameters.early_stopping},
      not_done_count_{parameters.batch_size} {
  size_t batch_beam_size = batch_size_ * num_beams_;

  auto beams = Allocate<HypothesisScore>(allocator, batch_beam_size, hypothesis_scores_ptr_);
  beam_hyps_ = Allocate<BeamHypotheses>(allocator, batch_size_, beam_hyps_ptr_);
  for (size_t i = 0; i < batch_size_; i++)
    beam_hyps_[i].Init(parameters.length_penalty, beams.subspan(i * num_beams_, num_beams_));

  next_beam_scores_ = Allocate<float>(allocator, batch_beam_size, next_beam_scores_ptr_);
  next_beam_tokens_ = Allocate<int32_t>(allocator, batch_beam_size, next_beam_tokens_ptr_);
  next_beam_indices_ = Allocate<int32_t>(allocator, batch_beam_size, next_beam_indices_ptr_);

  // Space to store intermediate sequence with length sequence_length, sequence_length + 1, ..., max_sequence_length.
  size_t per_beam = (SafeInt<size_t>(max_length_) * (max_length_ + 1) - (parameters.sequence_length - 1) * parameters.sequence_length) / 2;
  hypothesis_buffer_ = Allocate<int32_t>(allocator, batch_beam_size * per_beam, hypothesis_buffer_ptr_);

  // Per-token log-probability tracking. Allocated unconditionally; output is optional and is only
  // populated if the consumer requests output 3 (`chosen_token_logprobs`).
  size_t history_size = batch_beam_size * max_length_;
  auto history_buffer = Allocate<float>(allocator, 2 * history_size, logprobs_history_buffer_ptr_);
  logprobs_history_[0] = history_buffer.subspan(0, history_size);
  logprobs_history_[1] = history_buffer.subspan(history_size);
  std::fill_n(logprobs_history_[0].data(), logprobs_history_[0].size(), 0.0f);
  std::fill_n(logprobs_history_[1].data(), logprobs_history_[1].size(), 0.0f);
  next_beam_token_logprobs_ = Allocate<float>(allocator, batch_beam_size, next_beam_token_logprobs_ptr_);
  std::fill_n(next_beam_token_logprobs_.data(), next_beam_token_logprobs_.size(), 0.0f);
  hypothesis_logprobs_buffer_ = Allocate<float>(allocator, batch_beam_size * per_beam, hypothesis_logprobs_buffer_ptr_);
}

void BeamSearchScorer::Process(ISequences& sequences,
                               gsl::span<const float>& next_scores,
                               gsl::span<const int32_t>& next_tokens,
                               gsl::span<const int32_t>& next_indices) {
  // Sequences shape is (batch_size * num_beams, total_sequence_length)
  // It contains word ID of whole sequence generated so far.
  // It is different from subgraph input_ids, which only need one word when past state is not empty.

  const int sequence_length = sequences.GetSequenceLength();

  ORT_ENFORCE(next_scores.size() == next_tokens.size());
  ORT_ENFORCE(next_scores.size() == next_indices.size());

  for (size_t batch = 0; batch < batch_size_; batch++) {
    BeamHypotheses& beam_hyp = beam_hyps_[batch];
    if (beam_hyp.done_) {
      ORT_ENFORCE(beam_hyp.beams_used_ == gsl::narrow_cast<int>(num_beams_),
                  "Batch can only be done if all beams have been generated");

      // Pad the batch.
      for (size_t j = 0; j < num_beams_; j++) {
        next_beam_scores_[batch * num_beams_ + j] = 0.0f;
        next_beam_tokens_[batch * num_beams_ + j] = pad_token_id_;
        next_beam_indices_[batch * num_beams_ + j] = 0;
      }
      continue;
    }

    // Next tokens for this sentence.
    size_t beam_idx = 0;
    size_t top_k = 2 * num_beams_;
    for (size_t j = 0; j < top_k; j++) {
      int32_t next_token = next_tokens[batch * top_k + j];
      float next_score = next_scores[batch * top_k + j];
      int32_t next_index = next_indices[batch * top_k + j];

      int batch_beam_idx = static_cast<int>(batch * num_beams_) + next_index;
      // Add to generated hypotheses if end of sentence.
      if ((eos_token_id_ >= 0) && (next_token == eos_token_id_)) {
        bool is_beam_token_worse_than_top_num_beams = (j >= num_beams_);
        if (is_beam_token_worse_than_top_num_beams) {
          continue;
        }

        // Clone the sequence and append to buffer.
        gsl::span<const int32_t> src = sequences.GetSequence(batch_beam_idx);
        auto clone = hypothesis_buffer_.subspan(static_cast<size_t>(hypothesis_buffer_used_), sequence_length);

        gsl::copy(src, clone);
        hypothesis_buffer_used_ += sequence_length;
        auto sequence = ReinterpretAsSpan<const int32_t>(clone);
        beam_hyp.Add(sequence, next_score);
      } else {
        // Add next predicted token since it is not eos_token.
        next_beam_scores_[batch * num_beams_ + beam_idx] = next_score;
        next_beam_tokens_[batch * num_beams_ + beam_idx] = next_token;
        next_beam_indices_[batch * num_beams_ + beam_idx] = batch_beam_idx;
        ++beam_idx;
      }

      // Once the beam for next step is full, don't add more tokens to it.
      if (beam_idx == num_beams_)
        break;
    }

    ORT_ENFORCE(beam_idx == num_beams_);
    ORT_ENFORCE(static_cast<size_t>(hypothesis_buffer_used_) <= hypothesis_buffer_.size());

    //  Check if we are done so that we can save a pad step if all(done)
    if (static_cast<size_t>(beam_hyp.beams_used_) < num_beams_)
      continue;

    if (!early_stopping_) {
      gsl::span<const float> topk_scores = next_scores.subspan(batch * num_beams_, top_k);
      const auto best_sum_logprobs = std::max_element(topk_scores.begin(), topk_scores.end());
      if (beam_hyp.CanImprove(*best_sum_logprobs, sequence_length))
        continue;
    }

    beam_hyp.done_ = true;
    not_done_count_--;
  }
}

// Overload that also records per-token log-probabilities of every candidate token in
// `next_token_logprobs` (shape batch_size * 2 * num_beams). Mirrors the 4-arg Process body
// and additionally:
//   * EOS path: clones the parent beam's running logprob history into `hypothesis_logprobs_buffer_`,
//     appends the EOS token's logprob, and stores the resulting span on the new HypothesisScore.
//   * Surviving-beam path: records the chosen token's logprob in `next_beam_token_logprobs_`.
// After the per-batch loop, rotates `logprobs_history_` using the just-written `next_beam_indices_`
// (same reordering pattern as `Sequences::AppendNextTokenToSequences`).
void BeamSearchScorer::Process(ISequences& sequences,
                               gsl::span<const float>& next_scores,
                               gsl::span<const int32_t>& next_tokens,
                               gsl::span<const int32_t>& next_indices,
                               gsl::span<const float>& next_token_logprobs) {
  const int sequence_length = sequences.GetSequenceLength();

  ORT_ENFORCE(next_scores.size() == next_tokens.size());
  ORT_ENFORCE(next_scores.size() == next_indices.size());
  ORT_ENFORCE(next_scores.size() == next_token_logprobs.size());

  logprobs_tracking_enabled_ = true;

  const size_t batch_beam_size = batch_size_ * num_beams_;
  const size_t top_k = 2 * num_beams_;
  const size_t max_length = max_length_;

  gsl::span<const float> in_history = logprobs_history_[current_logprobs_buffer_];

  for (size_t batch = 0; batch < batch_size_; batch++) {
    BeamHypotheses& beam_hyp = beam_hyps_[batch];
    if (beam_hyp.done_) {
      for (size_t j = 0; j < num_beams_; j++) {
        next_beam_scores_[batch * num_beams_ + j] = 0.0f;
        next_beam_tokens_[batch * num_beams_ + j] = pad_token_id_;
        next_beam_indices_[batch * num_beams_ + j] = 0;
        next_beam_token_logprobs_[batch * num_beams_ + j] = 0.0f;
      }
      continue;
    }

    size_t beam_idx = 0;
    for (size_t j = 0; j < top_k; j++) {
      int32_t next_token = next_tokens[batch * top_k + j];
      float next_score = next_scores[batch * top_k + j];
      int32_t next_index = next_indices[batch * top_k + j];
      float next_token_lp = next_token_logprobs[batch * top_k + j];

      int batch_beam_idx = static_cast<int>(batch * num_beams_) + next_index;
      if ((eos_token_id_ >= 0) && (next_token == eos_token_id_)) {
        bool is_beam_token_worse_than_top_num_beams = (j >= num_beams_);
        if (is_beam_token_worse_than_top_num_beams) {
          continue;
        }

        // Clone the sequence and per-token logprob history of the parent beam, then add to buffers.
        gsl::span<const int32_t> src_tokens = sequences.GetSequence(batch_beam_idx);
        auto clone_tokens = hypothesis_buffer_.subspan(static_cast<size_t>(hypothesis_buffer_used_), sequence_length);
        gsl::copy(src_tokens, clone_tokens);

        auto src_lps = in_history.subspan(static_cast<size_t>(batch_beam_idx) * max_length, sequence_length);
        auto clone_lps = hypothesis_logprobs_buffer_.subspan(static_cast<size_t>(hypothesis_buffer_used_), sequence_length);
        gsl::copy(src_lps, clone_lps);
        // The hypothesis-stored logprobs match the returned sequence (which does not include the
        // EOS token itself); the EOS token's logprob is implicit in `next_score` and is not stored.

        hypothesis_buffer_used_ += sequence_length;
        auto sequence = ReinterpretAsSpan<const int32_t>(clone_tokens);
        gsl::span<const float> lp_span(clone_lps.data(), clone_lps.size());
        beam_hyp.Add(sequence, lp_span, next_score);
        (void)next_token_lp;  // EOS lp folded into next_score
      } else {
        next_beam_scores_[batch * num_beams_ + beam_idx] = next_score;
        next_beam_tokens_[batch * num_beams_ + beam_idx] = next_token;
        next_beam_indices_[batch * num_beams_ + beam_idx] = batch_beam_idx;
        next_beam_token_logprobs_[batch * num_beams_ + beam_idx] = next_token_lp;
        ++beam_idx;
      }

      if (beam_idx == num_beams_)
        break;
    }

    ORT_ENFORCE(beam_idx == num_beams_);
    ORT_ENFORCE(static_cast<size_t>(hypothesis_buffer_used_) <= hypothesis_buffer_.size());

    if (static_cast<size_t>(beam_hyp.beams_used_) < num_beams_)
      continue;

    if (!early_stopping_) {
      gsl::span<const float> topk_scores = next_scores.subspan(batch * num_beams_, top_k);
      const auto best_sum_logprobs = std::max_element(topk_scores.begin(), topk_scores.end());
      if (beam_hyp.CanImprove(*best_sum_logprobs, sequence_length))
        continue;
    }

    beam_hyp.done_ = true;
    not_done_count_--;
  }

  // Rotate the per-token logprob history in lockstep with how `Sequences` will reorder
  // its token buffers in `AppendNextTokenToSequences` (which uses the same `next_beam_indices_`).
  gsl::span<float> out_history = logprobs_history_[current_logprobs_buffer_ ^ 1];
  for (size_t i = 0; i < batch_beam_size; i++) {
    const int parent_slot = next_beam_indices_[i];
    auto src_lps = in_history.subspan(static_cast<size_t>(parent_slot) * max_length, sequence_length);
    auto dst_lps = out_history.subspan(i * max_length, sequence_length);
    gsl::copy(src_lps, dst_lps);
    out_history[i * max_length + sequence_length] = next_beam_token_logprobs_[i];
  }
  current_logprobs_buffer_ ^= 1;
}

template <typename T>
void OutputSequenceScores(BeamSearchScorer* scorer,
                          ISequences& sequences,
                          gsl::span<const float>& final_beam_scores,
                          Tensor* output_sequences,
                          Tensor* output_sequence_scores) {
  // Finalize all open beam hypotheses and add to generated hypotheses.
  for (size_t batch_index = 0; batch_index < scorer->batch_size_; batch_index++) {
    BeamHypotheses& beam_hyp = scorer->beam_hyps_[batch_index];
    if (beam_hyp.done_) {
      continue;
    }

    for (size_t beam_index = 0; beam_index < scorer->num_beams_; beam_index++) {
      size_t batch_beam_index = batch_index * scorer->num_beams_ + beam_index;
      float final_score = final_beam_scores[batch_beam_index];
      auto final_tokens = sequences.GetSequence(narrow<int>(batch_beam_index));
      if (scorer->logprobs_tracking_enabled_) {
        int seq_len = sequences.GetSequenceLength();
        auto lps_mut = scorer->logprobs_history_[scorer->current_logprobs_buffer_].subspan(
            batch_beam_index * scorer->max_length_, seq_len);
        gsl::span<const float> lps_const(lps_mut.data(), lps_mut.size());
        beam_hyp.Add(final_tokens, lps_const, final_score);
      } else {
        beam_hyp.Add(final_tokens, final_score);
      }
    }
  }

  // Word IDs of each sequence, with shape (batch_size * num_return_sequences, max_sequence_length).
  gsl::span<int32_t> output = output_sequences->MutableDataAsSpan<int32_t>();

  // Fill output sequences with pad token ID so that we do not need append it later.
  std::fill_n(output.data(), output.size(), scorer->pad_token_id_);

  // Score of each sequence, with shape (batch_size * num_return_sequences).
  gsl::span<T> sequence_scores;
  if (output_sequence_scores) {
    sequence_scores = output_sequence_scores->MutableDataAsSpan<T>();
  }

  // Select the best hypotheses according to number of sequences to return.
  for (size_t batch_index = 0; batch_index < scorer->batch_size_; batch_index++) {
    BeamHypotheses& beam_hyp = scorer->beam_hyps_[batch_index];

    auto batch_output = output.subspan(batch_index * scorer->num_return_sequences_ * scorer->max_length_,
                                       scorer->num_return_sequences_ * scorer->max_length_);
    gsl::span<T> sequence_scores_buffer;
    if (!sequence_scores.empty())
      sequence_scores_buffer = sequence_scores.subspan(batch_index * scorer->num_return_sequences_, scorer->num_return_sequences_);

    beam_hyp.template Output<T>(narrow<int>(scorer->num_return_sequences_), narrow<int>(scorer->max_length_), batch_output,
                                sequence_scores_buffer);
  }
}

void BeamSearchScorer::Finalize(ISequences& sequences,
                                gsl::span<const float>& final_beam_scores,
                                Tensor* output_sequences,
                                Tensor* output_sequence_scores) {
  ORT_ENFORCE(output_sequences != nullptr);

  if (output_sequence_scores == nullptr || output_sequence_scores->IsDataType<float>()) {
    OutputSequenceScores<float>(this, sequences, final_beam_scores, output_sequences, output_sequence_scores);
  } else {
    ORT_ENFORCE(output_sequence_scores->IsDataType<MLFloat16>());
    OutputSequenceScores<MLFloat16>(this, sequences, final_beam_scores, output_sequences, output_sequence_scores);
  }
}

void BeamSearchScorer::OutputScores(gsl::span<const float>& final_scores, Tensor* output_scores) {
  if (output_scores) {
    if (output_scores->IsDataType<float>()) {
      gsl::span<float> target = output_scores->MutableDataAsSpan<float>();
      ORT_ENFORCE(target.size() == final_scores.size());
      std::copy_n(final_scores.data(), final_scores.size(), target.data());
    } else {
      ORT_ENFORCE(output_scores->IsDataType<MLFloat16>());
      gsl::span<MLFloat16> target = output_scores->MutableDataAsSpan<MLFloat16>();
      ORT_ENFORCE(target.size() == final_scores.size());
      const float* src = final_scores.data();
      MLFloat16* dst = target.data();
      for (size_t i = 0; i < target.size(); i++) {
        dst[i] = MLFloat16(src[i]);
      }
    }
  }
}

void BeamSearchScorer::FinalizeTokenLogprobs(ISequences& /*sequences*/,
                                             Tensor* output_chosen_logprobs) {
  if (output_chosen_logprobs == nullptr) {
    return;
  }
  ORT_ENFORCE(output_chosen_logprobs->IsDataType<float>(),
              "chosen_token_logprobs output must be float32");

  gsl::span<float> output = output_chosen_logprobs->MutableDataAsSpan<float>();
  // Position 0 (decoder start token) and any unwritten / padded positions are zero.
  std::fill_n(output.data(), output.size(), 0.0f);

  if (!logprobs_tracking_enabled_) {
    return;
  }

  for (size_t batch_index = 0; batch_index < batch_size_; batch_index++) {
    BeamHypotheses& beam_hyp = beam_hyps_[batch_index];
    auto batch_output = output.subspan(batch_index * num_return_sequences_ * max_length_,
                                       num_return_sequences_ * max_length_);
    const int top_k = narrow<int>(num_return_sequences_);
    ORT_ENFORCE(top_k <= beam_hyp.beams_used_);
    for (int index = 0; index < top_k; index++) {
      const auto& item = beam_hyp.beams_[index];
      if (item.logprobs.empty()) {
        continue;  // Hypothesis was added without logprob tracking (legacy path); leave as zero.
      }
      auto lp_target = batch_output.subspan(static_cast<gsl::index>(index) * max_length_,
                                            item.logprobs.size());
      gsl::copy(item.logprobs, lp_target);
    }
  }
}

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
