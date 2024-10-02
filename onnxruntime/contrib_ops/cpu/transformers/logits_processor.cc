// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <assert.h>
#include "core/common/narrow.h"
#include "core/common/safeint.h"
#include "core/common/span_utils.h"
#include "contrib_ops/cpu/transformers/logits_processor.h"
#include <vector>
#include <numeric>
#include <algorithm>

namespace onnxruntime {
namespace contrib {
namespace transformers {

// Interface for all scorers for beam search or beam sample.
template <typename T>
MinLengthLogitsProcessor<T>::MinLengthLogitsProcessor(int min_length, int eos_token_id)
    : min_length_(min_length), eos_token_id_(eos_token_id) {}

// if we didn't reach yet minimum length
// set the eos token score to lowest for each beam
template <typename T>
void MinLengthLogitsProcessor<T>::Process(const ISequences* sequences,
                                          NextTokenScores<T>& next_token_scores) {
  if (sequences->GetSequenceLength() < min_length_) {
    next_token_scores.SetScore(eos_token_id_, std::numeric_limits<T>::lowest());
  }
}

template <typename T>
MaxLengthLogitsProcessor<T>::MaxLengthLogitsProcessor(int max_length, int eos_token_id)
    : max_length_(max_length), eos_token_id_(eos_token_id) {}

// We have to emit EOS on the last possible position.
// if we have reached the max length
// set scores for all tokens but eos to lowest for each beam
template <typename T>
void MaxLengthLogitsProcessor<T>::Process(const ISequences* sequences,
                                          NextTokenScores<T>& next_token_scores) {
  if (sequences->GetSequenceLength() >= max_length_ - 1) {
    for (int i = 0; i < next_token_scores.vocab_size; i++) {
      if (i != eos_token_id_) {
        next_token_scores.SetScore(i, std::numeric_limits<T>::lowest());
      }
    }
  }
}

template <typename T>
RepetitionPenaltyLogitsProcessor<T>::RepetitionPenaltyLogitsProcessor(float penalty) : penalty_(penalty) {
}

template <typename T>
void RepetitionPenaltyLogitsProcessor<T>::Process(const ISequences* sequences,
                                                  NextTokenScores<T>& next_token_scores) {
  const int batch_beam_size = next_token_scores.batch_beam_size;
  for (int i = 0; i < batch_beam_size; i++) {
    gsl::span<T> beam_token_scores = next_token_scores.GetScores(i);
    gsl::span<const int32_t> sequence = sequences->GetSequence(i);

    // Find unique word IDs in sequence.
    std::unordered_set<int32_t> unique_word_ids;
    for (const auto& word_id : sequence) {
      unique_word_ids.insert(word_id);
    }

    for (const int32_t word_id : unique_word_ids) {
      T score = beam_token_scores[word_id];

      // If score < 0, then repetition penalty > 1.0 has to multiplied to reduce the previous token probability,
      // This assumes that scores are either positive (like ctrl) or negative (like GPT-2), but not a mixture.
      beam_token_scores[word_id] = (score < 0 ? score * penalty_ : score / penalty_);
    }
  }
}

template <typename T>
NoRepeatNGramLogitsProcessor<T>::NoRepeatNGramLogitsProcessor(
    std::vector<int> ngram_size, std::vector<int> ngram_history_lengths, int ngram_format_mode,
    std::vector<int> ngram_format_tokens, std::vector<int> ngram_format_tokens_unique_sorted,
    int ngram_format_tokens_n_exclusions, std::vector<int> ngram_format_tokens_lengths,
    int ngram_format_tokens_max_length) :
      ngram_size_(ngram_size), history_lengths_(ngram_history_lengths), format_mode_(ngram_format_mode),
      format_tokens_(ngram_format_tokens), format_tokens_unique_sorted_(ngram_format_tokens_unique_sorted),
      format_tokens_num_exclusions_(ngram_format_tokens_n_exclusions),
      format_tokens_lengths_(ngram_format_tokens_lengths), format_tokens_max_length_(ngram_format_tokens_max_length) {
}

inline bool BinarySearch(const std::vector<int>& sorted_vector, int target) {
  int left = 0;
  int right = static_cast<int>(sorted_vector.size()) - 1;
  while (left <= right) {
    int mid = left + (right - left) / 2;
    if (sorted_vector[mid] == target) {
      return true;
    } else if (sorted_vector[mid] < target) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
  return false;
}

// Check if ngram is a format sequence, if so, ignore it.
// format mode is 0 for ALL method, 1 for ANY method, 2 for SIMPLE method
// In ALL method, we match a complete ngram against the format tokens.
// In ANY method, we match any token in the ngram against the format tokens.
// In SIMPLE method, we match the last token in the ngram against the format tokens.
template <typename T>
bool NoRepeatNGramLogitsProcessor<T>::CheckFormatNGram(int ngram_size, gsl::span<const int32_t> ngram) {
  ORT_UNUSED_PARAMETER(ngram_size);
  if (format_tokens_.empty()) {
    return false;
  }

  if (format_mode_ == 0) {
    for (int i = 0; i < static_cast<int>(ngram.size()); i++) {
      if (format_mode_ == 0) {
        if (!BinarySearch(format_tokens_unique_sorted_, ngram[i])) {
          return false;
        }
      }
    }
    return true;
  } else if (format_mode_ == 1 && format_tokens_max_length_ == 1) {
    for (int i = 0; i < static_cast<int>(ngram.size()); i++) {
      if (BinarySearch(format_tokens_unique_sorted_, ngram[i])) {
        return true;
      }
    }
    return false;
  } else if (format_mode_ == 1) {
    for (int i = 0; i < format_tokens_num_exclusions_; i++) {
      const int format_ngram_size = format_tokens_lengths_[i];
      auto format_tokens_as_span = AsSpan(format_tokens_);
      auto format_ngram = format_tokens_as_span.subspan(i * format_tokens_max_length_, format_ngram_size);
      // check presence of format_ngram in ngram and save it
      std::vector<bool> ngram_isin(ngram.size(), false);
      for (int j = 0; j < static_cast<int>(ngram.size()); j++) {
        if (std::find(format_ngram.begin(), format_ngram.end(), ngram[j]) != format_ngram.end()) {
          ngram_isin[j] = true;
        }
      }

      // convolute ngram_isin with ones vector of length format_ngram_size
      const int conv_out_size = static_cast<int>(ngram.size() - format_ngram_size + 1);
      std::vector<int> convoluted(conv_out_size, 0);
      const int m = static_cast<int>(format_ngram_size) / 2;
      for (int j = 0; j < conv_out_size; j++) {
        for (int k = 0; k < static_cast<int>(format_ngram_size); k++) {
          if (j - k + m >= 0) {
            convoluted[j] += ngram_isin[j - k + m];
          }
        }
      }

      // neglect elements in convoluted that are less than format_ngam.size()
      for (int j = 0; j < conv_out_size; j++) {
        if (convoluted[j] != format_ngram_size) {
          convoluted[j] = 0;
        }
      }

      // true if sum of convoluted is equal to format_ngram_size
      if (std::accumulate(convoluted.begin(), convoluted.end(), 0) == format_ngram_size) {
        return true;
      }
    }
  } else if (format_mode_ == 2) {
    if (std::find(format_tokens_.begin(), format_tokens_.end(), ngram.back()) != format_tokens_.end()) {
      return true;
    }
  }

  return false;
}

template <typename T>
void NoRepeatNGramLogitsProcessor<T>::Process(const ISequences* sequences,
                                              NextTokenScores<T>& next_token_scores) {
  int batch_beam_size = next_token_scores.batch_beam_size;
  int config_length = static_cast<int>(ngram_size_.size());
  for (int c = 0; c < config_length; c++) {
    int ngram = ngram_size_[c];
    if (ngram == 0 || ngram >= sequences->GetSequenceLength()) {
      continue;
    }
    int history_length = history_lengths_[c];

    gsl::index prefix_length = static_cast<gsl::index>(ngram) - 1;
    // We assume history_length is sorted
    if (c > 0) {
      int prefix_to_increase = history_lengths_[c - 1] - ngram + 1;
      prefix_length += prefix_to_increase;
    }
    const int seq_len = sequences->GetSequenceLength();
    if (seq_len > history_length && history_length > 0) {
      prefix_length += history_length - seq_len;
    }

    if (seq_len <= prefix_length) {
      continue;
    }

    for (int i = 0; i < batch_beam_size; i++) {
      gsl::span<T> beam_token_scores = next_token_scores.GetScores(i);
      gsl::span<const int32_t> sequence = sequences->GetSequence(i);
      if (seq_len > history_length && history_length > 0) {
        sequence = sequence.subspan(seq_len - history_length);
      }

      gsl::span<const int32_t> prefix = sequence.subspan(seq_len - prefix_length);
      ORT_ENFORCE(prefix.size() == narrow<size_t>(prefix_length));

      std::unordered_set<int32_t> blocked_word_ids;
      for (int j = 0; j <= static_cast<int>(sequence.size()) - ngram; j++) {
        // Here we use naive algorithm for matching. The complexity is O(batch_beam_size * ngram_size * sequence_length)
        // TODO(tianleiwu): build N-Gram index (hash table with prefix of length NGram - 1 as key,
        //                  and list of last word of NGram as value) for fast matching.
        if ((ngram == 1 || SpanEq(prefix, sequence.subspan(j, prefix_length))) && !CheckFormatNGram(ngram, sequence.subspan(j, ngram))) {
          blocked_word_ids.insert(sequence[static_cast<gsl::index>(j) + prefix_length]);
        }
      }

      for (const int32_t word_id : blocked_word_ids) {
        beam_token_scores[word_id] = std::numeric_limits<T>::lowest();
      }
    }

#ifdef DEBUG_GENERATION
    DumpScores("NoRepeatNGramLogitsProcessor", next_token_scores);
#endif
  }
}

template <typename T>
VocabMaskLogitsProcessor<T>::VocabMaskLogitsProcessor(const gsl::span<const int32_t>& vocab_mask)
    : vocab_mask_(vocab_mask) {
}

template <typename T>
void VocabMaskLogitsProcessor<T>::Process(const ISequences* /*sequences*/,
                                          NextTokenScores<T>& next_token_scores) {
  assert(!vocab_mask_.empty());

  // Process vocabulary mask and set tokens with mask value 0 to -inf.
  T* p = next_token_scores.scores.data();
  // next_token_scores shape (batch_size * num_beams, vocab_size)
  // vocab_mask shape (vocab_size).
  for (int i = 0; i < next_token_scores.batch_beam_size; i++) {
    for (int j = 0; j < next_token_scores.vocab_size; j++, p++) {
      if (vocab_mask_[j] == 0) {
        *p = std::numeric_limits<T>::lowest();
      }
    }
  }
}

template <typename T>
PrefixVocabMaskLogitsProcessor<T>::PrefixVocabMaskLogitsProcessor(const gsl::span<const int32_t>& prefix_vocab_mask,
                                                                  int batch_size)
    : prefix_vocab_mask_(prefix_vocab_mask),
      batch_size_(batch_size) {
}

template <typename T>
void PrefixVocabMaskLogitsProcessor<T>::Process(const ISequences* /*sequences*/,
                                                NextTokenScores<T>& next_token_scores) {
  assert(!prefix_vocab_mask_.empty());

  // next_token_scores shape (batch_size * num_beams, vocab_size)
  int num_beams = next_token_scores.batch_beam_size / batch_size_;
  assert(num_beams * batch_size_ == next_token_scores.batch_beam_size);

  // Process prefix vocabulary mask and set tokens with mask value 0 to -inf.
  // prefix_vocab_mask shape (batch_size, vocab_size).
  T* p = next_token_scores.scores.data();
  for (int i = 0; i < batch_size_; i++) {
    size_t prefix_vocab_mask_offset = SafeInt<size_t>(i) * next_token_scores.vocab_size;
    for (int j = 0; j < num_beams; j++) {
      for (int k = 0; k < next_token_scores.vocab_size; k++, p++) {
        if (prefix_vocab_mask_[prefix_vocab_mask_offset + static_cast<size_t>(k)] == 0) {
          *p = std::numeric_limits<T>::lowest();
        }
      }
    }
  }
}

template <typename T>
TemperatureLogitsProcessor<T>::TemperatureLogitsProcessor(float temperature) : temperature_(temperature) {
}

template <typename T>
void TemperatureLogitsProcessor<T>::Process(const ISequences* /*sequences*/,
                                            NextTokenScores<T>& next_token_scores) {
  if (temperature_ == 1.0f) {
    return;
  }

  T* p = next_token_scores.scores.data();
  for (size_t i = 0; i < next_token_scores.scores.size(); i++) {
    *p /= temperature_;
    ++p;
  }
}

template <typename T>
PresencePenaltyLogitsProcessor<T>::PresencePenaltyLogitsProcessor(const gsl::span<const int32_t>& presence_mask,
                                                                  float presence_penalty)
    : presence_mask_(presence_mask), presence_penalty_(presence_penalty) {
}

template <typename T>
void PresencePenaltyLogitsProcessor<T>::Process(const ISequences*,
                                                NextTokenScores<T>& next_token_scores) {
  if (presence_penalty_ == 0.0f) {
    return;
  }

  assert(!presence_mask_.empty());

  T* p = next_token_scores.scores.data();
  for (size_t i = 0; i < next_token_scores.scores.size(); i++) {
    *p -= presence_mask_[i] * presence_penalty_;
  }
}

template <typename T>
SequentialConstraintsFSALogitsProcessor<T>::SequentialConstraintsFSALogitsProcessor(
    const gsl::span<const int32_t>& constraints,
    const gsl::span<const int32_t>& grammar,
    int batch_beam_size,
    int max_grammar_rule_length,
    int vocab_size,
    const gsl::span<bool>& any_allowed_span,
    const gsl::span<bool>& next_constraint_allowed_span,
    const gsl::span<bool>& has_specific_allowed_tokens_span) : constraints_(constraints),
                      grammar_(grammar),
                      batch_beam_size_(batch_beam_size),
                      vocab_size_(vocab_size),
                      max_grammar_rule_length_(max_grammar_rule_length),
                      any_allowed_span_(any_allowed_span),
                      next_constraint_allowed_span_(next_constraint_allowed_span),
                      has_specific_allowed_tokens_span_(has_specific_allowed_tokens_span) {
  assert(static_cast<int>(grammar_.size()) == vocab_size_ * max_grammar_rule_length_);
  next_constraint_indexes_ = gsl::span<int32_t>(new int32_t[batch_beam_size_], batch_beam_size_);
  std::fill_n(next_constraint_indexes_.begin(), batch_beam_size_, 0);  // we point indexes to start constraint
}

template <typename T>
int SequentialConstraintsFSALogitsProcessor<T>::NextConstraint(int beam_index, int last_token) {
  int next_constraint = -1;
  int next_constraint_index = next_constraint_indexes_[beam_index];
  // -1 for next_constraint index means that we already reached the end of the constraints before
  if (next_constraint_index != -1) {
    next_constraint = constraints_[next_constraint_index];
  }

  // update pointer and get next_constraint if we have hit the next constraint
  // if we were already at the last possible constraint, set the pointer to -1 and next_constraint to -1
  if (next_constraint == last_token) {
    next_constraint_indexes_[beam_index]++;
    next_constraint_index = next_constraint_indexes_[beam_index];
    if (next_constraint_index >= static_cast<int>(constraints_.size())) {
      next_constraint_indexes_[beam_index] = -1;
      next_constraint = -1;
    } else {
      next_constraint = constraints_[next_constraint_index];
    }
  }
  return next_constraint;
}

template <typename T>
void SequentialConstraintsFSALogitsProcessor<T>::UpdateNextConstraintIndexes(const ISequences* sequences) {
  // we permute the next_constraint_indexes_ to match the new beam indexes
  std::vector<int32_t> new_next_constraint_indexes = std::vector<int32_t>(batch_beam_size_);
  for (int beam_index = 0; beam_index < batch_beam_size_; beam_index++) {
    int previous_index = sequences->GetPreviousBeamIndex(beam_index);
    new_next_constraint_indexes[beam_index] = next_constraint_indexes_[previous_index];
  }
  std::copy(new_next_constraint_indexes.begin(), new_next_constraint_indexes.end(), next_constraint_indexes_.begin());
}

template <typename T>
std::unordered_set<int32_t> SequentialConstraintsFSALogitsProcessor<T>::GetMaskedWordIds(int last_token, int next_constraint) {
  std::unordered_set<int32_t> masked_word_ids;

  // if any is set, we only start with blocking the constraint token ids
  // otherwise we start with maskin all all
  if (any_allowed_span_[last_token]) {
    for (int i = 0; i < static_cast<int>(constraints_.size()); i++) {
      masked_word_ids.insert(constraints_[i]);
    }
  } else {
    for (int i = 0; i < vocab_size_; i++) {
      masked_word_ids.insert(i);
    }
  }

  // if we can use next constraint token, we remove it from the masked word ids
  if (next_constraint_allowed_span_[last_token]) {
    masked_word_ids.erase(next_constraint);
  }

  // we go over token_id in grammar_, if token_id>=0 , remove from masked_word_ids
  // we check the has_specific_allowed_tokens_span_ to see if we need to go over the grammar
  if (has_specific_allowed_tokens_span_[last_token]) {
    gsl::span<const int32_t> grammar_subspan = grammar_.subspan(last_token * max_grammar_rule_length_, max_grammar_rule_length_);
    for (int j = 0; j < max_grammar_rule_length_; j++) {
      int32_t token_id = grammar_subspan[j];
      if (token_id >= 0) {
        masked_word_ids.erase(token_id);
      } else if (token_id == -1) {
        break;
      }
    }
  }
  return masked_word_ids;
}

template <typename T>
void SequentialConstraintsFSALogitsProcessor<T>::Process(const ISequences* sequences,
                                                         NextTokenScores<T>& next_token_scores) {
  UpdateNextConstraintIndexes(sequences);
  for (int beam_index = 0; beam_index < batch_beam_size_; beam_index++) {
    gsl::span<const int32_t> sequence = sequences->GetSequence(beam_index);
    int last_token = sequence[sequence.size() - 1];

    int next_constraint = NextConstraint(beam_index, last_token);
    std::unordered_set<int32_t> masked_word_ids = GetMaskedWordIds(last_token, next_constraint);

    next_token_scores.ApplyMask(beam_index, masked_word_ids);

#ifdef DEBUG_GENERATION
    DumpScores("SequentialConstraintsFSALogitsProcessor", next_token_scores);
#endif
  }
}

void LogitsProcessorList::Init(const BeamSearchParameters& parameters) {
  LogitsProcessorInitImpl<BeamSearchParameters>(parameters);
}

void LogitsProcessorList::Init(const GreedySearchParameters& parameters) {
  LogitsProcessorInitImpl<GreedySearchParameters>(parameters);
}

void LogitsProcessorList::Init(const SamplingParameters& parameters) {
  LogitsProcessorInitImpl<SamplingParameters>(parameters);
}

void LogitsProcessorList::Process(const ISequences* sequences,
                                  gsl::span<float>& next_token_scores,
                                  int step) {
  NextTokenScores<float> input_scores = {next_token_scores, batch_beam_size_, vocab_size_};
  for (size_t i = 0; i < processor_list_.size(); i++) {
    // Prefix vocab mask is applied to first iteration only.
    if (step > 1 && processor_list_[i] == prefix_vocab_mask_processor_.get()) {
      continue;
    }
    processor_list_[i]->Process(sequences, input_scores);
  }
}

template class MinLengthLogitsProcessor<float>;
template class MaxLengthLogitsProcessor<float>;
template class SequentialConstraintsFSALogitsProcessor<float>;

}  // namespace transformers
}  // namespace contrib
}  // namespace onnxruntime
