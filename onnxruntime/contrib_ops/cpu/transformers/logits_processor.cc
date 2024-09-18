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

template <typename T>
void MinLengthLogitsProcessor<T>::Process(const ISequences* sequences,
                                          NextTokenScores<T>& next_token_scores) {
  // if we didn't reach yet minimum length
  // set the eos token score to lowest for each beam
  if (sequences->GetSequenceLength() < min_length_) {
    next_token_scores.SetScore(eos_token_id_, std::numeric_limits<T>::lowest());
  }
}


template <typename T>
MaxLengthLogitsProcessor<T>::MaxLengthLogitsProcessor(int max_length, int eos_token_id)
    : max_length_(max_length), eos_token_id_(eos_token_id) {}

template <typename T>
void MaxLengthLogitsProcessor<T>::Process(const ISequences* sequences,
                                          NextTokenScores<T>& next_token_scores
                                          ) {
  // We have to emit EOS on the last possible position.
  // if we have reached the max length
  // set scores for all tokens but eos to lowest for each beam
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
NoRepeatNGramLogitsProcessor<T>::NoRepeatNGramLogitsProcessor(int ngram_size) : ngram_size_(ngram_size) {
}

template <typename T>
void NoRepeatNGramLogitsProcessor<T>::Process(const ISequences* sequences,
                                              NextTokenScores<T>& next_token_scores) {
  if (ngram_size_ == 0 || ngram_size_ > sequences->GetSequenceLength()) {
    return;
  }

  const gsl::index prefix_length = static_cast<gsl::index>(ngram_size_) - 1;
  int batch_beam_size = next_token_scores.batch_beam_size;

  for (int i = 0; i < batch_beam_size; i++) {
    gsl::span<T> beam_token_scores = next_token_scores.GetScores(i);
    gsl::span<const int32_t> sequence = sequences->GetSequence(i);

    gsl::span<const int32_t> prefix = sequence.subspan(sequence.size() - prefix_length);
    ORT_ENFORCE(prefix.size() == narrow<size_t>(prefix_length));

    std::unordered_set<int32_t> blocked_word_ids;
    for (int j = 0; j <= static_cast<int>(sequence.size()) - ngram_size_; j++) {
      // Here we use naive algorithm for matching. The complexity is O(batch_beam_size * ngram_size * sequence_length)
      // TODO(tianleiwu): build N-Gram index (hash table with prefix of length NGram - 1 as key,
      //                  and list of last word of NGram as value) for fast matching.
      if (ngram_size_ == 1 || SpanEq(prefix, sequence.subspan(j, prefix_length))) {
        blocked_word_ids.insert(sequence[static_cast<gsl::index>(j) + prefix_length]);
      }
    }

    for (const int32_t word_id : blocked_word_ids) {
      beam_token_scores[word_id] = std::numeric_limits<T>::lowest();
    }
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
    int vocab_size
)   : constraints_(constraints),
      grammar_(grammar),
      batch_beam_size_(batch_beam_size),
      vocab_size_(vocab_size),
      max_grammar_rule_length_(max_grammar_rule_length)
     {
      assert(static_cast<int>(grammar_.size()) == vocab_size_ * max_grammar_rule_length_);
      next_constraint_indexes_ = gsl::span<int32_t>(new int32_t[batch_beam_size_], batch_beam_size_);
      std::fill_n(next_constraint_indexes_.begin(), batch_beam_size_, 0);  // we point indexes to start constraint

      fixed_grammar_mask_span_ = gsl::span<int32_t>(new int32_t[vocab_size_*vocab_size_], vocab_size_*vocab_size_);
      std::fill_n(fixed_grammar_mask_span_.begin(), vocab_size_*vocab_size_, ALLOWED_);  //default all allowed


      for (int vocab_index = 0; vocab_index < vocab_size; vocab_index++) {
        // now lets fill the fixed_grammar_maks_span by creating first a rule_mask_span
        // from the vocab_rule_span (grammer_ for this c)
        // at the end we copy from one to the other

        // target mask
        gsl::span<int32_t> rule_mask_span = gsl::span<int32_t>(new int32_t[vocab_size], vocab_size);
        // // source grammar
        assert(vocab_index * max_grammar_rule_length_ + max_grammar_rule_length_ <= static_cast<int>(grammar_.size()));
        gsl::span<const int32_t> rule_span = grammar_.subspan(vocab_index * max_grammar_rule_length_, max_grammar_rule_length_);

        // check if -2 is in the rule_span if so, set all allowed otherwise none allowed
        if (std::find(rule_span.begin(), rule_span.end(), ANY_RULE_) != rule_span.end()) {
          std::fill_n(rule_mask_span.begin(), rule_mask_span.size(), ALLOWED_);
        } else {
          std::fill_n(rule_mask_span.begin(), rule_mask_span.size(), MASKED_);
        }
        // now set all tokens in constraints to 0
        for (int j = 0; j < static_cast<int>(constraints.size()); j++) {
          rule_mask_span[constraints[j]] = MASKED_;
        }
        // now go over the rules, if it contains anything >= 0, set the mask to 1
        // in theory we shouldn't allow items in the constraint list/ raise error?
        for (int rule_index = 0; rule_index < max_grammar_rule_length_; rule_index++) {
          if (rule_span[rule_index] >= 0) {
            rule_mask_span[rule_span[rule_index]] = ALLOWED_;
          }
        }
        // now we put it in the global span
        for (int i = 0; i < vocab_size_; i++) {
          fixed_grammar_mask_span_[static_cast<gsl::index>(vocab_index) * vocab_size_ + i] = rule_mask_span[i];
        }
      }
}


template <typename T>
void SequentialConstraintsFSALogitsProcessor<T>::Process(const ISequences* sequences,
                                          NextTokenScores<T>& next_token_scores) {

  int num_batch_beams = next_token_scores.batch_beam_size;
  assert(num_batch_beams == batch_beam_size_);

  std::vector<int32_t> new_next_constraint_indexes = std::vector<int32_t>(batch_beam_size_);

  for (int beam_index = 0; beam_index < batch_beam_size_; beam_index++) {
    int previous_index = sequences->GetPreviousBeamIndex(beam_index);
    new_next_constraint_indexes[beam_index] = next_constraint_indexes_[previous_index];
  }
  std::copy(new_next_constraint_indexes.begin(), new_next_constraint_indexes.end(), next_constraint_indexes_.begin());


  for (int beam_index = 0; beam_index < batch_beam_size_; beam_index++) {
    int next_constraint = -1;
    // int next_constraint_index = sequences->GetFSAState(beam_index)
    int next_constraint_index = next_constraint_indexes_[beam_index];
    if( next_constraint_index != -1){
      // -1 means that we already reached the end of the constraints before
      next_constraint = constraints_[next_constraint_index];
    }
    gsl::span<const int32_t> sequence = sequences->GetSequence(beam_index);

    // get last element of sequence  nd put in last_token
    int last_token = sequence[sequence.size() - 1];
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

    // get fixed grammar mask for last_token
    gsl::span<int32_t> grammar_mask_last_token = fixed_grammar_mask_span_.subspan(last_token * vocab_size_, vocab_size_);
    // if the next contstraint = -1, we are at the end and the dynamic_grammar_mask == fixed_grammar_mask_span
    // else we need to update with the next constraint token set to allow ( 1)
    // we do this by copy the fixed grammar mask span into a new (dynamic) span
    // then set the next constraint to 1

    gsl::span<int32_t> dynamic_grammar_mask_span = gsl::span<int32_t>(new int32_t[vocab_size_], vocab_size_);
    std::copy(grammar_mask_last_token.begin(), grammar_mask_last_token.end(), dynamic_grammar_mask_span.begin());
    if (next_constraint != -1) {
      // we are not at the end, we need to update the dynamic grammar mask
      dynamic_grammar_mask_span[next_constraint] = ALLOWED_;
      // assert this is now set to 1
      assert(dynamic_grammar_mask_span[next_constraint] == ALLOWED_);
    }
    // now we update the next token scores for the beam
    next_token_scores.ApplyMask(beam_index, dynamic_grammar_mask_span, MASKED_);
    // candidates = [
    //     (1, 2, 3)
    //        state = [fsa_0]
    //     (a, b, c)
    //        state = [fsa_1]
    //     (u, v, w)
    //        state = [fsa_5]
    // ]
    // candidates = [
    //     (1, 2, 3, 4)
    //        state = [fsa_0]
    //     (u,v, w, x)
    //        state = [fsa_5]
    //     (u,v, w, z)
    //        state = [fsa_5]
    // ]
    //  0 // 2 4 5 3 3 3
         //     1
    // 1 //  2 4 3 3 3 6
         //    1       2
    // 2 //  2 4 3 3 3 5
         //    1
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
