// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_LITERT_MTP_DRAFTER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_LITERT_MTP_DRAFTER_H_

#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/sampler.h"
#include "runtime/executor/llm_executor_settings.h"

namespace litert::lm {

class LlmLiteRtMtpDrafter {
 public:
  // Create an instance of LlmLiteRtMtpDrafter.
  // The executor_settings is used to create the MTP drafter model and its
  // sampler.
  // The base_model is used for verification. The model is expected to have
  // "verify" signature and be invokable when Draft is called (i.e., not busy).
  static absl::StatusOr<std::unique_ptr<LlmLiteRtMtpDrafter>> Create(
      const LlmExecutorSettings& executor_settings, CompiledModel& base_model);

  // Draft the next set of tokens using the MTP drafter model.
  // Inputs:
  //   position: The current position of the input sequence.
  //   logits: The logits from the base model.
  //   kv_cache_buffers: The key/value cache buffers for the base model.
  // Outputs:
  //   The drafted tokens from the MTP drafter model with the shape:
  //   [batch_size, num_tokens].
  absl::StatusOr<std::vector<std::vector<int>>> Draft(
      int position, TensorBuffer& logits,
      absl::flat_hash_map<absl::string_view, TensorBuffer>& kv_cache_buffers);

 private:
  LlmLiteRtMtpDrafter(std::unique_ptr<CompiledModel> mtp_drafter_model,
                      CompiledModel& base_model,
                      std::unique_ptr<Sampler> sampler)
      : mtp_drafter_model_(std::move(mtp_drafter_model)),
        base_model_(base_model),
        sampler_(std::move(sampler)) {}

  // The MTP drafter model.
  std::unique_ptr<CompiledModel> mtp_drafter_model_;

  // MTP drafter owned buffers.
  // Includes: Position, Mask, and Logits Tensors.
  absl::flat_hash_map<absl::string_view, TensorBuffer>
      mtp_drafter_input_buffers_;
  // Includes: Logits Tensors.
  absl::flat_hash_map<absl::string_view, TensorBuffer>
      mtp_drafter_output_buffers_;

  // The base model, used for verification.
  CompiledModel& base_model_;

  // Greedy sampler owned by the drafter to sample the logits from the MTP
  // drafter model.
  std::unique_ptr<Sampler> sampler_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_LITERT_MTP_DRAFTER_H_
