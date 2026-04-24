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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_LITERT_NPU_COMPILED_MODEL_EXECUTOR_UTILS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_LITERT_NPU_COMPILED_MODEL_EXECUTOR_UTILS_H_

#include <cstddef>
#include <limits>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert

namespace litert::lm {

#if defined(__ANDROID__) && defined(__ARM_NEON)
int FindMaxIndexFloatNeon(const float* data, int size);
int FindMaxIndexInt16Neon(const int16_t* data, int size);
int FindMaxIndexInt8Neon(const int8_t* data, int size);
#endif

// Generic function to find the index of the maximum value in a TensorBuffer.
// Uses NEON optimizations if available.
template <typename T>
absl::StatusOr<int> FindMaxIndex(::litert::TensorBuffer& decoded_logits,
                                 bool use_neon_sampling) {
  LITERT_ASSIGN_OR_RETURN(
      auto lock_and_addr,
      ::litert::TensorBufferScopedLock::Create(
          decoded_logits, ::litert::TensorBuffer::LockMode::kRead));
  const T* data = static_cast<const T*>(lock_and_addr.second);
  LITERT_ASSIGN_OR_RETURN(::litert::RankedTensorType tensor_type,
                          decoded_logits.TensorType());
  LITERT_ASSIGN_OR_RETURN(size_t size, tensor_type.Layout().NumElements());

  if (size == 0) {
    return absl::InvalidArgumentError("Logits buffer is empty.");
  }

#if defined(__ANDROID__) && defined(__ARM_NEON)
  if (use_neon_sampling) {
    if constexpr (std::is_same_v<T, float>) {
      return FindMaxIndexFloatNeon(data, static_cast<int>(size));
    } else if constexpr (std::is_same_v<T, int16_t>) {
      return FindMaxIndexInt16Neon(data, static_cast<int>(size));
    } else if constexpr (std::is_same_v<T, int8_t>) {
      return FindMaxIndexInt8Neon(data, static_cast<int>(size));
    }
  }
#endif

  int max_index = 0;
  T max_value = std::numeric_limits<T>::lowest();
  for (int i = 0; i < size; ++i) {
    if (data[i] > max_value) {
      max_value = data[i];
      max_index = i;
    }
  }
  return max_index;
}

// Applies greedy sampling (argmax) to the decoded logits.
absl::StatusOr<int> ApplyGreedySampling(::litert::TensorBuffer& decoded_logits,
                                        bool use_neon_sampling);

// Performs manual KV cache update.
absl::Status HWKVCacheUpdate(
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>& in_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        out_buffers);

// Performs manual attention mask update.
absl::Status HWMaskUpdate(
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>& in_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        out_buffers);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LLM_LITERT_NPU_COMPILED_MODEL_EXECUTOR_UTILS_H_
