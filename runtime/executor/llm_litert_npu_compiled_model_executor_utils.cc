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

#include "runtime/executor/llm_litert_npu_compiled_model_executor_utils.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

#if defined(__ANDROID__) && defined(__ARM_NEON)
#include <arm_neon.h>

#include <limits>  // IWYU pragma: keep
#endif

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert

namespace litert::lm {

#if defined(__ANDROID__) && defined(__ARM_NEON)
int FindMaxIndexFloatNeon(const float* data, int size) {
  if (size <= 0) return 0;
  float32x4_t max_v4 = vdupq_n_f32(-std::numeric_limits<float>::infinity());
  int i = 0;
  for (; i <= size - 4; i += 4) {
    max_v4 = vmaxq_f32(max_v4, vld1q_f32(data + i));
  }
  float max_vals_arr[4];
  vst1q_f32(max_vals_arr, max_v4);
  float max_v = max_vals_arr[0];
  for (int j = 1; j < 4; ++j) {
    if (max_vals_arr[j] > max_v) max_v = max_vals_arr[j];
  }
  for (; i < size; ++i) {
    if (data[i] > max_v) max_v = data[i];
  }

  // Second pass: find first index with max_v
  float32x4_t target = vdupq_n_f32(max_v);
  for (i = 0; i <= size - 4; i += 4) {
    uint32x4_t cmp = vceqq_f32(vld1q_f32(data + i), target);
    uint32_t mask[4];
    vst1q_u32(mask, cmp);
    if (mask[0] || mask[1] || mask[2] || mask[3]) {
      for (int j = 0; j < 4; ++j) {
        if (mask[j]) return i + j;
      }
    }
  }
  for (; i < size; ++i) {
    if (data[i] == max_v) return i;
  }
  return 0;
}

int FindMaxIndexInt16Neon(const int16_t* data, int size) {
  if (size <= 0) return 0;
  int16x8_t max_v8 = vdupq_n_s16(std::numeric_limits<int16_t>::lowest());
  int i = 0;
  for (; i <= size - 8; i += 8) {
    max_v8 = vmaxq_s16(max_v8, vld1q_s16(data + i));
  }
  int16_t max_vals_arr[8];
  vst1q_s16(max_vals_arr, max_v8);
  int16_t max_v = max_vals_arr[0];
  for (int j = 1; j < 8; ++j) {
    if (max_vals_arr[j] > max_v) max_v = max_vals_arr[j];
  }
  for (; i < size; ++i) {
    if (data[i] > max_v) max_v = data[i];
  }

  int16x8_t target = vdupq_n_s16(max_v);
  for (i = 0; i <= size - 8; i += 8) {
    uint16x8_t cmp = vceqq_s16(vld1q_s16(data + i), target);
    uint16_t mask[8];
    vst1q_u16(mask, cmp);
    if (mask[0] || mask[1] || mask[2] || mask[3] || mask[4] || mask[5] ||
        mask[6] || mask[7]) {
      for (int j = 0; j < 8; ++j) {
        if (mask[j]) return i + j;
      }
    }
  }
  for (; i < size; ++i) {
    if (data[i] == max_v) return i;
  }
  return 0;
}

int FindMaxIndexInt8Neon(const int8_t* data, int size) {
  if (size <= 0) return 0;
  int8x16_t max_v16 = vdupq_n_s8(std::numeric_limits<int8_t>::lowest());
  int i = 0;
  for (; i <= size - 16; i += 16) {
    max_v16 = vmaxq_s8(max_v16, vld1q_s8(data + i));
  }
  int8_t max_vals_arr[16];
  vst1q_s8(max_vals_arr, max_v16);
  int8_t max_v = max_vals_arr[0];
  for (int j = 1; j < 16; ++j) {
    if (max_vals_arr[j] > max_v) max_v = max_vals_arr[j];
  }
  for (; i < size; ++i) {
    if (data[i] > max_v) max_v = data[i];
  }

  int8x16_t target = vdupq_n_s8(max_v);
  for (i = 0; i <= size - 16; i += 16) {
    uint8x16_t cmp = vceqq_s8(vld1q_s8(data + i), target);
    uint8_t mask[16];
    vst1q_u8(mask, cmp);
    // Quick check if any lane matched
    uint64_t low = vget_lane_u64(vreinterpret_u64_u8(vget_low_u8(cmp)), 0);
    uint64_t high = vget_lane_u64(vreinterpret_u64_u8(vget_high_u8(cmp)), 0);
    if (low || high) {
      for (int j = 0; j < 16; ++j) {
        if (mask[j]) return i + j;
      }
    }
  }
  for (; i < size; ++i) {
    if (data[i] == max_v) return i;
  }
  return 0;
}
#endif

absl::StatusOr<int> ApplyGreedySampling(::litert::TensorBuffer& decoded_logits,
                                        bool use_neon_sampling) {
  LITERT_ASSIGN_OR_RETURN(::litert::RankedTensorType logits_tensor_type,
                          decoded_logits.TensorType());
  if (logits_tensor_type.ElementType() == ::litert::ElementType::Float32) {
    return FindMaxIndex<float>(decoded_logits, use_neon_sampling);
  } else if (logits_tensor_type.ElementType() == ::litert::ElementType::Int16) {
    return FindMaxIndex<int16_t>(decoded_logits, use_neon_sampling);
  } else if (logits_tensor_type.ElementType() == ::litert::ElementType::Int8) {
    return FindMaxIndex<int8_t>(decoded_logits, use_neon_sampling);
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported tensor element type for greedy sampling: ",
                     logits_tensor_type.ElementType()));
  }
}

absl::Status HWKVCacheUpdate(
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>& in_buffers,
    absl::flat_hash_map<absl::string_view, ::litert::TensorBuffer>&
        out_buffers) {
  static constexpr absl::string_view kInputPos = "input_pos";
  auto& input_pos_buffer = in_buffers.at(kInputPos);

  LITERT_ASSIGN_OR_RETURN(
      auto pos_lock,
      ::litert::TensorBufferScopedLock::Create(
          input_pos_buffer, ::litert::TensorBuffer::LockMode::kRead));
  int start_pos = static_cast<const int32_t*>(pos_lock.second)[0];

  auto perform_update =
      [&](::litert::TensorBuffer& cache,
          const ::litert::TensorBuffer& slice) -> absl::Status {
    LITERT_ASSIGN_OR_RETURN(auto cache_type, cache.TensorType());
    LITERT_ASSIGN_OR_RETURN(auto slice_type, slice.TensorType());
    auto cache_dims = cache_type.Layout().Dimensions();
    auto slice_dims = slice_type.Layout().Dimensions();
    int cache_rank = cache_type.Layout().Rank();
    int slice_rank = slice_type.Layout().Rank();

    LITERT_ASSIGN_OR_RETURN(size_t cache_bytes, cache.Size());
    LITERT_ASSIGN_OR_RETURN(size_t num_elements,
                            cache_type.Layout().NumElements());
    size_t element_size = cache_bytes / num_elements;

    // Assume hidden_dim is the smaller of the last two dimensions of cache.
    int cache_last_dim = cache_dims[cache_rank - 1];
    int cache_second_last_dim = cache_dims[cache_rank - 2];
    int64_t hidden_dim = std::min(cache_last_dim, cache_second_last_dim);
    int64_t cache_seq = std::max(cache_last_dim, cache_second_last_dim);

    int cache_seq_dim = (cache_dims[cache_rank - 1] == cache_seq)
                            ? cache_rank - 1
                            : cache_rank - 2;

    int slice_seq_dim = -1;
    int slice_hidden_dim = -1;
    int64_t slice_seq = -1;

    // Find dimensions in slice
    if (slice_dims[slice_rank - 1] == hidden_dim) {
      slice_hidden_dim = slice_rank - 1;
      slice_seq_dim = slice_rank - 2;
      slice_seq = slice_dims[slice_seq_dim];
    } else if (slice_dims[slice_rank - 2] == hidden_dim) {
      slice_hidden_dim = slice_rank - 2;
      slice_seq_dim = slice_rank - 1;
      slice_seq = slice_dims[slice_seq_dim];
    }

    if (slice_hidden_dim == -1) {
      return absl::InternalError(
          "Failed to identify hidden dimension in slice");
    }

    if (start_pos + slice_seq > cache_seq) {
      return absl::OutOfRangeError("KV-cache update out of range");
    }

    LITERT_ASSIGN_OR_RETURN(
        auto cache_lock, ::litert::TensorBufferScopedLock::Create(
                             cache, ::litert::TensorBuffer::LockMode::kWrite));
    LITERT_ASSIGN_OR_RETURN(
        auto slice_lock, ::litert::TensorBufferScopedLock::Create(
                             slice, ::litert::TensorBuffer::LockMode::kRead));

    uint8_t* cache_ptr = static_cast<uint8_t*>(cache_lock.second);
    const uint8_t* slice_ptr = static_cast<const uint8_t*>(slice_lock.second);

    bool cache_is_transposed = (cache_seq_dim == cache_rank - 1);
    bool slice_is_transposed = (slice_seq_dim == slice_rank - 1);

    int64_t outer_size = 1;
    for (int i = 0; i < cache_rank - 2; ++i) {
      outer_size *= cache_dims[i];
    }

    for (int64_t o = 0; o < outer_size; ++o) {
      uint8_t* c_ptr = cache_ptr + o * (cache_seq * hidden_dim * element_size);
      const uint8_t* s_ptr =
          slice_ptr + o * (slice_seq * hidden_dim * element_size);

      if (!cache_is_transposed) {
        if (!slice_is_transposed || slice_seq == 1) {
          // Cache is [..., seq, hidden], Slice is [..., seq, hidden] (or seq=1)
          std::memcpy(c_ptr + (start_pos * hidden_dim * element_size), s_ptr,
                      slice_seq * hidden_dim * element_size);
        } else {
          // Cache is [..., seq, hidden], Slice is [..., hidden, seq]
          for (int64_t s = 0; s < slice_seq; ++s) {
            for (int64_t h = 0; h < hidden_dim; ++h) {
              std::memcpy(
                  c_ptr + ((start_pos + s) * hidden_dim + h) * element_size,
                  s_ptr + (h * slice_seq + s) * element_size, element_size);
            }
          }
        }
      } else {
        // Cache is [..., hidden, seq]
        if (slice_seq == 1) {
#if defined(__ANDROID__) && defined(__ARM_NEON) && defined(__aarch64__)
          if (element_size == 1) {
            int64_t h = 0;
            for (; h <= hidden_dim - 16; h += 16) {
              uint8x16_t v = vld1q_u8(s_ptr + h);
              c_ptr[(h + 0) * cache_seq + start_pos] = vgetq_lane_u8(v, 0);
              c_ptr[(h + 1) * cache_seq + start_pos] = vgetq_lane_u8(v, 1);
              c_ptr[(h + 2) * cache_seq + start_pos] = vgetq_lane_u8(v, 2);
              c_ptr[(h + 3) * cache_seq + start_pos] = vgetq_lane_u8(v, 3);
              c_ptr[(h + 4) * cache_seq + start_pos] = vgetq_lane_u8(v, 4);
              c_ptr[(h + 5) * cache_seq + start_pos] = vgetq_lane_u8(v, 5);
              c_ptr[(h + 6) * cache_seq + start_pos] = vgetq_lane_u8(v, 6);
              c_ptr[(h + 7) * cache_seq + start_pos] = vgetq_lane_u8(v, 7);
              c_ptr[(h + 8) * cache_seq + start_pos] = vgetq_lane_u8(v, 8);
              c_ptr[(h + 9) * cache_seq + start_pos] = vgetq_lane_u8(v, 9);
              c_ptr[(h + 10) * cache_seq + start_pos] = vgetq_lane_u8(v, 10);
              c_ptr[(h + 11) * cache_seq + start_pos] = vgetq_lane_u8(v, 11);
              c_ptr[(h + 12) * cache_seq + start_pos] = vgetq_lane_u8(v, 12);
              c_ptr[(h + 13) * cache_seq + start_pos] = vgetq_lane_u8(v, 13);
              c_ptr[(h + 14) * cache_seq + start_pos] = vgetq_lane_u8(v, 14);
              c_ptr[(h + 15) * cache_seq + start_pos] = vgetq_lane_u8(v, 15);
            }
            for (; h < hidden_dim; ++h) {
              c_ptr[h * cache_seq + start_pos] = s_ptr[h];
            }
          } else if (element_size == 2) {
            int64_t h = 0;
            const uint16_t* s_ptr16 = reinterpret_cast<const uint16_t*>(s_ptr);
            uint16_t* c_ptr16 = reinterpret_cast<uint16_t*>(c_ptr);
            for (; h <= hidden_dim - 8; h += 8) {
              uint16x8_t v = vld1q_u16(s_ptr16 + h);
              c_ptr16[(h + 0) * cache_seq + start_pos] = vgetq_lane_u16(v, 0);
              c_ptr16[(h + 1) * cache_seq + start_pos] = vgetq_lane_u16(v, 1);
              c_ptr16[(h + 2) * cache_seq + start_pos] = vgetq_lane_u16(v, 2);
              c_ptr16[(h + 3) * cache_seq + start_pos] = vgetq_lane_u16(v, 3);
              c_ptr16[(h + 4) * cache_seq + start_pos] = vgetq_lane_u16(v, 4);
              c_ptr16[(h + 5) * cache_seq + start_pos] = vgetq_lane_u16(v, 5);
              c_ptr16[(h + 6) * cache_seq + start_pos] = vgetq_lane_u16(v, 6);
              c_ptr16[(h + 7) * cache_seq + start_pos] = vgetq_lane_u16(v, 7);
            }
            for (; h < hidden_dim; ++h) {
              c_ptr16[h * cache_seq + start_pos] = s_ptr16[h];
            }
          } else
#endif
          {
            for (int64_t h = 0; h < hidden_dim; ++h) {
              std::memcpy(c_ptr + (h * cache_seq + start_pos) * element_size,
                          s_ptr + h * element_size, element_size);
            }
          }
        } else if (slice_is_transposed) {
          // Cache is [..., hidden, seq], Slice is [..., hidden, seq]
          for (int64_t h = 0; h < hidden_dim; ++h) {
            std::memcpy(c_ptr + (h * cache_seq + start_pos) * element_size,
                        s_ptr + (h * slice_seq) * element_size,
                        slice_seq * element_size);
          }
        } else {
          // Cache is [..., hidden, seq], Slice is [..., seq, hidden]
          for (int64_t s = 0; s < slice_seq; ++s) {
            for (int64_t h = 0; h < hidden_dim; ++h) {
              std::memcpy(
                  c_ptr + (h * cache_seq + start_pos + s) * element_size,
                  s_ptr + (s * hidden_dim + h) * element_size, element_size);
            }
          }
        }
      }
    }
    return absl::OkStatus();
  };

  for (int layer_id = 0;; ++layer_id) {
    char k_cache_name[32];
    snprintf(k_cache_name, sizeof(k_cache_name), "kv_cache_k_%d", layer_id);
    if (!in_buffers.contains(k_cache_name)) break;

    char v_cache_name[32];
    snprintf(v_cache_name, sizeof(v_cache_name), "kv_cache_v_%d", layer_id);
    char k_slice_name[32];
    snprintf(k_slice_name, sizeof(k_slice_name), "kv_slice_k_%d", layer_id);
    char v_slice_name[32];
    snprintf(v_slice_name, sizeof(v_slice_name), "kv_slice_v_%d", layer_id);

    auto& in_k_cache = in_buffers.at(k_cache_name);
    auto& in_v_cache = in_buffers.at(v_cache_name);
    const auto& k_slice = in_buffers.at(k_slice_name);
    const auto& v_slice = in_buffers.at(v_slice_name);

    LITERT_RETURN_IF_ERROR(perform_update(in_k_cache, k_slice));
    LITERT_RETURN_IF_ERROR(perform_update(in_v_cache, v_slice));

    if (out_buffers.contains(k_cache_name)) {
      auto& out_k_cache = out_buffers.at(k_cache_name);
      if (in_k_cache.Get() != out_k_cache.Get()) {
        LITERT_RETURN_IF_ERROR(perform_update(out_k_cache, k_slice));
      }
    }
    if (out_buffers.contains(v_cache_name)) {
      auto& out_v_cache = out_buffers.at(v_cache_name);
      if (in_v_cache.Get() != out_v_cache.Get()) {
        LITERT_RETURN_IF_ERROR(perform_update(out_v_cache, v_slice));
      }
    }
  }
  return absl::OkStatus();
}
}  // namespace litert::lm
