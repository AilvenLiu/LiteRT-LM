#include "runtime/executor/llm_litert_mtp_drafter.h"

#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/executor/llm_executor_settings.h"

namespace litert::lm {

absl::StatusOr<std::unique_ptr<LlmLiteRtMtpDrafter>>
LlmLiteRtMtpDrafter::Create(const LlmExecutorSettings& executor_settings,
                            CompiledModel& base_model) {
  return absl::UnimplementedError("Not implemented yet.");
}

absl::StatusOr<std::vector<std::vector<int>>> LlmLiteRtMtpDrafter::Draft(
    int position, TensorBuffer& logits,
    absl::flat_hash_map<absl::string_view, TensorBuffer>& kv_cache_buffers) {
  return absl::UnimplementedError("Not implemented yet.");
}

}  // namespace litert::lm
