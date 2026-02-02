#ifndef THIRD_PARTY_ODML_LITERT_LM_KOTLIN_JAVA_COM_GOOGLE_AI_EDGE_LITRTLM_JNI_IMAGE_PREPROCESSING_H_
#define THIRD_PARTY_ODML_LITERT_LM_KOTLIN_JAVA_COM_GOOGLE_AI_EDGE_LITRTLM_JNI_IMAGE_PREPROCESSING_H_

#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/engine/io_types.h"

namespace litert::lm::jni {

absl::StatusOr<InputImage> PreprocessImage(const InputImage& input_image);

}  // namespace litert::lm::jni

#endif  // THIRD_PARTY_ODML_LITERT_LM_KOTLIN_JAVA_COM_GOOGLE_AI_EDGE_LITRTLM_JNI_IMAGE_PREPROCESSING_H_
