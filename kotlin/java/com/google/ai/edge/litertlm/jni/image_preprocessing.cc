#include "third_party/odml/litert_lm/kotlin/java/com/google/ai/edge/litertlm/jni/image_preprocessing.h"

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_layout.h"  // from @litert
#include "runtime/components/preprocessor/image_preprocessor.h"
#include "runtime/components/preprocessor/stb_image_preprocessor.h"
#include "runtime/engine/io_types.h"

namespace litert::lm::jni {

using ::litert::Dimensions;
using ::litert::lm::ImagePreprocessParameter;
using ::litert::lm::InputImage;
using ::litert::lm::StbImagePreprocessor;

absl::StatusOr<InputImage> PreprocessImage(const InputImage& input_image) {
  StbImagePreprocessor image_preprocessor;
  ImagePreprocessParameter image_preprocess_parameter;
  image_preprocess_parameter.SetTargetDimensions(Dimensions({1, 768, 768, 3}));
  ABSL_LOG(INFO) << "Processing Image with size: "
                 << input_image.GetRawImageBytes().value().size();
  absl::StatusOr<InputImage> processed_image =
      image_preprocessor.Preprocess(input_image, image_preprocess_parameter);
  return processed_image;
}

}  // namespace litert::lm::jni
