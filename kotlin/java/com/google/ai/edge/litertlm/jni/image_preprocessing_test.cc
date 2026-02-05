#include "third_party/odml/litert_lm/kotlin/java/com/google/ai/edge/litertlm/jni/image_preprocessing.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/engine/io_types.h"

namespace litert::lm::jni {
namespace {

using ::testing::status::StatusIs;

// Returns bytes of a 1x1 BMP image with a single red pixel.
std::string Create1x1RedBmp() {
  const char kBmpBytes[] = {
      0x42, 0x4D, 0x3A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x36, 0x00,
      0x00, 0x00, 0x28, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00,
      0x00, 0x00, 0x01, 0x00, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0x00};
  return std::string(kBmpBytes, sizeof(kBmpBytes));
}

TEST(ImagePreprocessingTest, PreprocessImageWithEmptyImage) {
  InputImage empty_image("");
  EXPECT_THAT(PreprocessImage(empty_image),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ImagePreprocessingTest, PreprocessImageWithValidRedImage) {
  InputImage red_image(Create1x1RedBmp());
  ASSERT_OK_AND_ASSIGN(InputImage result, PreprocessImage(red_image));
  EXPECT_TRUE(result.IsTensorBuffer());
  ASSERT_OK_AND_ASSIGN(const TensorBuffer* tensor_buffer,
                       result.GetPreprocessedImageTensor());
  auto packed_size = tensor_buffer->PackedSize();
  ASSERT_TRUE(packed_size);
  EXPECT_EQ(*packed_size, 768 * 768 * 3 * sizeof(float));
}

TEST(ImagePreprocessingTest, PreprocessImageWithInvalidImageData) {
  InputImage invalid_image("this is not an image");
  EXPECT_THAT(PreprocessImage(invalid_image),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace litert::lm::jni
