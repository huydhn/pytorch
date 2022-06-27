#include <gtest/gtest.h>
#include <torch/csrc/utils/schema_info.h>

namespace torch {
namespace utils {

TEST(SchemaInfoIsMutableTest, Basic) {
  SchemaInfo schema_info(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  ASSERT_TRUE(schema_info.isMutating(0));
  ASSERT_TRUE(schema_info.isMutating("self"));
  ASSERT_FALSE(schema_info.isMutating(1));
  ASSERT_FALSE(schema_info.isMutating("other"));
}

TEST(SchemaInfoIsMutableTest, InvalidArgument) {
  SchemaInfo schema_info(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  ASSERT_THROW(schema_info.isMutating(-1), c10::Error);
  ASSERT_THROW(schema_info.isMutating(4), c10::Error);
}

TEST(SchemaInfoAreAliasingTest, Basic) {
  SchemaInfo schema_info(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  ASSERT_TRUE(schema_info.areAliasing(
      {SchemaArgType::input, 0}, {SchemaArgType::output, 0}));
  ASSERT_FALSE(schema_info.areAliasing(
      {SchemaArgType::input, 1}, {SchemaArgType::output, 0}));
  ASSERT_FALSE(schema_info.areAliasing(
      {SchemaArgType::input, 1}, {SchemaArgType::input, 0}));
}

TEST(SchemaInfoAreAliasingTest, InvalidArgument) {
  SchemaInfo schema_info(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  ASSERT_THROW(
      schema_info.areAliasing(
          {SchemaArgType::input, -1}, {SchemaArgType::output, 0}),
      c10::Error);
  ASSERT_THROW(
      schema_info.areAliasing(
          {SchemaArgType::input, 0}, {SchemaArgType::output, -1}),
      c10::Error);
}

TEST(SchemaInfoAreAliasingTest, Wildcard) {
  SchemaInfo schema_info(
      "aten::split.Tensor(Tensor(a -> *) self, int split_size, int dim=0) -> Tensor(a)[]");
  ASSERT_TRUE(schema_info.areAliasing(
      {SchemaArgType::input, 0}, {SchemaArgType::output, 0}));
}

TEST(SchemaInfoIsDeterministicTest, Basic) {
  SchemaInfo deterministic_schema_info(
      "aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> (Tensor(a!))");
  SchemaInfo nondeterministic_schema_info(
      "aten::dropout(Tensor input, float p, bool train) -> Tensor");
  ASSERT_TRUE(deterministic_schema_info.isDeterministic());
  ASSERT_FALSE(nondeterministic_schema_info.isDeterministic());
}
} // namespace utils
} // namespace torch
