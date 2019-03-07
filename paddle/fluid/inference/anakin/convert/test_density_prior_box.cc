/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>
#include "paddle/fluid/inference/anakin/convert/density_prior_box.h"
#include "paddle/fluid/inference/anakin/convert/op_converter.h"
#include "paddle/fluid/inference/anakin/convert/ut_helper.h"

namespace paddle {
namespace inference {
namespace anakin {

TEST(density_prior_box_op, test) {
  auto* density_prior_box_converter =
      Registry<AnakinOpConverter>::Global().Lookup("density_prior_box");
  ASSERT_TRUE(density_prior_box_converter != nullptr);
  std::unordered_set<std::string> parameters({"density_prior_box-Y"});
  framework::Scope scope;
  AnakinConvertValidation validator(parameters, scope);
  validator.DeclInputVar("density_prior_box-Input", {1, 3, 3, 3});
  validator.DeclParamVar("density_prior_box-Image", {1, 3, 3, 3});
  validator.DeclOutputVar("density_prior_box-Boxes", {1, 3, 3, 3});

  // Prepare Op description
  framework::OpDesc desc;
  desc.SetType("density_prior_box");
  desc.SetInput("Input", {"density_prior_box-Input"});
  desc.SetInput("Filter", {"density_prior_box-Image"});
  desc.SetOutput("Output", {"density_prior_box-Boxes"});

  std::vector<int> variances;
  bool clip{true};
  bool flatten_to_2d{false};
  float step_w{0};
  float step_h{0};
  float offset{0};
  std::vector<float> fixed_sizes;
  std::vector<float> fixed_ratios;
  std::vector<int> densities;

  desc.SetAttr("variances", std::vector<int>(1, 3, 3, 3));
  desc.SetAttr("clip", clip);
  desc.SetAttr("flatten_to_2d", flatten_to_2d);
  desc.SetAttr("step_w", step_w);
  desc.SetAttr("step_h", step_h);
  desc.SetAttr("offset", offset);
  desc.SetAttr("fixed_sizes", fixed_sizes);
  desc.SetAttr("fixed_ratios");
  desc.SetAttr("densities");

  validator.SetOp(*desc.Proto());

  validator.Execute(1);
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

USE_OP(density_prior_box);
USE_ANAKIN_CONVERTER(density_prior_box);
