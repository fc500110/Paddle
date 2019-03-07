// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/inference/anakin/convert/density_prior_box.h"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

using anakin::graph::GraphGlobalMem;
using anakin::AK_FLOAT;
using anakin::saber::NV;
using anakin::saber::Shape;
using anakin::PTuple;

namespace paddle {
namespace inference {
namespace anakin {

void DensityPriorBoxOpConverter::operator()(const framework::proto::OpDesc &op,
                                            const framework::Scope &scope,
                                            bool test_mode) {
  framework::OpDesc op_desc(op, nullptr);
  PADDLE_ENFORCE_EQ(op_desc.Input("Input").size(), 1UL);
  PADDLE_ENFORCE_EQ(op_desc.Input("Image").size(), 1UL);
  PADDLE_ENFORCE_EQ(op_desc.Output("Boxes").size(), 1UL);
  PADDLE_ENFORCE_EQ(op_desc.Output("Variances").size(), 1UL);

  auto op_name = op_desc.Type() + ":" + op_desc.Output("Output").front();
  std::vector<std::string> inputs{op_desc.Input("Input"),
                                  op_desc.Input("Image")};
  std::vector<std::string> outputs{op_desc.Output("Boxes")};
  engine_->AddOp(op_name, "PriorBox", inputs, outputs);

  // fluid attrs
  auto variances = boost::get<std::vector<int>>(op_desc.GetAttr("variances"));
  auto is_clip = boost::get<bool>(op_desc.GetAttr("clip"));
  auto step_w = boost::get<float>(op_desc.GetAttr("step_w"));
  auto step_h = boost::get<float>(op_desc.GetAttr("step_h"));
  auto offset = boost::get<float>(op_desc.GetAttr("offset"));
  auto fixed_sizes =
      boost::get<std::vector<float>>(op_desc.GetAttr("fixed_sizes"));
  auto fixed_ratios =
      boost::get<std::vector<float>>(op_desc.GetAttr("fixed_ratios"));
  auto densities = boost::get<std::vector<float>>(op_desc.GetAttr("densities"));

  // anakin attrs:
  // min_size, max_size, aspect_ratio, fixed_size, fixed_ratio, density
  // is_flip, is_clip, variance, image_h, image_w, step_h, step_w, offset, order
  engine_->AddOpAttr<PTuple<float>>(op_name, "fixed_size", fixed_sizes);
  engine_->AddOpAttr<PTuple<float>>(op_name, "fixed_ratio", fixed_ratios);
  engine_->AddOpAttr<PTuple<float>>(op_name, "density", densities);
  engine_->AddOpAttr(op_name, "is_clip", false);
  engine_->AddOpAttr(op_name, "is_flip", false);
  engine_->AddOpAttr(op_name, "variance", variances);
  engine_->AddOpAttr(op_name, "image_h", 0);
  engine_->AddOpAttr(op_name, "image_w", 0);
  engine_->AddOpAttr(op_name, "step_h", step_h);
  engine_->AddOpAttr(op_name, "step_w", step_w);
  engine_->AddOpAttr(op_name, "offset", offset);
  engine_->AddOpAttr<PTuple<std::string>>(op_name, "order",
                                          {"MIN", "COM", "MAX"});
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

REGISTER_ANAKIN_OP_CONVERTER(density_prior_box, DensityPriorBoxOpConverter);
