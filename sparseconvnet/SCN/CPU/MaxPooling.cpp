// Copyright 2016-present, Facebook, Inc.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

template <typename T>
void MaxPooling_ForwardPass(T *input_features, T *output_features, Int nPlanes,
                            Int input_stride, Int output_stride, const Int *rules,
                            Int nHot) {
  Int outSite;
#pragma omp parallel for private(outSite)
  for (outSite = 0; outSite < nHot; outSite++) {
    Int i = rules[2 * outSite] * input_stride;
    Int o = rules[2 * outSite + 1] * output_stride;
    for (Int plane = 0; plane < nPlanes; plane++)
      if (output_features[o + plane] < input_features[i + plane])
        output_features[o + plane] = input_features[i + plane];
  }
}
template <typename T>
void MaxPooling_BackwardPass(T *input_features, T *d_input_features,
                             T *output_features, T *d_output_features,
                             Int nPlanes, Int input_stride, Int output_stride,
                             const Int *rules, Int nHot) {
  Int outSite;
#pragma omp parallel for private(outSite)
  for (outSite = 0; outSite < nHot; outSite++) {
    Int i = rules[2 * outSite] * input_stride;
    Int o = rules[2 * outSite + 1] * output_stride;
    for (Int plane = 0; plane < nPlanes; plane++)
      if (output_features[o + plane] == input_features[i + plane])
        d_input_features[i + plane] += d_output_features[o + plane];
  }
}

template <typename T, Int Dimension>
void cpu_MaxPooling_updateOutput(
    /*int64_t*/ at::Tensor &inputSize, /*int64_t*/ at::Tensor &outputSize,
    /*int64_t*/ at::Tensor &poolSize,
    /*int64_t*/ at::Tensor &poolStride, Metadata<Dimension> &m,
    /*float*/ at::Tensor &input_features,
    /*float*/ at::Tensor &output_features, int64_t nFeaturesToDrop) {

  Int nPlanes = input_features.size(1) - nFeaturesToDrop;
  const auto &_rules =
      m.getRuleBook(inputSize, outputSize, poolSize, poolStride, true);
  Int nActive = m.getNActive(outputSize);
  output_features.resize_({nActive, input_features.size(1) - nFeaturesToDrop});
  output_features.zero_();

  auto iF = input_features.data_ptr<T>() + nFeaturesToDrop;
  auto oF = output_features.data_ptr<T>();

  for (auto &r : _rules) {
    Int nHot = r.size() / 2;
    MaxPooling_ForwardPass<T>(iF, oF, nPlanes, input_features.stride(0),
                              output_features.stride(0), &r[0], nHot);
  }
}
template <typename T, Int Dimension>
void cpu_MaxPooling_updateGradInput(
    /*int64_t*/ at::Tensor &inputSize, /*int64_t*/ at::Tensor &outputSize,
    /*int64_t*/ at::Tensor &poolSize,
    /*int64_t*/ at::Tensor &poolStride, Metadata<Dimension> &m,
    /*float*/ at::Tensor &input_features,
    /*float*/ at::Tensor &d_input_features,
    /*float*/ at::Tensor &output_features,
    /*float*/ at::Tensor &d_output_features, int64_t nFeaturesToDrop) {

  Int nPlanes = input_features.size(1) - nFeaturesToDrop;
  const auto &_rules =
      m.getRuleBook(inputSize, outputSize, poolSize, poolStride, true);
  d_input_features.resize_as_(input_features);
  d_input_features.zero_();

  auto iF = input_features.data_ptr<T>();
  auto oF = output_features.data_ptr<T>();
  auto diF = d_input_features.data_ptr<T>();
  auto doF = d_output_features.data_ptr<T>();

  for (auto &r : _rules) {
    Int nHot = r.size() / 2;
    MaxPooling_BackwardPass<T>(iF, diF, oF, doF, nPlanes,
                               input_features.stride(0),
                               output_features.stride(0), &r[0], nHot);
  }
}
template <typename T, Int Dimension>
void cpu_RandomizedStrideMaxPooling_updateOutput(
    /*int64_t*/ at::Tensor &inputSize, /*int64_t*/ at::Tensor &outputSize,
    /*int64_t*/ at::Tensor &poolSize,
    /*int64_t*/ at::Tensor &poolStride, Metadata<Dimension> &m,
    /*float*/ at::Tensor &input_features,
    /*float*/ at::Tensor &output_features, int64_t nFeaturesToDrop) {

  Int nPlanes = input_features.size(1) - nFeaturesToDrop;
  const auto &_rules = m.getRandomizedStrideRuleBook(inputSize, outputSize, poolSize,
                                              poolStride, true);
  Int nActive = m.getNActive(outputSize);
  output_features.resize_({nActive, input_features.size(1) - nFeaturesToDrop});
  output_features.zero_();

  auto iF = input_features.data_ptr<T>() + nFeaturesToDrop;
  auto oF = output_features.data_ptr<T>();

  for (auto &r : _rules) {
    Int nHot = r.size() / 2;
    MaxPooling_ForwardPass<T>(iF, oF, nPlanes, input_features.stride(0),
                              output_features.stride(0), &r[0], nHot);
  }
}
template <typename T, Int Dimension>
void cpu_RandomizedStrideMaxPooling_updateGradInput(
    /*int64_t*/ at::Tensor &inputSize, /*int64_t*/ at::Tensor &outputSize,
    /*int64_t*/ at::Tensor &poolSize,
    /*int64_t*/ at::Tensor &poolStride, Metadata<Dimension> &m,
    /*float*/ at::Tensor &input_features,
    /*float*/ at::Tensor &d_input_features,
    /*float*/ at::Tensor &output_features,
    /*float*/ at::Tensor &d_output_features, int64_t nFeaturesToDrop) {

  Int nPlanes = input_features.size(1) - nFeaturesToDrop;
  const auto &_rules = m.getRandomizedStrideRuleBook(inputSize, outputSize, poolSize,
                                              poolStride, true);
  d_input_features.resize_as_(input_features);
  d_input_features.zero_();

  auto iF = input_features.data_ptr<T>();
  auto oF = output_features.data_ptr<T>();
  auto diF = d_input_features.data_ptr<T>();
  auto doF = d_output_features.data_ptr<T>();

  for (auto &r : _rules) {
    Int nHot = r.size() / 2;
    MaxPooling_BackwardPass<T>(iF, diF, oF, doF, nPlanes,
                               input_features.stride(0),
                               output_features.stride(0), &r[0], nHot);
  }
}
