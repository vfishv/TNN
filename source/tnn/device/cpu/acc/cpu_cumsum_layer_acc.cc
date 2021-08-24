// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "cpu_layer_acc.h"
#include "tnn/utils/data_type_utils.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

DECLARE_CPU_ACC(CumSum, LAYER_CUMSUM);

Status CpuCumSumLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    return TNN_OK;
}

Status CpuCumSumLayerAcc::Forward(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    const auto input_blob  = inputs[0];
    const auto output_blob = outputs[0];
    const auto input_dims  = input_blob->GetBlobDesc().dims;
    const auto output_dims = output_blob->GetBlobDesc().dims;

    auto layer_param = reinterpret_cast<CumSumLayerParam *>(param_);
    CHECK_PARAM_NULL(layer_param);

    const int exclusive    = layer_param->exclusive;
    const int reverse      = layer_param->reverse;
    const int axis         = layer_param->axis[0];
    const int outer_count  = DimsVectorUtils::Count(input_dims, 0, axis);
    const int cumsum_count = DimsFunctionUtils::GetDim(input_dims, axis);
    const int inner_count  = DimsVectorUtils::Count(input_dims, axis + 1);

    if (output_blob->GetBlobDesc().data_type == DATA_TYPE_FLOAT) {
        float *input_ptr  = static_cast<float *>(input_blob->GetHandle().base);
        float *output_ptr = static_cast<float *>(output_blob->GetHandle().base);
        if (!reverse) {
            for (int out_idx = 0; out_idx < outer_count; out_idx++) {
                int index = out_idx * cumsum_count * inner_count;
                for (int inner_idx = 0; inner_idx < inner_count; inner_idx++) {
                    output_ptr[index + inner_idx] = exclusive == 0 ? input_ptr[index + inner_idx] : 0;
                }
            }

            /*
             * if exclusive == 0
             * output_ptr[out][cumsum][inner] = output_ptr[out][cumsum - 1][inner] + input_ptr[out][cumsum][inner]
             *
             * if exclusive != 0
             * output_ptr[out][cumsum][inner] = output_ptr[out][cumsum - 1][inner] + input_ptr[out][cumsum - 1][inner]
             */
            const int offset = exclusive == 0 ? 0 : 1;
            for (int out_idx = 0; out_idx < outer_count; out_idx++) {
                for (int cumsum_idx = 1; cumsum_idx < cumsum_count; cumsum_idx++) {
                    int index = (out_idx * cumsum_count + cumsum_idx) * inner_count;
                    for (int inner_idx = 0; inner_idx < inner_count; inner_idx++) {
                        int index_base = index + inner_idx;
                        output_ptr[index_base] =
                            output_ptr[index_base - inner_count] + input_ptr[index_base - offset * inner_count];
                    }
                }
            }
        } else {
            for (int out_idx = 0; out_idx < outer_count; out_idx++) {
                int index = (out_idx * cumsum_count + cumsum_count - 1) * inner_count;
                for (int inner_idx = 0; inner_idx < inner_count; inner_idx++) {
                    output_ptr[index + inner_idx] = exclusive == 0 ? input_ptr[index + inner_idx] : 0;
                }
            }
            const int offset = exclusive == 0 ? 0 : 1;
            for (int out_idx = 0; out_idx < outer_count; out_idx++) {
                for (int cumsum_idx = cumsum_count - 1; cumsum_idx > 0; cumsum_idx--) {
                    int index = (out_idx * cumsum_count + cumsum_idx) * inner_count;
                    for (int inner_idx = 0; inner_idx < inner_count; inner_idx++) {
                        int index_base = index + inner_idx;
                        output_ptr[index_base] =
                            output_ptr[index_base + inner_count] + input_ptr[index_base + offset * inner_count];
                    }
                }
            }
        }
    } else {
        LOGE("Error: CpuCumSumLayerAcc layer acc dont support datatype: %d\n", output_blob->GetBlobDesc().data_type);
        return Status(TNNERR_MODEL_ERR, "Error: CpuCumSumLayerAcc layer acc dont support datatype");
    }

    return TNN_OK;
}

REGISTER_CPU_ACC(CumSum, LAYER_CUMSUM);
}  // namespace TNN_NS
