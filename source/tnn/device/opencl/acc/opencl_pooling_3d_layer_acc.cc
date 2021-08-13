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

#include "tnn/device/opencl/acc/opencl_layer_acc.h"
#include "tnn/device/opencl/imagebuffer_convertor.h"
#include "tnn/utils/string_utils_inner.h"

namespace TNN_NS {

DECLARE_OPENCL_ACC(Pooling3D);

Status OpenCLPooling3DLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                     const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Pooling Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    run_3d_ndrange_ = false;
    op_name_        = "Pooling3D";

    PoolingLayerParam *pooling3d_param = dynamic_cast<PoolingLayerParam *>(param);
    CHECK_PARAM_NULL(pooling3d_param);

    if (pooling3d_param->pad_type == 1) {  // VALID Type
        pooling3d_param->pads[0] = 0;
        pooling3d_param->pads[2] = 0;
    }

    // create kernel
    std::set<std::string> build_options;
    std::string program_name = "pooling_3d";
    std::string kernel_name  = "Pooling3D";

    if (pooling3d_param->pool_type != 0) {
        // 0:max_pooling  other:average pooling
        build_options.emplace("-DPOOL_AVG");
    }
    ret = CreateExecuteUnit(execute_units_[0], program_name, kernel_name, build_options);
    if (ret != TNN_OK) {
        LOGE("create execute unit failed!\n");
        return ret;
    }

    return TNN_OK;
}

Status OpenCLPooling3DLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Pooling Acc Reshape\n");
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret)

    PoolingLayerParam *pooling3d_param = dynamic_cast<PoolingLayerParam *>(param_);
    CHECK_PARAM_NULL(pooling3d_param);

    const DimsVector &input_dims  = inputs[0]->GetBlobDesc().dims;
    const DimsVector &output_dims = outputs[0]->GetBlobDesc().dims;
    const DimsVector input_whd    = {input_dims[4], input_dims[3], input_dims[2]};
    const DimsVector output_whd   = {output_dims[4], output_dims[3], output_dims[2]};
    const DimsVector pad_whd      = {pooling3d_param->pads[0], pooling3d_param->pads[2], pooling3d_param->pads[4]};

    uint32_t idx = SetExecuteUnit2DSizeInfoDefault(execute_units_[0], output_dims);
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)outputs[0]->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, 3 * sizeof(int), input_whd.data());
    execute_units_[0].ocl_kernel.setArg(idx++, 3 * sizeof(int), output_whd.data());
    execute_units_[0].ocl_kernel.setArg(idx++, 3 * sizeof(int), pooling3d_param->kernels.data());
    execute_units_[0].ocl_kernel.setArg(idx++, 3 * sizeof(int), pooling3d_param->strides.data());
    execute_units_[0].ocl_kernel.setArg(idx++, 3 * sizeof(int), pad_whd.data());

    return TNN_OK;
}

REGISTER_OPENCL_ACC(Pooling3D, LAYER_POOLING_3D)
REGISTER_OPENCL_LAYOUT(LAYER_POOLING_3D, DATA_FORMAT_NHC4W4);

}  // namespace TNN_NS
