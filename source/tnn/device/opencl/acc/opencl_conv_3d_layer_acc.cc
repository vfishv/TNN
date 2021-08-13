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
#include "tnn/device/opencl/acc/opencl_reshape_layer_acc.h"
#include "tnn/device/opencl/imagebuffer_convertor.h"
#include "tnn/device/opencl/opencl_memory.h"
#include "tnn/utils/dims_utils.h"

namespace TNN_NS {

class OpenCLConvolution3DLayerAcc : public OpenCLLayerAcc {
public:
    virtual Status Init(Context *context, LayerParam *param, LayerResource *resource, const std::vector<Blob *> &inputs,
                        const std::vector<Blob *> &outputs) override;

    virtual ~OpenCLConvolution3DLayerAcc() override;

    virtual Status Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) override;

private:
    Status ConvertWeights(float *weights_data_ptr);

private:
    std::shared_ptr<OpenCLMemory> ocl_weights_ = nullptr;
    std::shared_ptr<OpenCLMemory> ocl_bias_    = nullptr;
};

Status OpenCLConvolution3DLayerAcc::Init(Context *context, LayerParam *param, LayerResource *resource,
                                         const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Init Convolution3D Acc\n");
    Status ret = OpenCLLayerAcc::Init(context, param, resource, inputs, outputs);
    CHECK_TNN_OK(ret)

    run_3d_ndrange_ = false;
    op_name_        = "Convolution3D";

    ConvLayerParam *conv3d_param = dynamic_cast<ConvLayerParam *>(param);
    CHECK_PARAM_NULL(conv3d_param);

    ConvLayerResource *conv3d_resource = dynamic_cast<ConvLayerResource *>(resource);
    CHECK_PARAM_NULL(conv3d_resource);
    RawBuffer &filter_handle = conv3d_resource->filter_handle;
    RawBuffer &bias_handle   = conv3d_resource->bias_handle;

    // get weights
    if (filter_handle.GetDataType() == DATA_TYPE_FLOAT) {
        // get float pointer from raw buffer.
        float *weights_data_ptr = filter_handle.force_to<float *>();
        if (weights_data_ptr == nullptr) {
            return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "pointer is null");
        }
        ret = ConvertWeights(weights_data_ptr);
        CHECK_TNN_OK(ret)
    } else {
        // if handle is half, need convert to float first.
        auto float_data_ptr = GetFloatFromRawBuffer(filter_handle);
        if (float_data_ptr == nullptr) {
            return Status(TNNERR_OPENCL_ACC_INIT_ERROR, "pointer is null");
        }
        ret = ConvertWeights(float_data_ptr.get());
        CHECK_TNN_OK(ret)
    }

    // get bias
    ret = ConvertChannelWeights(bias_handle, ocl_bias_, conv3d_param->output_channel, conv3d_param->bias);
    CHECK_TNN_OK(ret)

    // create kernel
    std::string program_name = "convolution_3d";
    std::string kernel_name  = "Convolution3D";
    ret                      = CreateExecuteUnit(execute_units_[0], program_name, kernel_name);
    if (ret != TNN_OK) {
        LOGE("create execute unit failed!\n");
        return ret;
    }

    return TNN_OK;
}

OpenCLConvolution3DLayerAcc::~OpenCLConvolution3DLayerAcc() {}

Status OpenCLConvolution3DLayerAcc::Reshape(const std::vector<Blob *> &inputs, const std::vector<Blob *> &outputs) {
    LOGD("Conv3D Acc Reshape\n");
    Status ret = OpenCLLayerAcc::Reshape(inputs, outputs);
    CHECK_TNN_OK(ret)

    ConvLayerParam *conv3d_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv3d_param);

    const DimsVector &input_dims  = inputs[0]->GetBlobDesc().dims;
    const DimsVector &output_dims = outputs[0]->GetBlobDesc().dims;

    const int input_channel_blocks = UP_DIV(conv3d_param->input_channel, 4);
    const DimsVector input_whd     = {input_dims[4], input_dims[3], input_dims[2]};
    const DimsVector output_whd    = {output_dims[4], output_dims[3], output_dims[2]};
    const DimsVector padding_whd   = {conv3d_param->pads[0], conv3d_param->pads[2], conv3d_param->pads[4]};

    uint32_t idx = SetExecuteUnit2DSizeInfoDefault(execute_units_[0], output_dims);
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)inputs[0]->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_weights_->GetData()));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)ocl_bias_->GetData()));
    execute_units_[0].ocl_kernel.setArg(idx++, *((cl::Image *)outputs[0]->GetHandle().base));
    execute_units_[0].ocl_kernel.setArg(idx++, 3 * sizeof(int), input_whd.data());
    execute_units_[0].ocl_kernel.setArg(idx++, input_channel_blocks);
    execute_units_[0].ocl_kernel.setArg(idx++, 3 * sizeof(int), output_whd.data());
    execute_units_[0].ocl_kernel.setArg(idx++, 3 * sizeof(int), conv3d_param->kernels.data());
    execute_units_[0].ocl_kernel.setArg(idx++, 3 * sizeof(int), conv3d_param->strides.data());
    execute_units_[0].ocl_kernel.setArg(idx++, 3 * sizeof(int), padding_whd.data());
    execute_units_[0].ocl_kernel.setArg(idx++, 3 * sizeof(int), conv3d_param->dialations.data());
    execute_units_[0].ocl_kernel.setArg(idx++, (int)conv3d_param->activation_type);

    return TNN_OK;
}

Status OpenCLConvolution3DLayerAcc::ConvertWeights(float *weights_data_ptr) {
    OpenCLRuntime *opencl_runtime = OpenCLRuntime::GetInstance();

    ConvLayerParam *conv3d_param = dynamic_cast<ConvLayerParam *>(param_);
    CHECK_PARAM_NULL(conv3d_param);

    const int input_channel  = conv3d_param->input_channel;
    const int output_channel = conv3d_param->output_channel;
    const int kernel_size_x  = conv3d_param->kernels[0];
    const int kernel_size_y  = conv3d_param->kernels[1];
    const int kernel_size_d  = conv3d_param->kernels[2];

    // copy weights data into clBuffer
    DimsVector weight_shape = {output_channel, input_channel, kernel_size_d, kernel_size_y, kernel_size_x};
    const int buffer_size = output_channel * ROUND_UP(input_channel, 4) * kernel_size_d * kernel_size_y * kernel_size_x;
    const int weight_size = DimsVectorUtils::Count(weight_shape);
    shared_ptr<OpenCLMemory> weight_buffer(new OpenCLMemory(TNN_CL_BUFFER));
    cl_int ret = CL_SUCCESS;
    cl::Buffer buffer(*opencl_runtime->Context(), CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, buffer_size * sizeof(float),
                      nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL malloc memory failed");
    }
    weight_buffer->SetData(&buffer);
    auto weight_clbuffer_ptr = ocl_context_->CommandQueue()->enqueueMapBuffer(
        buffer, true, CL_MAP_WRITE, 0, buffer_size * sizeof(float), nullptr, nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMMAP_ERROR, "OpenCL MemMap failed");
    }
    memcpy(weight_clbuffer_ptr, weights_data_ptr, weight_size * sizeof(float));
    ret = ocl_context_->CommandQueue()->enqueueUnmapMemObject(buffer, weight_clbuffer_ptr);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        return Status(TNNERR_OPENCL_MEMUNMAP_ERROR, "OpenCL MemUnMap failed");
    }

    // create ocl_weights_
    DimsVector weight_imageshape{(int)(UP_DIV(input_channel, 4) * kernel_size_x),
                                 output_channel * kernel_size_d * kernel_size_y};
    cl_channel_type data_type = CL_FLOAT;
    if (opencl_runtime->GetPrecision() != PRECISION_HIGH)
        data_type = CL_HALF_FLOAT;
    cl::Image2D *image =
        new cl::Image2D(*opencl_runtime->Context(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_RGBA, data_type),
                        weight_imageshape[0], weight_imageshape[1], 0, nullptr, &ret);
    if (ret != CL_SUCCESS) {
        CHECK_CL_SUCCESS(ret)
        if (nullptr != image)
            delete image;
        return Status(TNNERR_OPENCL_MEMALLOC_ERROR, "OpenCL malloc memory failed");
    }
    ocl_weights_.reset(new OpenCLMemory(TNN_CL_IMAGE));
    ocl_weights_->SetData(image, true);

    // transfer from clBuffer to clImage
    ImageBufferConvertor convertor(opencl_runtime, ocl_context_->CommandQueue());
    return convertor.ConvertBufferToImage(weight_buffer.get(), NCHW_BUFFER, weight_shape, ocl_weights_.get(), true);
}

REGISTER_OPENCL_ACC(Convolution3D, LAYER_CONVOLUTION_3D)
REGISTER_OPENCL_LAYOUT(LAYER_CONVOLUTION_3D, DATA_FORMAT_NHC4W4);

}  // namespace TNN_NS
