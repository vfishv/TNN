#include "activation.inc"
#include "base.inc"
#include "io.inc"

//#define PRINT(tag, data) printf("%s %f %f %f %f\n", tag, data.x, data.y, data.z, data.w)

__kernel void Convolution3D(GLOBAL_SIZE_2_DIMS __read_only image2d_t input, __read_only image2d_t weights,
                            __read_only image2d_t bias, __write_only image2d_t output, __private const int3 input_whd,
                            __private const int in_channel_block_length, __private const int3 output_whd,
                            __private const int3 kernel_whd, __private const int3 stride_whd,
                            __private const int3 padding_whd, __private const int3 dilation_whd,
                            __private const int activation_type) {
    const int output_cw_idx  = get_global_id(0);
    const int output_bdh_idx = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(output_cw_idx, output_bdh_idx);

    const int input_dxh    = input_whd.z * input_whd.y;
    const int output_dxh   = output_whd.z * output_whd.y;
    const int kernel_dxh   = kernel_whd.z * kernel_whd.y;
    const int kernel_dxhx2 = kernel_dxh * 2;
    const int kernel_dxhx3 = kernel_dxh * 3;

    const int output_c_idx = output_cw_idx / output_whd.x;
    const int output_w_idx = output_cw_idx % output_whd.x;
    const int output_b_idx = output_bdh_idx / output_dxh;
    const int output_d_idx = (output_bdh_idx % output_dxh) / output_whd.y;
    const int output_h_idx = (output_bdh_idx % output_dxh) % output_whd.y;

    const int input_b_idx = output_b_idx;

    FLOAT4 one = (FLOAT4)(1.f);
    FLOAT4 in;
    FLOAT4 weight0, weight1, weight2, weight3;
    FLOAT4 out = RI_F(bias, SAMPLER, (int2)(output_c_idx, 0));

    const int input_d_start = output_d_idx * stride_whd.z - padding_whd.z;
    const int input_h_start = output_h_idx * stride_whd.y - padding_whd.y;
    const int input_w_start = output_w_idx * stride_whd.x - padding_whd.x;
    for (int input_c_idx = 0; input_c_idx < in_channel_block_length; input_c_idx++) {
        for (int kernel_d_idx = 0; kernel_d_idx < kernel_whd.z; kernel_d_idx++) {
            const int input_d_idx = input_d_start + kernel_d_idx * dilation_whd.z;
            if (input_d_idx < 0 || input_d_idx >= input_whd.z) {
                continue;
            }
            for (int kernel_h_idx = 0; kernel_h_idx < kernel_whd.y; kernel_h_idx++) {
                const int input_h_idx = input_h_start + kernel_h_idx * dilation_whd.y;
                if (input_h_idx < 0 || input_h_idx >= input_whd.y) {
                    continue;
                }
                for (int kernel_w_idx = 0; kernel_w_idx < kernel_whd.x; kernel_w_idx++) {
                    const int input_w_idx = input_w_start + kernel_w_idx * dilation_whd.x;
                    if (input_w_idx < 0 || input_w_idx >= input_whd.x) {
                        continue;
                    }

                    const int input_x_idx = input_c_idx * input_whd.x + input_w_idx;
                    const int input_y_idx = input_b_idx * input_dxh + input_d_idx * input_whd.y + input_h_idx;
                    in                    = RI_F(input, SAMPLER, (int2)(input_x_idx, input_y_idx));

                    const int weight_x_idx = input_c_idx * kernel_whd.x + kernel_w_idx;
                    const int weight_y_idx =
                        (output_c_idx << 2) * kernel_dxh + kernel_d_idx * kernel_whd.y + kernel_h_idx;
                    weight0 = RI_F(weights, SAMPLER, (int2)(weight_x_idx, weight_y_idx));
                    weight1 = RI_F(weights, SAMPLER, (int2)(weight_x_idx, weight_y_idx + kernel_dxh));
                    weight2 = RI_F(weights, SAMPLER, (int2)(weight_x_idx, weight_y_idx + kernel_dxhx2));
                    weight3 = RI_F(weights, SAMPLER, (int2)(weight_x_idx, weight_y_idx + kernel_dxhx3));

                    out.x += dot(in * weight0, one);
                    out.y += dot(in * weight1, one);
                    out.z += dot(in * weight2, one);
                    out.w += dot(in * weight3, one);
                }
            }
        }
    }

    out = ActivationProcess(out, activation_type);

    WI_F(output, (int2)(output_cw_idx, output_bdh_idx), out);
}
