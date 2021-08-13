#include "base.inc"

__kernel void Pooling3D(GLOBAL_SIZE_2_DIMS __read_only image2d_t input, __write_only image2d_t output,
                        __private const int3 input_whd, __private const int3 output_whd,
                        __private const int3 kernel_whd, __private const int3 stride_whd,
                        __private const int3 pad_whd) {
    const int output_cw_idx  = get_global_id(0);
    const int output_bdh_idx = get_global_id(1);
    DEAL_NON_UNIFORM_DIM2(output_cw_idx, output_bdh_idx);

    const int input_dxh  = input_whd.z * input_whd.y;
    const int output_dxh = output_whd.z * output_whd.y;

    const int output_c_idx = output_cw_idx / output_whd.x;
    const int output_w_idx = output_cw_idx % output_whd.x;
    const int output_b_idx = output_bdh_idx / output_dxh;
    const int output_d_idx = (output_bdh_idx % output_dxh) / output_whd.y;
    const int output_h_idx = (output_bdh_idx % output_dxh) % output_whd.y;

    const int input_b_idx = output_b_idx;
    const int input_c_idx = output_c_idx;

    int input_d_start = output_d_idx * stride_whd.z - pad_whd.z;
    int input_h_start = output_h_idx * stride_whd.y - pad_whd.y;
    int input_w_start = output_w_idx * stride_whd.x - pad_whd.x;

    const int input_d_end = min(input_d_start + kernel_whd.z, input_whd.z);
    const int input_h_end = min(input_h_start + kernel_whd.y, input_whd.y);
    const int input_w_end = min(input_w_start + kernel_whd.x, input_whd.x);

    input_d_start = max(input_d_start, 0);
    input_h_start = max(input_h_start, 0);
    input_w_start = max(input_w_start, 0);

    FLOAT4 in;

#ifdef POOL_AVG
    FLOAT4 output_result = (FLOAT4)0;
    const int kernel_count =
        (input_d_end - input_d_start) * (input_h_end - input_h_start) * (input_w_end - input_w_start);
    for (int input_d_idx = input_d_start; input_d_idx < input_d_end; input_d_idx++) {
        for (int input_h_idx = input_h_start; input_h_idx < input_h_end; input_h_idx++) {
            for (int input_w_idx = input_w_start; input_w_idx < input_w_end; input_w_idx++) {
                const int input_x_idx = input_c_idx * input_whd.x + input_w_idx;
                const int input_y_idx = input_b_idx * input_dxh + input_d_idx * input_whd.y + input_h_idx;
                in                    = RI_F(input, SAMPLER, (int2)(input_x_idx, input_y_idx));
                output_result += in;
            }
        }
    }
    output_result /= (float)kernel_count;
#else
    FLOAT4 output_result = (FLOAT4)(-FLT_MAX);
    for (int input_d_idx = input_d_start; input_d_idx < input_d_end; input_d_idx++) {
        for (int input_h_idx = input_h_start; input_h_idx < input_h_end; input_h_idx++) {
            for (int input_w_idx = input_w_start; input_w_idx < input_w_end; input_w_idx++) {
                const int input_x_idx = input_c_idx * input_whd.x + input_w_idx;
                const int input_y_idx = input_b_idx * input_dxh + input_d_idx * input_whd.y + input_h_idx;
                in                    = RI_F(input, SAMPLER, (int2)(input_x_idx, input_y_idx));
                output_result         = fmax(output_result, in);
            }
        }
    }
#endif
    WI_F(output, (int2)(output_cw_idx, output_bdh_idx), output_result);
}
