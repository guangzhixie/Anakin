
#include "saber/funcs/impl/bm/vender_pooling.h"
#include "saber/funcs/impl/impl_pooling.h"

namespace anakin{
namespace saber {
    
template class VenderPooling<BM, AK_FLOAT>;
    
template <>
SaberStatus VenderPooling<BM, AK_FLOAT>::\
    create(const std::vector<DataTensor_in*>& inputs,
           std::vector<DataTensor_out*>& outputs,
           PoolingParam<BM> &pooling_param, Context<BM> &ctx) {

}
    
template<>
SaberStatus VenderPooling<BM, AK_FLOAT> :: \
    init(const std::vector<DataTensor_in*>& inputs,
         std::vector<DataTensor_out*>& outputs,
         PoolingParam<BM> &pooling_param, Context<BM> &ctx) {
            
        _handle = ctx.get_handle();
        return create(inputs, outputs, pooling_param, ctx);
}

int nodechip_pooling_getLineNum(
    float           percentOneFeatureMem,
    int             input_c,
    int             input_h,
    int             input_w,
    int             kh,
    int             kw,
    int             pad_h,
    int             pad_w,
    int             stride_h,
    int             stride_w,
    int             is_avg_pooling
    )
{
    int N_tmp = (int)(percentOneFeatureMem * input_h * 0.95);
    int output_h = (N_tmp - kh) / stride_h + 1;
    int N_initial = (output_h-1) * stride_h + kh;
    CHECK_GE(N_initial, kh) << "nodechip_pooling_getLineNum failed";

    return N_initial;
}

int nodechip_pooling_getMemRequirement(
    int             input_c,
    int             input_h,
    int             input_w,
    int             kh,
    int             kw,
    int             pad_h,
    int             pad_w,
    int             stride_h,
    int             stride_w,
    int             is_avg_pooling,
    int             is_lines
    )
{// one input feature map mem requirement
    int ifmap_align_size = get_neuron_csize_local(input_h, input_w);
    int ifmap_single_size = (ceiling_func_shift(input_c,NPU_SHIFT)) * ifmap_align_size;// corresponding output feature map mem requirement
    int output_h = 0;
    if (is_lines == 0) {
       output_h = (input_h + 2 * pad_h - kh) / stride_h + 1;
           if(is_avg_pooling ==0 && (output_h*stride_h < input_h+pad_h) && ((input_h + 2 * pad_h - kh)%stride_h !=0 )) output_h++;
        } else {
       output_h = (input_h + pad_h -kh) / stride_h + 1;
    }
    int output_w = (input_w + 2 * pad_w - kw) / stride_w + 1;
        if(is_avg_pooling ==0 && (output_w*stride_w < input_w+pad_w) && ((input_w + 2 * pad_w - kw)%stride_w !=0 )) output_w++;
    int ofmap_align_size = get_neuron_csize_local(output_h, output_w);
    int ofmap_single_size = (ceiling_func_shift(input_c,NPU_SHIFT)) * ofmap_align_size;
    int ofmap_single_offset_local = ifmap_single_size;
    int offset_single_local_end = 0;
    offset_single_local_end = ofmap_single_offset_local + ofmap_single_size;
    return offset_single_local_end;
}

static bm_status_t bmdnn_pooling_forward_runtime_check(
    int                 input_n,
    int                 input_c,
    int                 input_h,
    int                 input_w,
    int                 kh,
    int                 kw,
    int                 pad_h,
    int                 pad_w,
    int                 stride_h,
    int                 stride_w,
    int                 is_avg_pooling,
    pooling_secs_info_t *pooling_secs_info
    )
{
    if (pad_h > kh || pad_w > kw)
        return BM_NOT_SUPPORTED;
    int local_mem_banks = 4;
    int output_h = (input_h + 2 * pad_h - kh) / stride_h + 1;
    int output_w = (input_w + 2 * pad_w - kw) / stride_w + 1;
	//check if have one more output h and w
	if((output_h*stride_h < input_h+pad_h) && ((input_h + 2 * pad_h - kh)%stride_h !=0 )) output_h++;
	if((output_w*stride_w < input_w+pad_w) && ((input_w + 2 * pad_w - kw)%stride_w !=0 )) output_w++;
    //*********** get the single feature map size ******************************/
    int ifmap_single_size = get_neuron_csize_local(input_h, input_w);
    int ofmap_single_size = get_neuron_csize_local(output_h, output_w);
    int oneFeatureMem = 0;
    oneFeatureMem = ofmap_single_size;
    //********* for forward pooling, ordinary the ofmap size is smaller than ifmap
    if (oneFeatureMem < ifmap_single_size)
        oneFeatureMem = ifmap_single_size;
    float inRatio = 1.0/2.0;
    int local_mem_inputbanks = (int)(inRatio*local_mem_banks)/2;
    if (local_mem_inputbanks == 0)
         local_mem_inputbanks = 1;
    float percentOneFeatureMem = (float)LOCAL_MEM_SIZE / (float)oneFeatureMem*(float)local_mem_inputbanks/local_mem_banks;
    int oneFeatureMem_s = 0;
    oneFeatureMem_s  = nodechip_pooling_getMemRequirement(
                            input_c,
                            input_h,
                            input_w,
                            kh,
                            kw,
                            pad_h,
                            pad_w,
                            stride_h,
                            stride_w,
                            is_avg_pooling,
                            0
    );
    float percentOneFeatureMem_s = (float)LOCAL_MEM_SIZE / (float)oneFeatureMem_s;
    int n_step = 0;
    int h_step = 0;
    if(percentOneFeatureMem_s >= input_n) {
        n_step = input_n;
        h_step = output_h * stride_h;
    } else if(percentOneFeatureMem_s < input_n && percentOneFeatureMem_s >= 1) {
        n_step = (int)percentOneFeatureMem_s;
        h_step = output_h * stride_h;
    } else {
        n_step = 1;
        h_step = nodechip_pooling_getLineNum(
                        percentOneFeatureMem_s,
                        input_c,
                        input_h,
                        input_w,
                        kh,
                        kw,
                        pad_h,
                        pad_w,
                        stride_h,
                        stride_w,
                        is_avg_pooling
        );
        h_step = ((h_step - kh)/stride_h + 1) * stride_h;
    }
    pooling_secs_info->nsecs = n_step;
    pooling_secs_info->hsecs = h_step;

    if(percentOneFeatureMem < 1.0) {
        int N_lines = (int)(percentOneFeatureMem*input_h*0.98);
        int output_h_lines = (N_lines - kh) / stride_h + 1;
        int N_initial = (output_h_lines - 1)*stride_h + kh;
        if(N_initial< kh)
            return BM_NOT_SUPPORTED;
        else
            return BM_SUCCESS;
    }
    else {
        return BM_SUCCESS;
    }
}

template <>
SaberStatus VenderPooling<BM, AK_FLOAT>::\
    dispatch(const std::vector<DataTensor_in*>& inputs,
                    std::vector<DataTensor_out*>& outputs,
                    PoolingParam<BM> &param) {
            
        const BM_mem_addr in_data = (const BM_mem_addr) inputs[0]->data();
        BM_mem_addr out_data = (BM_mem_addr) outputs[0]->mutable_data();

        int input_n = inputs[0]->num();
        int input_c = inputs[0]->channel();
        int input_h = inputs[0]->height();
        int input_w = inputs[0]->width();
        int kh = param.window_h;
        int kw = param.window_w;
        int pad_h = param.pad_h;
        int pad_w = param.pad_w;
        int stride_h = param.stride_h;
        int stride_w = param.stride_w;
        int is_avg_pooling;
        if(param.pooling_type == Pooling_max){
            is_avg_pooling = 0;
        } else {
            is_avg_pooling = 1;
        }
    
        pooling_secs_info_t pooling_secs_info;

        bm_status_t result =  bmdnn_pooling_forward_runtime_check(
            input_n, input_c, input_h, input_w,
            kh, kw, pad_h, pad_w, stride_h,
            stride_w, is_avg_pooling,
            &pooling_secs_info
        );
        if (result == BM_NOT_SUPPORTED) {
            return result;
        }

        bm_device_mem_t input_mem, output_mem;
        int output_h = (input_h + 2 * pad_h - kh) / stride_h + 1;
        int output_w = (input_w + 2 * pad_w - kw) / stride_w + 1;
        //check if have one more output h and w
        if(is_avg_pooling ==0 && (output_h*stride_h < input_h+pad_h) && ((input_h + 2 * pad_h - kh)%stride_h !=0 )) output_h++;
        if(is_avg_pooling ==0 && (output_w*stride_w < input_w+pad_w) && ((input_w + 2 * pad_w - kw)%stride_w !=0 )) output_w++;
        
        bm_api_pooling_forward bm_pooling_param = {
            bm_mem_get_device_addr(in_data),
            bm_mem_get_device_addr(out_data),
            input_n,
            input_c,
            input_h,
            input_w,
            kh,
            kw,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            is_avg_pooling,
            pooling_secs_info.nsecs,
            pooling_secs_info.hsecs
        };

        bm_status_t bm_stat = bmlib_kernel_launch(_handle, "/usr/local/include/bm/bmkernel_bin.bin");
        CHECK_EQ(BM_SUCCESS, bm_stat) << "bmlib_kernel_launch failed.";
        
        /* Send arguments. */
        enum BmOpType op = POOLING;
        bmkernel_api_base api = { op, reinterpret_cast<void *>(&bm_pooling_param) };
        BM_CHECK(bmlib_kernel_send_args(_handle, reinterpret_cast<void *>(&api), sizeof(api)));

        return SaberSuccess;
}
DEFINE_OP_TEMPLATE(VenderPooling, PoolingParam, BM, AK_HALF);
DEFINE_OP_TEMPLATE(VenderPooling, PoolingParam, BM, AK_INT8);
} //namespace saber
} // namespace anakin
