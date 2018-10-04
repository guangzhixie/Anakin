#include <stdio.h>
#include "bm_common.h"
#include "atomic_dma_gen_cmd.h"
#include "atomic_pooling_gen_cmd.h"

int bm_pooling_fwd(bm_api_pooling_forward pooling_param){
    // Unpack parameters
    u64 ifmap_offset_global = pooling_param.ifmap_offset_global;
    u64 ofmap_offset_global = pooling_param.ofmap_offset_global;
    int input_n = pooling_param.input_n;
    int input_c = pooling_param.input_c;
    int input_h = pooling_param.input_h;
    int input_w = pooling_param.input_w;
    int kh = pooling_param.kh;
    int kw = pooling_param.kw;
    int pad_h = pooling_param.pad_h;
    int pad_w = pooling_param.pad_w;
    int stride_h = pooling_param.stride_h;
    int stride_w = pooling_param.stride_w;
    int is_avg_pooling = pooling_param.is_avg_pooling;
    int n_step = pooling_param.n_step;
    int h_step = pooling_param.h_step;

    P_COMMAND dma_command;

    int top_pad_h = pad_h;
    int bottom_pad_h = pad_h;
    int left_pad_w = pad_w;
    int right_pad_w = pad_w;

    int output_h = (input_h + 2 * pad_h - kh) / stride_h + 1;
    int output_w = (input_w + 2 * pad_w - kw) / stride_w + 1;
    //check if have one more output h and w
    if(is_avg_pooling ==0 && (output_h*stride_h < input_h+pad_h) && ((input_h + 2 * pad_h - kh)%stride_h !=0 )) {
        output_h++;
        bottom_pad_h += (stride_h - (input_h + 2*pad_h - kh)%stride_h);
    }
    if(is_avg_pooling ==0 &&(output_w*stride_w < input_w+pad_w) && ((input_w + 2 * pad_w - kw)%stride_w !=0 )) {
        output_w++;
        right_pad_w += (stride_w - (input_w + 2 * pad_w - kw)%stride_w);
    }
    // need to update
    int ifmap_offset_local = 0;
    int ofmap_offset_local = 0;
    int offset_local_end = 0;

    int ifmap_align_size = 0;
    int ofmap_align_size = 0;

    CMD_ID_NODE id_node;
    resync_cmd_id( &id_node );

    int n_start = 0;
    int h_start = 0;
    int n_slice = 0;
    int h_slice = 0;
    int new_top_pad_h = 0;
    int new_bottom_pad_h = 0;
    int input_h_start = 0;
    int output_h_start = 0;
    int h_end;
    int new_output_h;
    for(n_start = 0; n_start < input_n; n_start += n_step) { 
        //calculate the n_slice, n_slice is the number of pictures processed one time
        if(n_start + n_step - 1 < input_n) {
        n_slice = n_step;
        } else {
        n_slice = input_n - n_start;
        }
        for(h_start = 0-top_pad_h; h_start < input_h+bottom_pad_h-kh+1; h_start += h_step) {
        input_h_start = (h_start < 0) ? 0 : h_start;
        output_h_start = (h_start + top_pad_h)/stride_h;
        new_top_pad_h = 0 - h_start;
        if(new_top_pad_h < 0) new_top_pad_h = 0;
        // calculate h_slice, h_slice is the number of feature_map_height processed one time
        h_end = h_start + h_step - stride_h + kh - 1;
        if(h_end < input_h) {
            h_slice = h_end - input_h_start + 1;
            new_bottom_pad_h = 0;
        } else if(h_end >= input_h && h_end < input_h + bottom_pad_h) {
            h_slice = input_h - input_h_start;
            new_bottom_pad_h = h_end - input_h + 1;
        } else {
            h_slice = input_h - input_h_start;
            new_bottom_pad_h = bottom_pad_h;
        }

        //Now, we can call the DMA and pooling atomic operation
        ifmap_align_size = n_slice * (ceiling_func_shift(input_c,NPU_SHIFT)) *
                            get_neuron_csize_local(h_slice, input_w);

        new_output_h = (h_slice + new_top_pad_h + new_bottom_pad_h -kh) / stride_h + 1;
        ofmap_align_size = n_slice * (ceiling_func_shift(input_c,NPU_SHIFT)) *
                            get_neuron_csize_local(new_output_h, output_w);

        // 1.2 get the local memory address offset
        ofmap_offset_local = ifmap_align_size;
        offset_local_end = ofmap_offset_local + ofmap_align_size;
        ASSERT(offset_local_end <= LOCAL_MEM_SIZE);

        //Load ifmap from global mem to local mem
        dma_command = get_command(ENGINE_GDMA);
        tensor_stride_move_gen_cmd(
                ifmap_offset_local,                       // dst local mem start address
                ifmap_offset_global + n_start*input_c*input_h*input_w*FLOAT_SIZE
                                    + input_h_start*input_w*FLOAT_SIZE, // src tensor address
                n_slice, input_c, h_slice, input_w,          // src N/C/H/W
                0,                                        // dst local mem idx
                0,                                        // sys2loc
                input_c * input_h * input_w,              // src N/C/H stride
                input_h * input_w,
                input_w,
                (ceiling_func_shift(input_c,NPU_SHIFT)) * get_cstride_local(h_slice, input_w),//dst N/C/H stride
                get_cstride_local(h_slice, input_w),
                input_w,
                GDMA_TYPE_f32,
                0,
                (void*)dma_command,
                &id_node);
            call_atomic(nodechip_idx, atomic_global_dma, dma_command, ENGINE_GDMA);
            // 3 gen task cmdripter
            P_COMMAND pooling_command = get_command(ENGINE_BD);
            atomic_pooling_gen_cmd(
                    pooling_command,
                    //Here the ifmap_offset_local is the offset in local memory 0
                    //we need to translate it into the absolute addr in address map
                    //of our SOC
                    LOCAL_MEM_START_ADDR | ifmap_offset_local,
                    LOCAL_MEM_START_ADDR | ofmap_offset_local,
                    n_slice, input_c, h_slice, input_w,
                    new_output_h, output_w, kh, kw,
                    new_top_pad_h, new_bottom_pad_h, left_pad_w, right_pad_w, stride_h, stride_w,
                    0,0,0,0,    //ins0 parameters
                    is_avg_pooling,
                    1.0f / (kw * kh),
                    &id_node);
            call_atomic(nodechip_idx, atomic_pooling, pooling_command, ENGINE_BD);
            
            dma_command = get_command(ENGINE_GDMA);
            tensor_stride_move_gen_cmd(
                ofmap_offset_local,                    // src local mem start address
                ofmap_offset_global + n_start*input_c*output_h*output_w*FLOAT_SIZE
                                        + output_h_start*output_w*FLOAT_SIZE, // dst tensor address
                n_slice, input_c, new_output_h, output_w,        // src N/C/H/W
                0,
                1,                                         // loc2sys
                (ceiling_func_shift(input_c,NPU_SHIFT)) * get_cstride_local(new_output_h, output_w), // src N/C/H stride
                get_cstride_local(new_output_h , output_w),
                output_w,
                input_c * output_h * output_w, // dst N/C/H stride ? align
                output_h * output_w,
                output_w,
                GDMA_TYPE_f32,
                0,
                (void*)dma_command,
                &id_node);
            call_atomic(nodechip_idx, atomic_global_dma, dma_command, ENGINE_GDMA);
        }
    }
    poll_all_engine_done(&id_node);

    return 0;
}