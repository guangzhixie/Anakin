
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
  
template <>
SaberStatus VenderPooling<BM, AK_FLOAT>::\
    dispatch(const std::vector<DataTensor_in*>& inputs,
                    std::vector<DataTensor_out*>& outputs,
                    PoolingParam<BM> &param) {
            
        const BM_mem_addr in_data = (const BM_mem_addr) inputs[0]->data();
        BM_mem_addr out_data = (BM_mem_addr) outputs[0]->mutable_data();
            
            
        return SaberSuccess;
}
DEFINE_OP_TEMPLATE(VenderPooling, PoolingParam, BM, AK_HALF);
DEFINE_OP_TEMPLATE(VenderPooling, PoolingParam, BM, AK_INT8);
} //namespace saber
} // namespace anakin
