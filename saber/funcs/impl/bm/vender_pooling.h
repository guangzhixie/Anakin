#ifndef ANAKIN_SABER_FUNCS_IMPL_BM_POOLING_H
#define ANAKIN_SABER_FUNCS_IMPL_BM_POOLING_H

#include "saber/funcs/impl/impl_pooling.h"

namespace anakin{

namespace saber {

template <DataType OpDtype>
class VenderPooling<BM, OpDtype>:\
 public ImplBase<
    BM,
    OpDtype,
    PoolingParam<BM>> {
    public:
        typedef Tensor<BM> DataTensor_in;
        typedef Tensor<BM> DataTensor_out;
        typedef Tensor<BM> OpTensor;
        
        VenderPooling() : _handle(NULL) {}
        
        ~VenderPooling() {
        }
        
        virtual SaberStatus init(const std::vector<DataTensor_in*>& inputs,
                                 std::vector<DataTensor_out*>& outputs,
                                 PoolingParam<BM> &pooling_param, Context<BM> &ctx);
        
        virtual SaberStatus create(const std::vector<DataTensor_in*>& inputs,
                                   std::vector<DataTensor_out*>& outputs,
                                   PoolingParam<BM> &pooling_param, Context<BM> &ctx);
        
        virtual SaberStatus dispatch(const std::vector<DataTensor_in*>& inputs,
                                     std::vector<DataTensor_out*>& outputs,
                                     PoolingParam<BM> &param);
        
    private:
        bm_handle_t _handle;
        
};

} //namespace saber

} // namespace anakin

#endif //ANAKIN_SABER_FUNCS_IMPL_BM_POOLING_H
