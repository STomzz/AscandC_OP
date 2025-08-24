/**
 * @file reduce_custom.cpp
 *
 * Copyright (C) 2022-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"
#define REDUCE_TILING_1 1
#define REDUCE_TILING_2 2
#define REDUCE_TILING_3 3
#define REDUCE_TILING_4 4
#define REDUCE_TILING_5 5

template<typename DTYPE>
class KernelReduce {
static constexpr uint32_t DEFAULT_BLK_STRIDE = 1;
static constexpr uint32_t DEFAULT_REP_STRIDE = 8;
static constexpr uint32_t REP_LEN = 256;
static constexpr uint32_t BLK_LEN = 32;
static constexpr uint32_t ONE_REPEAT_FLOAT_SIZE = REP_LEN / 4;
static constexpr uint32_t BINARY_BOUNDARY = DEFAULT_REP_STRIDE * 2;
public:
    __aicore__ inline KernelReduce() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, uint32_t totalLength, uint32_t outLength)
    {
        this->totalLength = totalLength;
        this->outLength = outLength;

        xGm.SetGlobalBuffer((__gm__ DTYPE *)x, totalLength);
        zGm.SetGlobalBuffer((__gm__ DTYPE *)z, outLength);
        pipe.InitBuffer(calcBuf, 1, totalLength * sizeof(DTYPE));
        pipe.InitBuffer(inQueueX, 1, totalLength * sizeof(DTYPE));
        pipe.InitBuffer(outQueueZ, 1, outLength * sizeof(DTYPE));
    }

    template<size_t ComputeKey = 0>
    __aicore__ inline void Compute()
    {
        Compute1();
        // if constexpr (ComputeKey == REDUCE_TILING_1) {
        //     Compute1();
        // } else if constexpr (ComputeKey == REDUCE_TILING_2) {
        //     Compute2();
        // } else if constexpr (ComputeKey == REDUCE_TILING_3) {
        //     Compute3();
        // } else if constexpr (ComputeKey == REDUCE_TILING_4) {
        //     Compute4();
        // } else if constexpr (ComputeKey == REDUCE_TILING_5) {
        //     Compute5();
        // }
    }

    template<size_t ComputeKey = 0>
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute<ComputeKey>();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<DTYPE> xLocal = inQueueX.AllocTensor<DTYPE>();
        AscendC::DataCopy(xLocal, xGm, totalLength);
        inQueueX.EnQue(xLocal);
        //-------------debug------------
        // for(int i = 0;i<this->totalLength;i++){
        //     AscendC::printf("id: %d input: %f\n",AscendC::GetBlockIdx(),(float)xLocal.GetValue(i));
        // }
    }
    // Only WholeReduceSum is used under 256B.
    __aicore__ inline void Compute1()
    {
        AscendC::LocalTensor<DTYPE> xLocal = inQueueX.DeQue<DTYPE>();
        AscendC::LocalTensor<DTYPE> zLocal = outQueueZ.AllocTensor<DTYPE>();
        AscendC::LocalTensor<DTYPE> workLocal = calcBuf.AllocTensor<DTYPE>();
        //-------------debug------------        
        // for(int i = 0;i<this->totalLength;i++){
        //         AscendC::printf("Compute1: %d input: %f\n",AscendC::GetBlockIdx(),(float)xLocal.GetValue(i));
        //     }
        AscendC::printf("idx in Compute1: %d input: %f\n",AscendC::GetBlockIdx(),(half)xLocal.GetValue(0));

        AscendC::ReduceMax<DTYPE>(zLocal, xLocal, workLocal, 128);

        //--------------debug------------
        AscendC::printf("ans in Compute1 : %d  ans[0] : %f\n",AscendC::GetBlockIdx(),(half)zLocal.GetValue(0));


        outQueueZ.EnQue<DTYPE>(zLocal);
        inQueueX.FreeTensor(xLocal);
        calcBuf.FreeTensor(workLocal);
    }

    // __aicore__ inline DTYPE reducemax_custom(AscendC::LocalTensor<DTYPE> x,size_t left , size_t right){
    //     if(left == right){
    //         return (DTYPE)x.GetValue(left);
    //     }
    //     int32_t mid = left + (right - left)/2;
    //     DTYPE Leftmax = reducemax_custom(x,left,mid);
    //     DTYPE Rightmax = reducemax_custom(x,mid+1,right);
    //     return (Leftmax > Rightmax)?Leftmax:Rightmax;
    // }
    // __aicore__ inline void Compute2()
    // {
    //     AscendC::LocalTensor<DTYPE> xLocal = inQueueX.DeQue<DTYPE>();
    //     AscendC::LocalTensor<DTYPE> zLocal = outQueueZ.AllocTensor<DTYPE>();
    //     AscendC::LocalTensor<DTYPE> workLocal = calcBuf.AllocTensor<DTYPE>();
    //     //-------------debug------------        
    //     // for(int i = 0;i<this->totalLength;i++){
    //     //         AscendC::printf("Compute1: %d input: %f\n",AscendC::GetBlockIdx(),(float)xLocal.GetValue(i));
    //     //     }
    //     AscendC::printf("idx in Compute1: %d input: %f\n",AscendC::GetBlockIdx(),(half)xLocal.GetValue(0));

    //     DTYPE temp = reducemax_custom(xLocal,0,this->totalLength);
    //     zLocal.SetValue(temp);

    //     //--------------debug------------
    //     AscendC::printf("ans in Compute1 : %d  ans[0] : %f\n",AscendC::GetBlockIdx(),(half)zLocal.GetValue(0));


    //     outQueueZ.EnQue<DTYPE>(zLocal);
    //     inQueueX.FreeTensor(xLocal);
    //     calcBuf.FreeTensor(workLocal);
    // }

    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<DTYPE> zLocal = outQueueZ.DeQue<DTYPE>();
        AscendC::DataCopy(zGm, zLocal, this->outLength);

        //--------------debug------------
        AscendC::printf("ans in CopyOut : %d  ans[0] : %d\n",AscendC::GetBlockIdx(),(half)zLocal.GetValue(0));

        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueZ;
    AscendC::TQue<AscendC::TPosition::VECCALC,1> calcBuf;
    AscendC::GlobalTensor<DTYPE> xGm;
    AscendC::GlobalTensor<DTYPE> zGm;
    uint32_t totalLength;
    uint32_t outLength;
};

extern "C" __global__ __aicore__ void reduce_custom(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, int32_t totalLength,int32_t outLength)
{
    KernelReduce<half> op;
    op.Init(x, z, totalLength, outLength);
    op.Process<1>();//数据量小于256B=128 float16
    // if (TILING_KEY_IS(REDUCE_TILING_1)) {
    //     op.Process<REDUCE_TILING_1>();
    // } else if (TILING_KEY_IS(REDUCE_TILING_2)) {
    //     op.Process<REDUCE_TILING_2>();
    // } else if (TILING_KEY_IS(REDUCE_TILING_3)) {
    //     op.Process<REDUCE_TILING_3>();
    // } else if (TILING_KEY_IS(REDUCE_TILING_4)) {
    //     op.Process<REDUCE_TILING_4>();
    // } else if (TILING_KEY_IS(REDUCE_TILING_5)) {
    //     op.Process<REDUCE_TILING_5>();
    // }
}

// call of kernel function
void reduce_custom_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *x, uint8_t *z,
                   uint8_t *workspace, int32_t totalLength,int32_t outLength)
{
    reduce_custom<<<blockDim, l2ctrl, stream>>>(x, z, workspace, totalLength,outLength);
}