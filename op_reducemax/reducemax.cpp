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

template<typename DTYPE>//模板类
class KernelReduce {
// static constexpr uint32_t DEFAULT_BLK_STRIDE = 1;
// static constexpr uint32_t DEFAULT_REP_STRIDE = 8;
// static constexpr uint32_t REP_LEN = 256;
// static constexpr uint32_t BLK_LEN = 32;
// static constexpr uint32_t ONE_REPEAT_FLOAT_SIZE = REP_LEN / 4;
// static constexpr uint32_t BINARY_BOUNDARY = DEFAULT_REP_STRIDE * 2;
public:
    __aicore__ inline KernelReduce() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, uint32_t totalLength, uint32_t outLength)
    {
        this->totalLength = totalLength;//数据总长度 测试的时候用的是64个float16= 64x2Byte=128Byte(每个ai core能处理256Byte)
        this->outLength = outLength; //输出长度 reducemax 获取最大值outlength = 1

        xGm.SetGlobalBuffer((__gm__ DTYPE *)x, totalLength); //xGm set 输入
        zGm.SetGlobalBuffer((__gm__ DTYPE *)z, outLength); // zGm set  输出
        pipe.InitBuffer(calcBuf, 1, totalLength * sizeof(DTYPE));//中间变量  1代表buffer=1 不使用双重缓冲,同时Blocknum = 1 意味着只调用一个ai core
        pipe.InitBuffer(inQueueX, 1, totalLength * sizeof(DTYPE));//输入队列
        pipe.InitBuffer(outQueueZ, 1, outLength * sizeof(DTYPE));//输出队列
    }

    template<size_t ComputeKey = 0>
    __aicore__ inline void Compute()
    {
        // Compute1();
        if constexpr (ComputeKey == REDUCE_TILING_1) {
            Compute1();//调用reducemax接口
        } else if constexpr (ComputeKey == REDUCE_TILING_2) {
            Compute2();//使用Ascand::Std::max手搓
        }
    }

    template<size_t ComputeKey = 0>
    __aicore__ inline void Process()
    {   //因为数据少并且只用了一个aicore 因此不用重复搬运
        CopyIn();
        Compute<ComputeKey>();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<DTYPE> xLocal = inQueueX.AllocTensor<DTYPE>(); //使用输入队列分配一个局部变量xlocal
        AscendC::DataCopy(xLocal, xGm, totalLength);    //搬运数据 Gm -> Local
        inQueueX.EnQue(xLocal);                 //数据入队 Local -> inputQue
        //-------------debug------------
        // for(int i = 0;i<this->totalLength;i++){
            AscendC::printf("id: %d in CopyIn: %f\n",AscendC::GetBlockIdx(),(DTYPE)xLocal.GetValue(0));
        // }
    }
    // Only WholeReduceSum is used under 256B.
    __aicore__ inline void Compute1()
    {
        AscendC::printf("===use compute 1, datalength : %d ====\n",totalLength);
        AscendC::LocalTensor<DTYPE> xLocal = inQueueX.DeQue<DTYPE>(); //数据出队 inputQue -> xlocal
        AscendC::LocalTensor<DTYPE> zLocal = outQueueZ.AllocTensor<DTYPE>();    //outQue分配局部变量 zlocal
        AscendC::LocalTensor<DTYPE> workLocal = calcBuf.AllocTensor<DTYPE>();   //临时空间分配
        //-------------debug------------ √        
        for(int i = 0;i<this->totalLength;i++){
                AscendC::printf("Compute1: %d input: %f\n",i,(half)xLocal.GetValue(i));
            }
        AscendC::ReduceMax<DTYPE>(zLocal, xLocal, workLocal, totalLength,false); 
        // AscendC::ReduceMax<DTYPE>(zLocal, xLocal, workLocal, totalLength); //调用ReduceMax接口(dst,src,workspace,length)

        //--------------debug------------ ×
        AscendC::printf("ans in Compute1 : %d  ans[0] : %f\n",AscendC::GetBlockIdx(),(half)zLocal.GetValue(0));


        outQueueZ.EnQue<DTYPE>(zLocal);//输出队列把zlocal入队
        inQueueX.FreeTensor(xLocal); //释放空间
        calcBuf.FreeTensor(workLocal);
    }

    __aicore__ inline void Compute2()
    {
        AscendC::printf("use compute 2, datalength : %d",totalLength);
        AscendC::LocalTensor<DTYPE> xLocal = inQueueX.DeQue<DTYPE>(); //同上
        AscendC::LocalTensor<DTYPE> zLocal = outQueueZ.AllocTensor<DTYPE>();
        AscendC::LocalTensor<DTYPE> workLocal = calcBuf.AllocTensor<DTYPE>();
        //-------------debug------------        
        // for(int i = 0;i<this->totalLength;i++){
        //         AscendC::printf("Compute1: %d input: %f\n",AscendC::GetBlockIdx(),(DTYPE)xLocal.GetValue(i));
        //     }
        // AscendC::Cast(workLocal,xLocal,0,totalLength);
        // AscendC::printf("idx in Compute2 %d input(DTYPE): %f\n",AscendC::GetBlockIdx(),(DTYPE)workLocal.GetValue(0));
        DTYPE temp = xLocal.GetValue(0); //Register中申请的LocalTensor
        for(int i = 0;i<totalLength;i++){
            temp = AscendC::Std::max(temp,xLocal.GetValue(i));//问题,使用的是half(float16),不支持> < max等比较符号 ,float类型执行错误
        }
        zLocal.SetValue(0,workLocal.GetValue(0));

        //--------------debug------------
        // AscendC::printf("ans in Compute2 : %d  ans[0] : %f\n",AscendC::GetBlockIdx(),(DTYPE)temp);


        outQueueZ.EnQue<DTYPE>(zLocal);
        inQueueX.FreeTensor(xLocal);
        calcBuf.FreeTensor(workLocal);
    }

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
    KernelReduce<half> op; //<>选择类型
    op.Init(x, z, totalLength, outLength);
    op.Process<1>();//<>选择compute x
}

// call of kernel function
void reduce_custom_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *x, uint8_t *z,
                   uint8_t *workspace, int32_t totalLength,int32_t outLength)
{
    reduce_custom<<<blockDim, l2ctrl, stream>>>(x, z, workspace, totalLength,outLength);
}