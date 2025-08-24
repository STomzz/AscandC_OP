/**
 * @file reduce_max_fp16_16x16_main.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "data_utils.h"
#include "acl/acl.h"
/* NPU 上板调用接口 */
extern void reduce_custom_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *x, uint8_t *z,
                   uint8_t *workspace, int32_t totalLength,int32_t outLength);
int32_t main(int32_t argc, char *argv[])
{
    /* 16*16 = 256 个 float16，总字节 = 256 * 2 = 512 */
    constexpr uint32_t kElementNum = 32;
    constexpr uint32_t kBlockDim   = 1;          // 单核即可
    constexpr size_t   kInputBytes = kElementNum * sizeof(uint16_t);
    constexpr size_t   kOutputBytes = 1 * sizeof(uint16_t);

    /* ---------- NPU 上板 ---------- */
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *xHost, *zHost;
    uint8_t *xDevice, *zDevice;

    /* Host 端内存 */
    CHECK_ACL(aclrtMallocHost((void **)&xHost, kInputBytes));
    CHECK_ACL(aclrtMallocHost((void **)&zHost, kOutputBytes));

    /* Device 端内存 */
    CHECK_ACL(aclrtMalloc((void **)&xDevice, kInputBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&zDevice, kOutputBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    /* 读数据 → Host → Device */
    ReadFile("/root/code/AscandC_op_test/data/input_x.bin", kInputBytes, xHost, kInputBytes);
    CHECK_ACL(aclrtMemcpy(xDevice, kInputBytes, xHost, kInputBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    /* 启动核函数 */
    reduce_custom_do(kBlockDim,nullptr ,stream, xDevice, zDevice,nullptr,kElementNum,kOutputBytes);

    /* 等待完成 & 拷回 Host */
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(zHost, kOutputBytes, zDevice, kOutputBytes, ACL_MEMCPY_DEVICE_TO_HOST));

    // printf("ans host : %f",*((float16*)zHost));
    /* 写结果文件 */
    WriteFile("/root/code/AscandC_op_test/data/output_npu.bin", zHost, kOutputBytes);

    /* 释放资源 */
    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(zDevice));
    CHECK_ACL(aclrtFreeHost(xHost));
    CHECK_ACL(aclrtFreeHost(zHost));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    return 0;
}