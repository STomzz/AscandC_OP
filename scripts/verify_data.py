import numpy as np
import argparse

def read_bin_files(dtype):
    # 根据数据类型选择对应的numpy类型
    if dtype == 'fp16':
        data_type = np.float16
    else:
        data_type = np.float32

    bin_file = "/root/code/AscandC_op_test/data/input_x.bin"
    input_data = np.fromfile(bin_file, dtype=data_type)
    output_py = np.fromfile("/root/code/AscandC_op_test/data/output_py.bin", dtype=data_type)
    output_npu = np.fromfile("/root/code/AscandC_op_test/data/output_npu.bin", dtype=data_type)

    print("\n=== inputx ===")
    print(input_data)
    print(f"数据类型: {input_data.dtype}")
    
    print("\n=== output_py ===")
    print(output_py)
    print(f"数据类型: {output_py.dtype}")
    
    print("\n=== output_npu ===")
    print(output_npu)
    print(f"数据类型: {output_npu.dtype}")

    # 可以添加验证逻辑
    print("\n=== 验证结果 ===")
    if np.allclose(output_py, output_npu, atol=1e-3) and len(output_npu):
        print("结果一致")
    else:
        print("结果不一致")
        print("误差:", np.abs(output_py - output_npu))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='读取二进制文件并验证结果')
    parser.add_argument('-d','--dtype', type=str, default='fp32', 
                      choices=['fp16', 'fp32'],
                      help='数据类型，可选fp16或fp32，默认fp32')
    
    args = parser.parse_args()
    read_bin_files(args.dtype)