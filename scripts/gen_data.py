import numpy as np
import argparse


def gen_golden_data_simple(dtype):
    # 根据参数选择数据类型
    if dtype == 'fp16':
        data_type = np.float16
    else:  # 默认fp32
        data_type = np.float32

    input_x = np.random.uniform(1, 20, [32]).astype(data_type)
    golden = np.max(input_x).astype(data_type)

    input_x.tofile("/root/code/AscandC_op_test/data/input_x.bin")
    golden.tofile("/root/code/AscandC_op_test/data/output_py.bin")
    print(f"已生成{data_type}类型的数据")


if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='生成测试数据，可指定数据类型')
    # 添加数据类型参数，可选值为fp16，默认fp32
    parser.add_argument('-d','--dtype', type=str, default='fp32', 
                      choices=['fp16', 'fp32'],
                      help='数据类型，可选fp16或fp32，默认fp32')
    
    # 解析参数
    args = parser.parse_args()
    
    # 调用函数生成数据
    gen_golden_data_simple(args.dtype)
