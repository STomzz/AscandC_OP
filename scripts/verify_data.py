import numpy as np

if __name__ == "__main__":
    bin_file = "/root/code/AscandC_op_test/data/input_x.bin"  # 替换为你的.bin文件路径
    
    input = np.fromfile(bin_file,dtype=np.float16)
    output_py = np.fromfile("/root/code/AscandC_op_test/data/output_py.bin",dtype=np.float16)
    output_npu = np.fromfile("/root/code/AscandC_op_test/data/output_npu.bin",dtype=np.float16)

    # 方法2：以指定数据类型读取（根据实际情况选择）
    print("\n=== inputx ===")
    print(input)  # float类型（4字节）
    print("\n=== output_py ===")
    print(output_py)  # float类型（4字节）
    print("\n=== output_npu ===")
    print(output_npu)  # float类型（4字节）