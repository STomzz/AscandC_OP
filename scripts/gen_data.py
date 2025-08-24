#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import numpy as np


def gen_golden_data_simple():
    input_x = np.random.uniform(1, 20, [64]).astype(np.float16)
    # input_x = np.append(input_x, 45.0).astype(np.float16)
    golden = np.max(input_x).astype(np.float16)

    input_x.tofile("/root/code/AscandC_op_test/data/input_x.bin")
    golden.tofile("/root/code/AscandC_op_test/data/output_py.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
