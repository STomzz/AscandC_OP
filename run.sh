#!/bin/bash
##############################################################################
# 脚本功能：支持指定 fp16/fp32 类型，搬运可执行文件和依赖库到目标目录并执行，
#          自动匹配对应类型的验证脚本
# 用法：
#   1. 默认使用 fp32 执行：./run.sh
#   2. 指定 fp16 执行：./run.sh fp16
#   3. 指定 fp32 执行：./run.sh fp32
# 依赖路径：
#   - 可执行文件：./out/bin/ascendc_kernels_bbit
#   - 依赖库：./out/lib/libascendc_kernels_npu.so
#   - 验证脚本：./scripts/verify_data.py
# 目标目录：./out/exe
##############################################################################

# ========================== 1. 工具函数：错误处理 ==========================
error_handler() {
    echo -e "\033[31m[ERROR] $1\033[0m"  # 红色错误提示
    if [ $2 -eq 1 ]; then
        echo -e "\033[33m脚本执行失败，已退出。\033[0m"
        exit 1
    fi
}

# ========================== 2. 解析命令行参数（新增：处理 fp16/fp32） ==========================
echo -e "\033[34m[INIT] 解析执行类型参数...\033[0m"

# 定义支持的类型列表
SUPPORTED_TYPES=("fp16" "fp32")
# 默认类型为 fp32
RUN_TYPE="fp32"

# 处理位置参数（若用户传入参数，则验证并更新类型）
if [ $# -ge 1 ]; then
    USER_INPUT=$1
    # 检查用户输入是否在支持的类型中
    if [[ " ${SUPPORTED_TYPES[@]} " =~ " ${USER_INPUT} " ]]; then
        RUN_TYPE=$USER_INPUT
        echo "已指定执行类型：${RUN_TYPE}"
    else
        error_handler "不支持的类型：${USER_INPUT}\n仅支持 ${SUPPORTED_TYPES[*]}（例：./run.sh fp16）" 1
    fi
else
    echo "未指定执行类型，默认使用：${RUN_TYPE}"
fi

# ========================== 3. 定义基础参数（与原逻辑一致） ==========================
# 项目根目录（基于脚本所在路径，确保跨目录执行时路径正确）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 源文件路径
EXEC_FILE_SRC="${SCRIPT_DIR}/out/bin/ascendc_kernels_bbit"
EXEC_LIB_SRC="${SCRIPT_DIR}/out/lib/libascendc_kernels_npu.so"

# 目标目录与文件路径
EXEC_DIR="${SCRIPT_DIR}/out/exe"
EXEC_FILE_DEST="${EXEC_DIR}/ascendc_kernels_bbit"
EXEC_LIB_DEST="${EXEC_DIR}/libascendc_kernels_npu.so"

# 验证脚本路径（新增：关联类型参数）
GEN_SCRIPT="${SCRIPT_DIR}/scripts/gen_data.py"
VERIFY_SCRIPT="${SCRIPT_DIR}/scripts/verify_data.py"


# ========================== 4. 前置检查：源文件 + 验证脚本是否存在 ==========================
echo -e "\033[34m[STEP 1/6] 检查必要文件是否存在...\033[0m"

# 检查可执行文件
if [ ! -f "${EXEC_FILE_SRC}" ]; then
    error_handler "可执行文件不存在：${EXEC_FILE_SRC}\n请确认编译是否成功！" 1
fi

# 检查依赖库
if [ ! -f "${EXEC_LIB_SRC}" ]; then
    error_handler "依赖库不存在：${EXEC_LIB_SRC}\n请确认编译是否成功！" 1
fi

# 检查验证脚本（新增：确保验证脚本存在）
if [ ! -f "${VERIFY_SCRIPT}" ]; then
    error_handler "验证脚本不存在：${VERIFY_SCRIPT}\n请确认脚本路径是否正确！" 1
fi


# ========================== 5. 创建目标文件夹 ==========================
echo -e "\033[34m[STEP 2/6] 检查并创建目标文件夹...\033[0m"

if [ ! -d "${EXEC_DIR}" ]; then
    echo "创建目标文件夹：${EXEC_DIR}"
    mkdir -p "${EXEC_DIR}" || error_handler "创建目标文件夹失败（权限不足？）" 1
else
    echo "目标文件夹 ${EXEC_DIR} 已存在，跳过创建"
fi


# ========================== 6. 复制文件到目标文件夹 ==========================
echo -e "\033[34m[STEP 3/6] 复制文件到目标文件夹...\033[0m"

# 复制可执行文件
echo "复制可执行文件：${EXEC_FILE_SRC} → ${EXEC_FILE_DEST}"
cp -f "${EXEC_FILE_SRC}" "${EXEC_FILE_DEST}" || error_handler "复制可执行文件失败" 1

# 复制依赖库
echo "复制依赖库：${EXEC_LIB_SRC} → ${EXEC_LIB_DEST}"
cp -f "${EXEC_LIB_SRC}" "${EXEC_LIB_DEST}" || error_handler "复制依赖库失败" 1


# ========================== 7. 检查目标文件有效性 ==========================
echo -e "\033[34m[STEP 4/6] 检查目标文件有效性...\033[0m"

# 给可执行文件添加执行权限
if [ ! -x "${EXEC_FILE_DEST}" ]; then
    echo "给可执行文件添加执行权限：${EXEC_FILE_DEST}"
    chmod +x "${EXEC_FILE_DEST}" || error_handler "添加执行权限失败（权限不足？）" 1
fi

# 检查依赖库是否为有效共享库
if ! file "${EXEC_LIB_DEST}" | grep -q "shared object"; then
    error_handler "依赖库 ${EXEC_LIB_DEST} 不是有效共享库（文件损坏？）" 1
fi

# 检查可执行文件是否依赖目标库
if ! ldd "${EXEC_FILE_DEST}" | grep -q "libascendc_kernels_npu.so"; then
    echo -e "\033[33m[WARNING] 可执行文件未依赖 libascendc_kernels_npu.so（版本不匹配？）\033[0m"
    read -p "是否继续执行？(y/N) " choice
    if [ "${choice}" != "y" ] && [ "${choice}" != "Y" ]; then
        error_handler "用户终止执行" 1
    fi
fi


# ========================== 8. 执行可执行程序 ==========================
echo -e "\033[34m[STEP 5/6] 执行 ${RUN_TYPE} 类型程序...\033[0m"

# 进入目标目录（确保程序优先加载同目录的依赖库）
cd "${EXEC_DIR}" || error_handler "进入目标目录 ${EXEC_DIR} 失败" 1

# 执行程序（若后续需要给可执行程序传参数，可在此处添加，例：./ascendc_kernels_bbit --dtype ${RUN_TYPE}）
./ascendc_kernels_bbit
EXEC_RESULT=$?

# 判断程序执行结果
if [ ${EXEC_RESULT} -eq 0 ]; then
    echo -e "\033[32m[SUCCESS] 程序执行完成（${RUN_TYPE}），退出码：${EXEC_RESULT}\033[0m"
else
    error_handler "程序执行失败（${RUN_TYPE}），退出码：${EXEC_RESULT}" 0
    exit ${EXEC_RESULT}
fi


# ========================== 9. 执行验证脚本（新增：传递 -d 参数） ==========================
echo -e "\033[34m[STEP 6/6] 执行 ${RUN_TYPE} 类型数据验证...\033[0m"

# 执行验证脚本，自动传递 -d 参数（匹配当前 RUN_TYPE）
python3 "${GEN_SCRIPT}" -d "${RUN_TYPE}"
python3 "${VERIFY_SCRIPT}" -d "${RUN_TYPE}"  # 若系统默认python是3.x，可改为 python
VERIFY_RESULT=$?

# 判断验证脚本执行结果
if [ ${VERIFY_RESULT} -eq 0 ]; then
    echo -e "\033[32m[SUCCESS] 数据验证完成（${RUN_TYPE}），验证脚本退出码：${VERIFY_RESULT}\033[0m"
else
    error_handler "数据验证失败（${RUN_TYPE}），验证脚本退出码：${VERIFY_RESULT}" 0
    exit ${VERIFY_RESULT}
fi