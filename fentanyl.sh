#!/bin/bash

# 脚本名称: fentanyl.sh
# 用法: ./fentanyl.sh <command>

set -e  # 遇到错误时退出

LOCK_FILE="gpu.lock"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK_PATH="${SCRIPT_DIR}/${LOCK_FILE}"

# 显示用法
show_usage() {
    echo "用法: $0 <command>"
    echo "示例: $0 python train.py"
    echo "示例: $0 'python train.py --epochs 100'"
    exit 1
}

# 清理函数
cleanup() {
    if [ -f "${LOCK_PATH}" ]; then
        rm -f "${LOCK_PATH}"
        echo "已删除lock文件: ${LOCK_PATH}"
    fi
}

# 设置陷阱，确保在脚本退出时清理
trap cleanup EXIT INT TERM

# 检查参数
if [ $# -eq 0 ]; then
    echo "错误: 请提供要执行的命令"
    show_usage
fi

# 创建lock文件
echo "创建lock文件: ${LOCK_PATH}"
touch "${LOCK_PATH}"

echo "执行命令: $*"
echo "在命令执行期间，GPU占用脚本将被暂停..."
echo "----------------------------------------"

# 执行用户提供的命令
if eval "$*"; then
    echo "----------------------------------------"
    echo "命令执行成功"
    exit_code=0
else
    exit_code=$?
    echo "----------------------------------------"
    echo "命令执行失败，退出码: ${exit_code}"
fi

# cleanup函数会通过trap自动调用
exit ${exit_code}
