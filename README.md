# GPU Occupier

简单的GPU资源管理工具，在需要时暂停GPU占用。

## 快速开始

1. **启动GPU占用**
```bash
./run.sh <conda_env_name>
```

2. **临时释放GPU执行任务**
```bash
./fentanyl.sh "your_command_here"
```

## 文件说明

- `gpu-occ.py` - GPU占用脚本，占用80%显存和满利用率
- `fentanyl.sh` - 任务执行脚本，执行期间暂停GPU占用
- `run.sh` - 启动脚本，激活conda环境并运行GPU占用程序

## 工作原理

- GPU占用脚本每秒检查当前目录的`gpu.lock`文件
- 存在lock文件时自动释放GPU资源
- fentanyl.sh执行任务时创建lock文件，完成后自动删除

## 使用示例

```bash
# 启动GPU占用（后台运行）
./run.sh pytorch &

# 执行训练任务
./fentanyl.sh "python train.py --epochs 100"

# 执行其他命令
./fentanyl.sh "nvidia-smi"
```

## 依赖

- PyTorch with CUDA
- Conda环境

按Ctrl+C停止GPU占用程序。
