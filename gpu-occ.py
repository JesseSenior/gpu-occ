import torch
import time
import os
import threading
import signal
import sys
from typing import List, Optional


class GPUOccupier:
    def __init__(self, lock_file: str = "gpu.lock", memory_ratio: float = 0.8):
        self.lock_file = lock_file
        self.memory_ratio = memory_ratio
        self.gpu_tensors: List[Optional[torch.Tensor]] = []
        self.compute_thread: Optional[threading.Thread] = None
        self.running = True
        self.paused = True
        self.occupied_gpus: List[int] = []
        self.available_gpus: List[int] = []  # 实际可用的GPU列表

        # 检查CUDA是否可用
        if not torch.cuda.is_available():
            print("CUDA不可用，退出程序")
            sys.exit(1)

        # 获取并验证可用的GPU
        self._discover_available_gpus()

        # 注册信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _discover_available_gpus(self):
        """发现并验证实际可用的GPU"""
        self.num_gpus = torch.cuda.device_count()
        print(f"torch.cuda.device_count() 返回: {self.num_gpus}")

        # 验证每个GPU是否真正可用
        self.available_gpus = []
        for gpu_id in range(self.num_gpus):
            try:
                # 尝试创建一个小的测试张量来验证GPU可用性
                with torch.cuda.device(gpu_id):
                    test_tensor = torch.tensor([1.0], device=f"cuda:{gpu_id}")
                    del test_tensor
                    torch.cuda.empty_cache()
                    self.available_gpus.append(gpu_id)
                    print(f"GPU {gpu_id}: 可用")
            except Exception as e:
                print(f"GPU {gpu_id}: 不可用 - {e}")

        if not self.available_gpus:
            print("没有找到可用的GPU，退出程序")
            sys.exit(1)

        print(f"发现 {len(self.available_gpus)} 个可用GPU: {self.available_gpus}")

        # 根据实际可用GPU数量初始化tensor列表
        # 使用字典而不是列表，避免索引问题
        self.gpu_tensors = {}

    def _signal_handler(self, signum, frame):
        print(f"\n接收到信号 {signum}，正在清理资源...")
        self.stop()
        sys.exit(0)

    def _allocate_gpu_memory(self, gpu_id: int) -> bool:
        """在指定GPU上分配内存，返回是否成功"""
        if gpu_id not in self.available_gpus:
            print(f"GPU {gpu_id} 不在可用GPU列表中")
            return False

        try:
            with torch.cuda.device(gpu_id):
                # 获取GPU总内存
                total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
                target_memory = int(total_memory * self.memory_ratio)

                # 计算需要分配的元素数量（float32，每个元素4字节）
                elements_needed = target_memory // 4

                # 分配内存
                tensor = torch.randn(elements_needed, dtype=torch.float32, device=f"cuda:{gpu_id}")
                self.gpu_tensors[gpu_id] = tensor

                used_memory = torch.cuda.memory_allocated(gpu_id)
                total_memory_gb = total_memory / (1024**3)
                used_memory_gb = used_memory / (1024**3)

                print(
                    f"GPU {gpu_id}: 已占用 {used_memory_gb:.2f}GB / {total_memory_gb:.2f}GB "
                    f"({used_memory / total_memory * 100:.1f}%)"
                )
                return True

        except Exception as e:
            print(f"GPU {gpu_id} 内存分配失败: {e}")
            return False

    def _compute_workload_all_gpus(self):
        """在所有占用内存的GPU上执行计算任务"""
        try:
            compute_tensors = {}
            for gpu_id in self.occupied_gpus:
                try:
                    size = 4 * 1024  # 可以根据需要调整
                    # 使用二维张量进行矩阵乘法
                    a = torch.randn(size, size, device=f"cuda:{gpu_id}")
                    b = torch.randn(size, size, device=f"cuda:{gpu_id}")
                    compute_tensors[gpu_id] = [a, b]  # 使用列表便于原地修改
                except Exception as e:
                    print(f"GPU {gpu_id} 计算张量创建失败: {e}")

            while self.running and not self.paused:
                for gpu_id in list(compute_tensors.keys()):
                    try:
                        a, b = compute_tensors[gpu_id]

                        with torch.cuda.device(gpu_id):
                            # 矩阵乘法
                            c = torch.mm(a, b)
                            c = torch.relu(c)
                            c = torch.sigmoid(c)

                            # 原地更新以减少内存分配
                            noise = torch.randn_like(c) * 0.01
                            a.copy_((c + noise).clamp(max=10))

                    except Exception as e:
                        print(f"GPU {gpu_id} 计算任务出错: {e}")
                        compute_tensors.pop(gpu_id, None)

                time.sleep(0.001)

        except Exception as e:
            print(f"计算线程出错: {e}")

    def _release_gpu_memory(self):
        """释放所有GPU内存"""
        for gpu_id in list(self.gpu_tensors.keys()):
            try:
                if self.gpu_tensors[gpu_id] is not None:
                    del self.gpu_tensors[gpu_id]
                    with torch.cuda.device(gpu_id):
                        torch.cuda.empty_cache()
            except Exception as e:
                print(f"释放GPU {gpu_id} 内存时出错: {e}")

        self.gpu_tensors.clear()
        self.occupied_gpus.clear()
        print("已释放所有GPU内存")

    def _stop_compute_thread(self):
        """停止计算线程"""
        if self.compute_thread and self.compute_thread.is_alive():
            self.compute_thread.join(timeout=2.0)
        self.compute_thread = None

    def _start_gpu_occupation(self):
        """开始占用GPU"""
        if self.paused:
            print("开始占用GPU资源...")

            # 分配GPU内存并记录成功的GPU
            self.occupied_gpus.clear()
            for gpu_id in self.available_gpus:  # 只尝试可用的GPU
                if self._allocate_gpu_memory(gpu_id):
                    self.occupied_gpus.append(gpu_id)

            if self.occupied_gpus:
                print(f"成功占用 {len(self.occupied_gpus)} 个GPU: {self.occupied_gpus}")

                # 启动单个计算线程处理所有GPU
                self.compute_thread = threading.Thread(target=self._compute_workload_all_gpus, daemon=True)
                self.compute_thread.start()

                self.paused = False
            else:
                print("没有成功占用任何GPU")

    def _stop_gpu_occupation(self):
        """停止占用GPU"""
        if not self.paused:
            print("暂停GPU占用...")
            self.paused = True
            self._stop_compute_thread()
            self._release_gpu_memory()

    def run(self):
        """主运行循环"""
        print(f"开始监控lock文件: {self.lock_file}")
        print("按 Ctrl+C 退出程序")

        try:
            while self.running:
                # 检查lock文件是否存在
                if os.path.exists(self.lock_file):
                    if not self.paused:
                        self._stop_gpu_occupation()
                else:
                    if self.paused:
                        self._start_gpu_occupation()

                # 等待1秒
                time.sleep(1)

        except KeyboardInterrupt:
            print("\n接收到中断信号...")
        finally:
            self.stop()

    def stop(self):
        """停止程序"""
        self.running = False
        self.paused = True
        self._stop_compute_thread()
        self._release_gpu_memory()
        print("程序已停止")


if __name__ == "__main__":
    occupier = GPUOccupier()
    occupier.run()
