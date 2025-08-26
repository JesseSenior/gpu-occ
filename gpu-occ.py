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
        self.compute_threads: List[threading.Thread] = []
        self.running = True
        self.paused = False
        
        # 检查CUDA是否可用
        if not torch.cuda.is_available():
            print("CUDA不可用，退出程序")
            sys.exit(1)
            
        self.num_gpus = torch.cuda.device_count()
        print(f"检测到 {self.num_gpus} 个GPU")
        
        # 初始化GPU tensors列表
        self.gpu_tensors = [None] * self.num_gpus
        
        # 注册信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        print(f"\n接收到信号 {signum}，正在清理资源...")
        self.stop()
        sys.exit(0)
    
    def _allocate_gpu_memory(self, gpu_id: int):
        """在指定GPU上分配内存"""
        try:
            with torch.cuda.device(gpu_id):
                # 获取GPU总内存
                total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
                target_memory = int(total_memory * self.memory_ratio)
                
                # 计算需要分配的元素数量（float32，每个元素4字节）
                elements_needed = target_memory // 4
                
                # 分配内存
                tensor = torch.randn(elements_needed, dtype=torch.float32, device=f'cuda:{gpu_id}')
                self.gpu_tensors[gpu_id] = tensor
                
                used_memory = torch.cuda.memory_allocated(gpu_id)
                total_memory_gb = total_memory / (1024**3)
                used_memory_gb = used_memory / (1024**3)
                
                print(f"GPU {gpu_id}: 已占用 {used_memory_gb:.2f}GB / {total_memory_gb:.2f}GB "
                      f"({used_memory/total_memory*100:.1f}%)")
                      
        except Exception as e:
            print(f"GPU {gpu_id} 内存分配失败: {e}")
    
    def _compute_workload(self, gpu_id: int):
        """在指定GPU上执行计算任务"""
        try:
            with torch.cuda.device(gpu_id):
                # 创建用于计算的张量
                size = 1024
                a = torch.randn(size, size, device=f'cuda:{gpu_id}')
                b = torch.randn(size, size, device=f'cuda:{gpu_id}')
                
                while self.running and not self.paused:
                    # 执行矩阵乘法来占用GPU计算资源
                    c = torch.mm(a, b)
                    # 添加一些其他操作
                    c = torch.relu(c)
                    c = torch.sigmoid(c)
                    # 更新输入张量以避免优化
                    a = c + 0.01 * torch.randn_like(c)
                    
        except Exception as e:
            print(f"GPU {gpu_id} 计算任务出错: {e}")
    
    def _release_gpu_memory(self):
        """释放所有GPU内存"""
        for gpu_id in range(self.num_gpus):
            if self.gpu_tensors[gpu_id] is not None:
                del self.gpu_tensors[gpu_id]
                self.gpu_tensors[gpu_id] = None
                with torch.cuda.device(gpu_id):
                    torch.cuda.empty_cache()
        print("已释放所有GPU内存")
    
    def _stop_compute_threads(self):
        """停止所有计算线程"""
        for thread in self.compute_threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
        self.compute_threads.clear()
    
    def _start_gpu_occupation(self):
        """开始占用GPU"""
        if self.paused:
            print("开始占用GPU资源...")
            
            # 分配GPU内存
            for gpu_id in range(self.num_gpus):
                self._allocate_gpu_memory(gpu_id)
            
            # 启动计算线程
            for gpu_id in range(self.num_gpus):
                thread = threading.Thread(
                    target=self._compute_workload, 
                    args=(gpu_id,),
                    daemon=True
                )
                thread.start()
                self.compute_threads.append(thread)
            
            self.paused = False
    
    def _stop_gpu_occupation(self):
        """停止占用GPU"""
        if not self.paused:
            print("暂停GPU占用...")
            self.paused = True
            self._stop_compute_threads()
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
        self._stop_compute_threads()
        self._release_gpu_memory()
        print("程序已停止")

if __name__ == "__main__":
    occupier = GPUOccupier()
    occupier.run()
