import csv
import time
import pynvml
from datetime import datetime
from pynvml import NVML_TEMP_GPU, NVMLError_NotSupported, NVMLError

class GPULogger:
    def __init__(self, target_gpus, log_file="gpu_power_log.csv"):
        self.target_gpus = target_gpus
        self.log_file = log_file
        self.handles = []
        self.gpu_indices = []
        self._initialize_nvml()

    def _initialize_nvml(self):
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            self.gpu_indices = list(range(min(self.target_gpus, device_count)))
            self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in self.gpu_indices]
            print(f"[{self._now()}] NVML Initialized. Monitoring GPUs: {self.gpu_indices}")
        except NVMLError as e:
            print(f"[{self._now()}] Failed to initialize NVML: {e}")

    def _now(self):
        return datetime.now().strftime('%H:%M:%S')

    def log_iteration(self, writer, case_id):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        for i, handle in zip(self.gpu_indices, self.handles):
            try:
                pwr = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                temp = pynvml.nvmlDeviceGetTemperature(handle, NVML_TEMP_GPU)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle).used // (1024**2)
                try:
                    fan = pynvml.nvmlDeviceGetFanSpeed(handle)
                except NVMLError_NotSupported:
                    fan = "N/A"
                writer.writerow([timestamp, case_id, i, fan, temp, pwr, mem, util])
            except NVMLError:
                writer.writerow([timestamp, case_id, i, "ERR", "ERR", "ERR", "ERR", "ERR"])

    def start_logging(self, stop_event, case_id, interval=1.0):
        file_exists = False
        try:
            with open(self.log_file, 'r') as f: file_exists = True
        except FileNotFoundError: pass

        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "CaseID", "GPU_ID", "Fan%", "TempC", "PowerW", "MemMiB", "Util%"])
            while not stop_event.is_set():
                self.log_iteration(writer, case_id)
                f.flush()
                time.sleep(interval)
        pynvml.nvmlShutdown()