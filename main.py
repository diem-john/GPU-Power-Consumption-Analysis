import argparse
import threading
import subprocess
import time
from logger import GPULogger


class ExperimentManager:
    def __init__(self, args):
        self.args = args
        self.stop_event = threading.Event()
        self.logger = GPULogger(target_gpus=args.gpus)

    def run(self):
        print(f"\n>>> Running Experiment Case: {self.args.case_id} <<<")

        log_thread = threading.Thread(target=self.logger.start_logging, args=(self.stop_event, self.args.case_id))
        log_thread.start()

        time.sleep(self.args.pre_log)

        cmd = [
            "accelerate", "launch", "--num_processes", str(self.args.gpus),
            "trainer.py",
            "--batch_size", str(self.args.batch_size),
            "--grad_accum", str(self.args.grad_accum),
            "--workers", str(self.args.workers),
            "--epochs", str(self.args.epochs),
            "--seq_len", str(self.args.seq_len),
            "--token", self.args.hf_token
        ]

        subprocess.run(cmd)

        print(f"Cooling down for {self.args.post_log}s...")
        time.sleep(self.args.post_log)

        self.stop_event.set()
        log_thread.join()
        print(f">>> Case {self.args.case_id} Finished. <<<\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-GPU Power Analysis CLI")
    parser.add_argument("--case_id", required=True)
    parser.add_argument("--hf_token", required=True, help="Hugging Face Access Token")
    parser.add_argument("--gpus", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--pre_log", type=int, default=30)
    parser.add_argument("--post_log", type=int, default=30)

    args = parser.parse_args()
    ExperimentManager(args).run()