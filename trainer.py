import argparse
import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments, Trainer,
                          DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator
from huggingface_hub import login

class LLMTrainer:
    def __init__(self, model_id, batch_size, grad_accum, workers, epochs, seq_len, token):
        self.model_id = model_id
        self.batch_size = batch_size
        self.grad_accum = grad_accum
        self.workers = workers
        self.epochs = epochs
        self.seq_len = seq_len
        self.token = token
        self.accelerator = Accelerator()

    def train(self):
        # Authenticate
        if self.token:
            login(token=self.token)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, quantization_config=bnb_config,
            device_map={"": self.accelerator.process_index},
            token=self.token
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=self.token)
        tokenizer.pad_token = tokenizer.eos_token

        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM")
        model = get_peft_model(model, lora_config)

        dataset = load_dataset("tatsu-lab/alpaca", split="train[:1000]")
        tokenized_ds = dataset.map(lambda x: tokenizer(x["instruction"], truncation=True, max_length=self.seq_len), batched=True)

        args = TrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.grad_accum,
            num_train_epochs=self.epochs,
            dataloader_num_workers=self.workers,
            fp16=True,
            report_to="none"
        )

        trainer = Trainer(
            model=model, args=args, train_dataset=tokenized_ds,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
        )
        trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--grad_accum", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seq_len", type=int)
    parser.add_argument("--token", type=str)
    args = parser.parse_args()

    trainer = LLMTrainer("meta-llama/Llama-3.1-8B-Instruct", args.batch_size, args.grad_accum, args.workers, args.epochs, args.seq_len, args.token)
    trainer.train()