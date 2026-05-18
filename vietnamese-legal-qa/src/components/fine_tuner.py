"""Fine Tuner - QLoRA fine-tuning cho LLM."""

from typing import List, Optional, Dict
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

from src.models.data_models import QAPair, InstructionSample, TrainingResult
from src.config.settings import Settings


class FineTuner:
    """Fine-tuning LLM với QLoRA trên tập Q&A pháp luật."""

    SYSTEM_PROMPT = (
        "Bạn là trợ lý pháp luật Việt Nam chuyên nghiệp. Nhiệm vụ của bạn là trả lời "
        "câu hỏi pháp luật một cách chi tiết, dễ hiểu và chính xác. "
        "Luôn trích dẫn cụ thể Điều/Khoản/Điểm và số hiệu văn bản. "
        "Giải thích bằng ngôn ngữ đơn giản, có cấu trúc rõ ràng. "
        "Nếu vấn đề phức tạp, khuyên người dùng tham khảo luật sư chuyên nghiệp."
    )

    def __init__(self, settings: Settings):
        self.settings = settings
        self.ft_config = settings.fine_tuning

    def prepare_dataset(
        self, qa_pairs: List[QAPair], format_type: str = "chatml"
    ) -> Dataset:
        """
        Chuẩn bị dataset cho fine-tuning.
        
        Args:
            qa_pairs: Danh sách cặp Q&A
            format_type: Format template ('chatml' hoặc 'alpaca')
            
        Returns:
            HuggingFace Dataset
        """
        samples = []

        for qa in qa_pairs:
            if format_type == "chatml":
                text = self._format_chatml(qa)
            else:
                text = self._format_alpaca(qa)
            samples.append({"text": text})

        dataset = Dataset.from_list(samples)

        # Shuffle with seed for reproducibility
        seed = self.ft_config.get("training", {}).get("seed", 42)
        dataset = dataset.shuffle(seed=seed)

        print(f"[FineTuner] Prepared dataset: {len(dataset)} samples ({format_type} format)")
        return dataset

    def configure_qlora(self, model_name: str) -> Dict:
        """
        Cấu hình QLoRA parameters.
        
        Args:
            model_name: HuggingFace model name
            
        Returns:
            Dict with model, tokenizer, lora_config, bnb_config
        """
        qlora_config = self.ft_config.get("qlora", {})

        # BitsAndBytes config for 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        # LoRA config
        lora_config = LoraConfig(
            r=qlora_config.get("lora_rank", 64),
            lora_alpha=qlora_config.get("lora_alpha", 16),
            lora_dropout=qlora_config.get("lora_dropout", 0.1),
            target_modules=qlora_config.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Load model with quantization
        print(f"[FineTuner] Loading model: {model_name} (4-bit quantized)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Prepare model for training
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[FineTuner] Trainable params: {trainable_params:,} / {total_params:,} "
              f"({100 * trainable_params / total_params:.2f}%)")

        return {
            "model": model,
            "tokenizer": tokenizer,
            "lora_config": lora_config,
            "bnb_config": bnb_config,
        }

    def train(
        self,
        model,
        tokenizer,
        dataset: Dataset,
        output_dir: str,
        model_key: str = "model",
    ) -> TrainingResult:
        """
        Thực hiện fine-tuning.
        
        Args:
            model: PEFT model
            tokenizer: Tokenizer
            dataset: Training dataset
            output_dir: Thư mục lưu adapter
            model_key: Key để identify model
            
        Returns:
            TrainingResult
        """
        training_config = self.ft_config.get("training", {})

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=training_config.get("epochs", 3),
            per_device_train_batch_size=training_config.get("batch_size", 4),
            gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
            learning_rate=training_config.get("learning_rate", 2e-4),
            warmup_ratio=training_config.get("warmup_ratio", 0.03),
            max_grad_norm=0.3,
            logging_steps=10,
            save_strategy="epoch",
            fp16=True,
            optim="paged_adamw_32bit",
            seed=training_config.get("seed", 42),
            report_to="none",
        )

        max_seq_length = training_config.get("max_seq_length", 2048)

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=training_args,
            max_seq_length=max_seq_length,
            dataset_text_field="text",
        )

        print(f"[FineTuner] Starting training: {model_key}...")
        import time
        start_time = time.time()

        train_result = trainer.train()

        training_time = (time.time() - start_time) / 60

        # Save adapter
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        result = TrainingResult(
            model_name=model_key,
            final_loss=train_result.training_loss,
            training_time_minutes=training_time,
            epochs_completed=training_config.get("epochs", 3),
            adapter_path=output_dir,
        )

        print(f"[FineTuner] Training complete: {model_key}")
        print(f"  Loss: {result.final_loss:.4f}")
        print(f"  Time: {result.training_time_minutes:.1f} minutes")
        print(f"  Adapter saved: {output_dir}")

        return result

    def save_adapter(self, model, tokenizer, path: str) -> None:
        """Save LoRA adapter weights."""
        Path(path).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        print(f"[FineTuner] Adapter saved to: {path}")

    def _format_chatml(self, qa: QAPair) -> str:
        """Format Q&A pair as ChatML template."""
        context_part = ""
        if qa.context:
            context_part = f"\n\nNgữ cảnh pháp luật:\n{qa.context}"

        return (
            f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{qa.question}{context_part}<|im_end|>\n"
            f"<|im_start|>assistant\n{qa.answer}<|im_end|>"
        )

    def _format_alpaca(self, qa: QAPair) -> str:
        """Format Q&A pair as Alpaca template."""
        context_part = ""
        if qa.context:
            context_part = f"\n\n### Input:\n{qa.context}"

        return (
            f"### System:\n{self.SYSTEM_PROMPT}\n\n"
            f"### Instruction:\n{qa.question}{context_part}\n\n"
            f"### Response:\n{qa.answer}"
        )
