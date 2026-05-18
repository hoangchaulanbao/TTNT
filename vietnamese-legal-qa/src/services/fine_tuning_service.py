"""Fine-tuning Service - Orchestrate LLM fine-tuning pipeline."""

from typing import Dict, List
from pathlib import Path

from src.components.data_collector import DataCollector
from src.components.fine_tuner import FineTuner
from src.models.data_models import QAPair, TrainingResult
from src.config.settings import Settings


class FineTuningService:
    """Orchestrate fine-tuning pipeline cho multiple models."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.collector = DataCollector(settings)
        self.fine_tuner = FineTuner(settings)
        self.adapters_dir = Path(settings.config.get("models", {}).get(
            "adapters_dir", "models/adapters"
        ))
        self.adapters_dir.mkdir(parents=True, exist_ok=True)

    def finetune_model(
        self,
        model_key: str,
        qa_pairs: List[QAPair],
        format_type: str = "chatml",
    ) -> TrainingResult:
        """
        Fine-tune một model cụ thể.
        
        Args:
            model_key: Key trong config (vistral, phogpt, qwen)
            qa_pairs: Tập Q&A cho training
            format_type: Format template
            
        Returns:
            TrainingResult
        """
        model_config = self.settings.get_llm_model(model_key)
        model_name = model_config.get("name")
        output_dir = str(self.adapters_dir / model_key)

        print(f"\n{'='*60}")
        print(f"[FineTuningService] Starting fine-tuning: {model_key}")
        print(f"  Model: {model_name}")
        print(f"  Dataset: {len(qa_pairs)} samples")
        print(f"  Output: {output_dir}")
        print(f"{'='*60}\n")

        # Step 1: Prepare dataset
        dataset = self.fine_tuner.prepare_dataset(qa_pairs, format_type)

        # Step 2: Configure QLoRA
        config = self.fine_tuner.configure_qlora(model_name)

        # Step 3: Train
        result = self.fine_tuner.train(
            model=config["model"],
            tokenizer=config["tokenizer"],
            dataset=dataset,
            output_dir=output_dir,
            model_key=model_key,
        )

        return result

    def finetune_all_models(
        self,
        qa_pairs: List[QAPair],
        format_type: str = "chatml",
    ) -> Dict[str, TrainingResult]:
        """
        Fine-tune tất cả 3 models.
        
        Args:
            qa_pairs: Tập Q&A
            format_type: Format template
            
        Returns:
            Dict mapping model_key -> TrainingResult
        """
        results = {}
        model_keys = ["vistral", "phogpt", "qwen"]

        for model_key in model_keys:
            try:
                result = self.finetune_model(model_key, qa_pairs, format_type)
                results[model_key] = result
            except Exception as e:
                print(f"[FineTuningService] Error fine-tuning {model_key}: {e}")
                results[model_key] = TrainingResult(
                    model_name=model_key,
                    final_loss=-1,
                    training_time_minutes=0,
                    epochs_completed=0,
                    adapter_path="",
                )

        # Print summary
        print(f"\n{'='*60}")
        print("[FineTuningService] Fine-tuning Summary:")
        print(f"{'='*60}")
        for key, result in results.items():
            status = "✓" if result.final_loss >= 0 else "✗"
            print(f"  {status} {key}: loss={result.final_loss:.4f}, "
                  f"time={result.training_time_minutes:.1f}min")
        print(f"{'='*60}\n")

        return results

    def load_dataset(self, dataset_path: str) -> List[QAPair]:
        """Load Q&A dataset from file."""
        return self.collector.load_alqac_dataset(dataset_path)
