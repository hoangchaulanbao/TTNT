"""Configuration settings loader."""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Settings:
    """Application settings loaded from config.yaml."""

    config: dict = field(default_factory=dict)

    @classmethod
    def load(cls, config_path: str = "config.yaml") -> "Settings":
        """Load settings from YAML config file."""
        path = Path(config_path)
        if not path.exists():
            # Try parent directories
            for parent in Path.cwd().parents:
                candidate = parent / config_path
                if candidate.exists():
                    path = candidate
                    break

        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        return cls(config=config)

    @property
    def data(self) -> dict:
        return self.config.get("data", {})

    @property
    def llm(self) -> dict:
        return self.config.get("llm", {})

    @property
    def embedding(self) -> dict:
        return self.config.get("embedding", {})

    @property
    def reranker(self) -> dict:
        return self.config.get("reranker", {})

    @property
    def rag(self) -> dict:
        return self.config.get("rag", {})

    @property
    def conversation(self) -> dict:
        return self.config.get("conversation", {})

    @property
    def quality_guard(self) -> dict:
        return self.config.get("quality_guard", {})

    @property
    def fine_tuning(self) -> dict:
        return self.config.get("fine_tuning", {})

    @property
    def crawling(self) -> dict:
        return self.config.get("crawling", {})

    def get_llm_model(self, model_key: str) -> dict:
        """Get LLM model config by key (vistral, phogpt, qwen)."""
        return self.llm.get("models", {}).get(model_key, {})

    def get_embedding_model(self, model_key: str) -> dict:
        """Get embedding model config by key (phobert, multilingual_e5)."""
        return self.embedding.get("models", {}).get(model_key, {})
