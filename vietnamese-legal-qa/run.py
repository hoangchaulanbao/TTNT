"""
Quick launcher - Khởi chạy hệ thống hỏi đáp pháp luật.

Usage:
    python run.py              # Launch Gradio demo
    python run.py --cli        # CLI mode (no UI)
    python run.py --share      # Launch with public URL
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config.settings import Settings
from src.components.embedding_engine import EmbeddingEngine
from src.components.vector_store import VectorStoreManager
from src.components.retriever import Retriever
from src.components.conversation_manager import ConversationManager
from src.components.llm_engine import LLMEngine
from src.components.quality_guard import QualityGuard
from src.services.chat_service import ChatService


def initialize_system(settings: Settings, model_key: str = "qwen") -> ChatService:
    """Initialize all components and return ChatService."""
    print("🔧 Initializing components...")

    embedding_engine = EmbeddingEngine(settings)
    vector_store = VectorStoreManager(settings)
    retriever = Retriever(settings, embedding_engine, vector_store)
    conversation_manager = ConversationManager(settings)
    llm_engine = LLMEngine(settings)
    quality_guard = QualityGuard(settings)

    # Load fine-tuned model
    adapter_path = f"models/adapters/{model_key}"
    if os.path.exists(adapter_path):
        llm_engine.load_model(model_key, adapter_path=adapter_path)
    else:
        print(f"⚠️ Adapter not found at {adapter_path}, loading base model...")
        llm_engine.load_model(model_key)

    chat_service = ChatService(
        settings=settings,
        retriever=retriever,
        conversation_manager=conversation_manager,
        llm_engine=llm_engine,
        quality_guard=quality_guard,
    )

    print("✅ System ready!")
    return chat_service


def run_cli(chat_service: ChatService, model_key: str = "qwen"):
    """Run in CLI mode (interactive chat)."""
    session_id = chat_service.new_session()

    print("\n" + "=" * 60)
    print("⚖️  Hệ thống Hỏi đáp Pháp luật Việt Nam")
    print("=" * 60)
    print("Gõ 'quit' để thoát | 'new' để bắt đầu phiên mới | 'summary' để tóm tắt")
    print("=" * 60 + "\n")

    while True:
        user_input = input("👤 Bạn: ").strip()

        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Tạm biệt!")
            break
        if user_input.lower() == "new":
            session_id = chat_service.new_session()
            print("🔄 Phiên mới đã được tạo.\n")
            continue
        if user_input.lower() == "summary":
            summary = chat_service.get_session_summary(session_id)
            print(f"\n{summary}\n")
            continue

        response = chat_service.chat(
            user_message=user_input,
            session_id=session_id,
            model_key=model_key,
        )

        print(f"\n🤖 Bot: {response.answer}")

        if response.sources:
            print(f"\n📎 Nguồn:")
            for s in response.sources[:3]:
                print(f"   - {s.breadcrumb}")

        if response.validity_warnings:
            for w in response.validity_warnings:
                print(f"   {w}")

        if response.suggestions:
            print(f"\n💡 Gợi ý:")
            for s in response.suggestions:
                print(f"   • {s}")

        print(f"\n   ⏱️ {response.inference_time:.2f}s\n")


def main():
    parser = argparse.ArgumentParser(description="Vietnamese Legal QA System")
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode")
    parser.add_argument("--share", action="store_true", help="Create public Gradio URL")
    parser.add_argument("--model", default="qwen", choices=["qwen", "vistral", "phogpt"],
                       help="LLM model to use")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    settings = Settings.load(args.config)
    chat_service = initialize_system(settings, args.model)

    if args.cli:
        run_cli(chat_service, args.model)
    else:
        from src.ui.gradio_app import create_app
        app = create_app(chat_service)
        app.launch(share=args.share)


if __name__ == "__main__":
    main()
