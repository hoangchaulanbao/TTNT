# ⚖️ Hệ thống Hỏi đáp Pháp luật Tiếng Việt

Chatbot pháp luật thông minh sử dụng **RAG (Retrieval-Augmented Generation)** kết hợp **Fine-tuning LLM** mã nguồn mở với **QLoRA**.

## Tổng quan

Hệ thống hỗ trợ tra cứu và tư vấn pháp luật Việt Nam cho người dân, doanh nghiệp và sinh viên luật thông qua giao diện chat thông minh.

### Tính năng chính
- Hỏi đáp pháp luật bằng tiếng Việt tự nhiên
- Trích dẫn nguồn văn bản pháp luật cụ thể (Điều/Khoản/Điểm)
- Hội thoại đa lượt (multi-turn) với hiểu ngữ cảnh
- Phát hiện chuyển đổi chủ đề
- So sánh 3 mô hình LLM (Vistral, PhoGPT, Qwen2.5)
- So sánh RAG vs Non-RAG
- Đánh giá bằng RAGAS framework
- Gợi ý câu hỏi liên quan
- Feedback loop (thumbs up/down)

## Kiến trúc

```
User → Gradio UI → ChatService
                        │
                        ├── QualityGuard (scope check, ambiguity detection)
                        ├── ConversationManager (history, topic detection)
                        ├── Retriever (embed → search → rerank)
                        │       ├── EmbeddingEngine (PhoBERT / multilingual-e5)
                        │       ├── VectorStore (ChromaDB)
                        │       └── Reranker (bge-reranker-v2-m3)
                        └── LLMEngine (Vistral / PhoGPT / Qwen2.5 + QLoRA)
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Cấu trúc dự án

```
vietnamese-legal-qa/
├── src/
│   ├── components/          # Core components
│   │   ├── data_collector.py
│   │   ├── text_processor.py
│   │   ├── embedding_engine.py
│   │   ├── vector_store.py
│   │   ├── retriever.py
│   │   ├── conversation_manager.py
│   │   ├── llm_engine.py
│   │   ├── quality_guard.py
│   │   ├── fine_tuner.py
│   │   └── evaluator.py
│   ├── services/            # Service layer
│   │   ├── chat_service.py
│   │   ├── data_ingestion_service.py
│   │   ├── fine_tuning_service.py
│   │   └── evaluation_service.py
│   ├── models/              # Data models
│   │   └── data_models.py
│   ├── ui/                  # Gradio interface
│   │   └── gradio_app.py
│   └── config/
│       └── settings.py
├── notebooks/               # Jupyter notebooks (execution)
├── data/                    # Data storage
├── models/                  # Trained adapters
├── evaluation/              # Reports
├── config.yaml              # Configuration
├── requirements.txt
└── README.md
```

## Sử dụng

### 1. Thu thập dữ liệu
```python
from src.services.data_ingestion_service import DataIngestionService
from src.config.settings import Settings

settings = Settings.load("config.yaml")
ingestion = DataIngestionService(settings)

# Load từ files đã chuẩn bị
stats = ingestion.ingest_from_files("data/raw", embedding_model="multilingual_e5")
```

### 2. Fine-tuning
```python
from src.services.fine_tuning_service import FineTuningService

ft_service = FineTuningService(settings)
qa_pairs = ft_service.load_dataset("data/alqac")
results = ft_service.finetune_all_models(qa_pairs)
```

### 3. Chạy demo
```python
from src.ui.gradio_app import launch_app
# (sau khi khởi tạo chat_service)
launch_app(chat_service, share=True)
```

## Công nghệ

| Thành phần | Công nghệ |
|---|---|
| LLM | Vistral-7B, PhoGPT-7.5B, Qwen2.5-7B |
| Fine-tuning | QLoRA (4-bit, PEFT, TRL) |
| Embedding | PhoBERT, multilingual-e5-large |
| Vector DB | ChromaDB |
| Reranker | bge-reranker-v2-m3 |
| RAG Framework | LangChain |
| Evaluation | RAGAS |
| UI | Gradio |
| Environment | Google Colab Pro (A100) |

## Dữ liệu

- **ALQAC 2023**: Bộ dữ liệu QA pháp luật tiếng Việt
- **vbpl.vn**: Cơ sở dữ liệu QPPL quốc gia (500-2000 văn bản)

## Đánh giá

Sử dụng RAGAS framework với các metrics:
- **Faithfulness**: Độ trung thực với context
- **Context Recall**: Khả năng truy xuất context đúng
- **Answer Relevancy**: Độ liên quan của câu trả lời

## Lưu ý

⚠️ Hệ thống chỉ mang tính chất tham khảo và nghiên cứu. Vui lòng tham khảo luật sư chuyên nghiệp cho các vấn đề pháp lý quan trọng.
