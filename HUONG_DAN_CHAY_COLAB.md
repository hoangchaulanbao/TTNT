# 🚀 Hướng dẫn chạy trên Google Colab

## Bước 1: Chuẩn bị

### 1.1 Yêu cầu
- Tài khoản Google Colab **Pro** hoặc **Pro+** (cần GPU A100)
- Google Drive có ít nhất **50GB** trống
- Dữ liệu ALQAC 2023 (download trước)

### 1.2 Upload project lên Google Drive

```
Google Drive/
└── vietnamese-legal-qa/        ← Upload toàn bộ folder này
    ├── src/
    ├── notebooks/
    ├── config.yaml
    ├── requirements.txt
    └── ...
```

**Cách upload:**
1. Zip toàn bộ folder `vietnamese-legal-qa/`
2. Upload file zip lên Google Drive
3. Giải nén trên Drive (chuột phải → Open with → Connected apps → Zip Extractor)

Hoặc dùng GitHub:
1. Push code lên GitHub repo
2. Clone từ Colab (sẽ hướng dẫn ở dưới)

---

## Bước 2: Mở Google Colab

1. Vào https://colab.research.google.com
2. **Runtime → Change runtime type:**
   - Hardware accelerator: **GPU**
   - GPU type: **A100** (cần Colab Pro/Pro+)
   - High-RAM: **Bật**
3. Click **Save**

---

## Bước 3: Chạy từng Notebook theo thứ tự

### ⚠️ QUAN TRỌNG: Mỗi notebook chạy trong 1 session riêng

Do giới hạn VRAM, nên:
- Notebook 01, 02: Chạy chung 1 session (không cần GPU mạnh)
- Notebook 03: Chạy riêng cho TỪNG model (restart runtime giữa các models)
- Notebook 04, 05, 06: Chạy chung 1 session (load 1 model)

---

### 📥 Notebook 01: Thu thập dữ liệu (15-30 phút)

Mở file: `notebooks/01_data_collection.ipynb`

**Việc cần làm trước khi chạy:**
1. Upload dữ liệu ALQAC 2023 vào `data/alqac/` trên Drive
2. Upload văn bản pháp luật (JSON) vào `data/raw/` trên Drive

**Nếu chưa có dữ liệu:** Notebook sẽ tạo sample data để test pipeline trước.

**Kết quả:** `data/raw/`, `data/processed/`, `data/alqac/` có dữ liệu

---

### 🔢 Notebook 02: Embedding & Indexing (10-20 phút)

Mở file: `notebooks/02_embedding_indexing.ipynb`

**Việc cần làm:** Chạy tất cả cells theo thứ tự

**Kết quả:** 
- `data/chroma_db/` có 2 collections (legal_docs_e5, legal_docs_phobert)
- Test retrieval thành công

---

### 🎯 Notebook 03: Fine-tuning (2-4 giờ/model)

Mở file: `notebooks/03_fine_tuning.ipynb`

**⚠️ ĐÂY LÀ BƯỚC TỐN THỜI GIAN NHẤT**

**Cách chạy tối ưu:**

```
Lần 1: Fine-tune Qwen2.5-7B
  → Chạy Cell 1-3 (setup + Qwen)
  → Chạy Cell cuối (backup to Drive)
  → Runtime → Restart runtime

Lần 2: Fine-tune Vistral-7B  
  → Chạy Cell 1-2 (setup + load data)
  → Chạy Cell 4 (Vistral)
  → Chạy Cell cuối (backup)
  → Runtime → Restart runtime

Lần 3: Fine-tune PhoGPT-7.5B
  → Chạy Cell 1-2 (setup + load data)
  → Chạy Cell 5 (PhoGPT)
  → Chạy Cell cuối (backup)
```

**Kết quả:** `models/adapters/qwen/`, `models/adapters/vistral/`, `models/adapters/phogpt/`

**Tip:** Nếu bị disconnect giữa chừng, adapters đã backup trên Drive vẫn an toàn.

---

### 🔄 Notebook 04: RAG Pipeline Test (10-15 phút)

Mở file: `notebooks/04_rag_pipeline.ipynb`

**Việc cần làm:** Chạy tất cả cells

**Test bao gồm:**
- Basic Q&A
- Multi-turn conversation (cùng chủ đề)
- Topic change (chuyển lĩnh vực)
- Out-of-scope rejection
- Ambiguity detection
- Session summary

**Kết quả:** Tất cả tests PASS, pipeline hoạt động end-to-end

---

### 📊 Notebook 05: Evaluation (30-60 phút)

Mở file: `notebooks/05_evaluation.ipynb`

**Việc cần làm:** Chạy tất cả cells

**Kết quả:**
- RAGAS scores (faithfulness, context_recall, answer_relevancy)
- RAG vs Non-RAG comparison table
- Reports trong `evaluation/reports/`

**Dùng cho báo cáo:** Copy bảng kết quả vào luận văn/đồ án

---

### 🚀 Notebook 06: Demo UI (2-3 phút)

Mở file: `notebooks/06_demo.ipynb`

**Việc cần làm:** Chạy 2 cells

**Kết quả:** 
- Gradio app chạy với **public URL** (share=True)
- URL dạng: `https://xxxxx.gradio.live`
- Chia sẻ URL này cho giảng viên/hội đồng để demo

**Lưu ý:** URL public chỉ tồn tại khi notebook đang chạy. Đừng tắt notebook khi demo!

---

## Bước 4: Lấy kết quả cho báo cáo

### Screenshots cần chụp:
1. Giao diện chat (tab Hỏi đáp)
2. So sánh RAG vs Non-RAG (tab So sánh)
3. Multi-turn conversation (nhiều lượt hỏi)
4. Out-of-scope rejection
5. Gợi ý câu hỏi liên quan

### Bảng kết quả cần copy:
1. RAGAS scores cho từng model
2. RAG vs Non-RAG comparison
3. Embedding comparison (PhoBERT vs E5)
4. Response time statistics

---

## Troubleshooting

### ❌ "CUDA out of memory"
```python
# Restart runtime rồi chạy lại
# Hoặc giảm batch_size trong config.yaml:
# fine_tuning.training.batch_size: 2  (thay vì 4)
```

### ❌ "Module not found"
```python
import sys
sys.path.insert(0, '/content/vietnamese-legal-qa')
```

### ❌ "Connection to ChromaDB failed"
```python
!rm -rf data/chroma_db
!mkdir -p data/chroma_db
# Chạy lại notebook 02
```

### ❌ Colab bị disconnect giữa fine-tuning
- Adapters đã backup trên Drive → không mất
- Restart runtime, load adapter từ Drive:
```python
import shutil
shutil.copytree('/content/drive/MyDrive/legal-qa-adapters/qwen', 'models/adapters/qwen')
```

### ❌ Gradio không tạo được public URL
```python
# Thử lại với:
app.launch(share=True, debug=False)
# Hoặc dùng ngrok:
# !pip install pyngrok
# from pyngrok import ngrok
# ngrok.set_auth_token("YOUR_TOKEN")
```

---

## Timeline ước tính

| Bước | Thời gian | GPU cần |
|---|---|---|
| Notebook 01 | 15-30 phút | Không |
| Notebook 02 | 10-20 phút | Không (CPU đủ) |
| Notebook 03 (Qwen) | 2-4 giờ | A100 |
| Notebook 03 (Vistral) | 2-4 giờ | A100 |
| Notebook 03 (PhoGPT) | 2-4 giờ | A100 |
| Notebook 04 | 10-15 phút | A100 |
| Notebook 05 | 30-60 phút | A100 |
| Notebook 06 | 2-3 phút | A100 |
| **TỔNG** | **~8-14 giờ** | |

**Tip tiết kiệm thời gian:** 
- Chạy notebook 01-02 trên CPU runtime (miễn phí)
- Chỉ dùng GPU A100 cho notebook 03-06
- Fine-tune 1 model trước (Qwen), demo xong rồi fine-tune 2 model còn lại sau

---

## Cấu trúc thư mục sau khi chạy xong

```
vietnamese-legal-qa/
├── data/
│   ├── raw/              ← Văn bản pháp luật gốc
│   ├── processed/        ← Chunks đã xử lý
│   ├── alqac/            ← ALQAC 2023 dataset
│   ├── qa_dataset/       ← Q&A cho fine-tuning
│   ├── chroma_db/        ← Vector database
│   └── feedback/         ← User feedback logs
├── models/
│   └── adapters/
│       ├── qwen/         ← LoRA adapter Qwen2.5
│       ├── vistral/      ← LoRA adapter Vistral
│       └── phogpt/       ← LoRA adapter PhoGPT
├── evaluation/
│   └── reports/          ← RAGAS evaluation reports
├── src/                  ← Source code
├── notebooks/            ← Jupyter notebooks
└── config.yaml           ← Configuration
```
