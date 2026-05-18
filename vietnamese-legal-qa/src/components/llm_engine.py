"""LLM Engine - Quản lý và gọi các LLM models."""

import time
from typing import List, Optional, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from src.models.data_models import ConversationTurn
from src.config.settings import Settings


class LLMEngine:
    """Quản lý và generate responses từ fine-tuned LLM models."""

    SYSTEM_PROMPT = (
        "Bạn là trợ lý pháp luật Việt Nam chuyên nghiệp. Nhiệm vụ của bạn là trả lời "
        "câu hỏi pháp luật một cách chi tiết, dễ hiểu và chính xác.\n\n"
        "QUY TẮC TRẢ LỜI:\n"
        "1. Trích dẫn cụ thể: Luôn nêu rõ Điều/Khoản/Điểm và số hiệu văn bản (VD: Điều 20 Bộ luật Lao động 2019, số 45/2019/QH14)\n"
        "2. Giải thích dễ hiểu: Sau khi trích dẫn, giải thích bằng ngôn ngữ đơn giản để người không chuyên cũng hiểu\n"
        "3. Có cấu trúc: Sử dụng đánh số, gạch đầu dòng để trình bày rõ ràng\n"
        "4. Đầy đủ: Nêu cả quyền, nghĩa vụ, điều kiện, ngoại lệ (nếu có)\n"
        "5. Thực tế: Đưa ví dụ minh họa khi cần thiết\n"
        "6. Trung thực: Nếu không tìm thấy thông tin trong ngữ cảnh, nói rõ 'Tôi không tìm thấy thông tin cụ thể về vấn đề này trong cơ sở dữ liệu hiện tại'\n"
        "7. Cảnh báo: Nếu vấn đề phức tạp hoặc có rủi ro cao, khuyên người dùng tham khảo luật sư chuyên nghiệp\n"
        "8. Ngữ cảnh: Khi trả lời câu hỏi tiếp theo trong cùng chủ đề, tham chiếu lại nội dung đã thảo luận trước đó"
    )

    def __init__(self, settings: Settings):
        self.settings = settings
        self._models: Dict[str, dict] = {}  # {model_key: {"model": ..., "tokenizer": ...}}

    def load_model(self, model_key: str, adapter_path: Optional[str] = None) -> None:
        """
        Load LLM model (with optional LoRA adapter).
        
        Args:
            model_key: Key trong config (vistral, phogpt, qwen)
            adapter_path: Path to LoRA adapter (None = base model)
        """
        if model_key in self._models:
            return

        model_config = self.settings.get_llm_model(model_key)
        model_name = model_config.get("name")

        print(f"[LLMEngine] Loading model: {model_name}...")

        # 4-bit quantization for inference
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load LoRA adapter if provided
        if adapter_path:
            print(f"[LLMEngine] Loading adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)

        self._models[model_key] = {
            "model": model,
            "tokenizer": tokenizer,
            "config": model_config,
        }
        print(f"[LLMEngine] Model loaded: {model_key}")

    def generate(
        self,
        query: str,
        context: List[str],
        model_key: str = "qwen",
        history: Optional[List[ConversationTurn]] = None,
        max_tokens: int = None,
    ) -> str:
        """
        Generate câu trả lời với RAG context.
        
        Args:
            query: Câu hỏi người dùng
            context: Retrieved context documents
            model_key: Model để sử dụng
            history: Lịch sử hội thoại (cho prompt)
            max_tokens: Max tokens to generate
            
        Returns:
            Generated answer text
        """
        if model_key not in self._models:
            raise ValueError(f"Model '{model_key}' not loaded. Call load_model() first.")

        model_data = self._models[model_key]
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        config = model_data["config"]

        if max_tokens is None:
            max_tokens = config.get("max_tokens", 1024)

        # Build prompt
        prompt = self._build_rag_prompt(query, context, history)

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=config.get("temperature", 0.1),
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode (only new tokens)
        input_length = inputs["input_ids"].shape[1]
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

        return response.strip()

    def generate_without_rag(self, query: str, model_key: str = "qwen") -> str:
        """
        Generate câu trả lời KHÔNG có RAG context (cho so sánh).
        
        Args:
            query: Câu hỏi
            model_key: Model key
            
        Returns:
            Generated answer (no retrieval)
        """
        if model_key not in self._models:
            raise ValueError(f"Model '{model_key}' not loaded.")

        model_data = self._models[model_key]
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        config = model_data["config"]

        # Simple prompt without context
        prompt = (
            f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.get("max_tokens", 1024),
                temperature=config.get("temperature", 0.1),
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_length = inputs["input_ids"].shape[1]
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

        return response.strip()

    def suggest_related_questions(
        self, query: str, response: str, context: List[str]
    ) -> List[str]:
        """
        Gợi ý 2-3 câu hỏi liên quan dựa trên nội dung câu trả lời.
        
        Strategy: Phân tích response để tìm các khía cạnh chưa được khai thác.
        """
        suggestions = []
        response_lower = response.lower()
        query_lower = query.lower()

        # Extract legal terms from response
        import re
        articles = re.findall(r"Điều\s+\d+", response)
        laws = re.findall(r"(?:Luật|Nghị định|Thông tư|Bộ luật)\s+[^\n,\.]{5,30}", response)

        # === Pattern-based suggestions ===
        
        # Quyền → hỏi về nghĩa vụ
        if "quyền" in response_lower and "nghĩa vụ" not in query_lower:
            suggestions.append("Nghĩa vụ tương ứng với quyền này là gì?")
        
        # Nghĩa vụ → hỏi về quyền
        if "nghĩa vụ" in response_lower and "quyền" not in query_lower:
            suggestions.append("Quyền lợi được hưởng khi thực hiện nghĩa vụ này?")

        # Xử phạt → hỏi mức phạt cụ thể
        if any(kw in response_lower for kw in ["xử phạt", "phạt tiền", "hình phạt"]):
            if "mức" not in query_lower:
                suggestions.append("Mức phạt cụ thể cho từng trường hợp vi phạm?")

        # Thủ tục → hỏi hồ sơ
        if "thủ tục" in response_lower and "hồ sơ" not in query_lower:
            suggestions.append("Hồ sơ cần chuẩn bị gồm những giấy tờ gì?")
        
        # Hồ sơ → hỏi thời gian
        if "hồ sơ" in response_lower and "thời gian" not in query_lower:
            suggestions.append("Thời gian xử lý hồ sơ là bao lâu?")

        # Điều kiện → hỏi trường hợp không đủ điều kiện
        if "điều kiện" in response_lower:
            suggestions.append("Nếu không đủ điều kiện thì hậu quả pháp lý là gì?")

        # Hợp đồng → hỏi vi phạm/chấm dứt
        if "hợp đồng" in response_lower:
            if "chấm dứt" not in query_lower and "vi phạm" not in query_lower:
                suggestions.append("Trường hợp nào được đơn phương chấm dứt hợp đồng?")

        # Thừa kế → hỏi về hàng thừa kế
        if "thừa kế" in response_lower and "hàng" not in query_lower:
            suggestions.append("Thứ tự hàng thừa kế được quy định như thế nào?")

        # Có trích dẫn Điều → hỏi chi tiết
        if articles and len(suggestions) < 2:
            suggestions.append(f"Giải thích chi tiết hơn về {articles[0]}?")

        # Có nhắc đến luật → hỏi văn bản hướng dẫn
        if laws and len(suggestions) < 3:
            suggestions.append("Có nghị định hoặc thông tư nào hướng dẫn chi tiết hơn không?")

        # === Fallback suggestions theo chủ đề ===
        topic_suggestions = {
            "lao động": [
                "Quyền lợi khi bị sa thải trái pháp luật?",
                "Chế độ bảo hiểm xã hội cho người lao động?",
            ],
            "hôn nhân": [
                "Tài sản chung vợ chồng được chia như thế nào khi ly hôn?",
                "Quyền nuôi con sau ly hôn được xác định thế nào?",
            ],
            "doanh nghiệp": [
                "Trách nhiệm của người đại diện pháp luật?",
                "Thủ tục giải thể doanh nghiệp?",
            ],
            "đất đai": [
                "Thủ tục cấp sổ đỏ lần đầu?",
                "Quyền chuyển nhượng quyền sử dụng đất?",
            ],
        }

        for topic, topic_sugs in topic_suggestions.items():
            if topic in response_lower or topic in query_lower:
                for s in topic_sugs:
                    if s.lower() not in query_lower and len(suggestions) < 3:
                        suggestions.append(s)
                break

        # Generic fallbacks
        generic = [
            "Có ngoại lệ nào cho quy định này không?",
            "Thời hiệu áp dụng quy định này là bao lâu?",
            "Cơ quan nào có thẩm quyền giải quyết vấn đề này?",
        ]

        while len(suggestions) < 2:
            if generic:
                suggestions.append(generic.pop(0))
            else:
                break

        # Deduplicate and limit
        seen = set()
        unique = []
        for s in suggestions:
            if s.lower() not in seen:
                seen.add(s.lower())
                unique.append(s)
        
        return unique[:3]

    def get_available_models(self) -> List[str]:
        """Get list of loaded model keys."""
        return list(self._models.keys())

    def _build_rag_prompt(
        self,
        query: str,
        context: List[str],
        history: Optional[List[ConversationTurn]] = None,
    ) -> str:
        """Build prompt with RAG context and conversation history."""
        # Format context
        context_text = "\n\n".join(
            f"[Nguồn {i+1}]: {doc}" for i, doc in enumerate(context[:5])
        )

        # Format history (last 3 turns)
        history_text = ""
        if history:
            recent = history[-3:]
            history_parts = []
            for turn in recent:
                history_parts.append(f"User: {turn.user_message}")
                history_parts.append(f"Bot: {turn.bot_response[:200]}")
            history_text = "\n".join(history_parts)

        # Build full prompt (ChatML format)
        prompt = f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n"

        if history_text:
            prompt += (
                f"<|im_start|>user\n"
                f"[Lịch sử hội thoại trước đó - dùng để hiểu ngữ cảnh]\n"
                f"{history_text}<|im_end|>\n"
            )

        prompt += (
            f"<|im_start|>user\n"
            f"Ngữ cảnh pháp luật (trích từ văn bản QPPL):\n{context_text}\n\n"
            f"Câu hỏi của người dùng: {query}\n\n"
            f"Hãy trả lời chi tiết, có cấu trúc, trích dẫn Điều/Khoản cụ thể, "
            f"và giải thích bằng ngôn ngữ dễ hiểu.<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        return prompt
