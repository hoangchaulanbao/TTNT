"""QA Generator - Tạo thêm cặp Q&A từ corpus pháp luật cho fine-tuning."""

import re
import uuid
import random
from typing import List

from src.models.data_models import Chunk, QAPair, InstructionSample
from src.config.settings import Settings


class QAGenerator:
    """Tự động tạo cặp Q&A từ corpus pháp luật để bổ sung dataset fine-tuning."""

    # Templates câu hỏi theo loại
    QUESTION_TEMPLATES = {
        "what": [
            "Theo {doc}, {subject} là gì?",
            "{subject} được quy định như thế nào trong {doc}?",
            "Nội dung {article} {doc} quy định gì?",
        ],
        "condition": [
            "Điều kiện để {action} theo {doc} là gì?",
            "Khi nào thì {subject} được {action}?",
            "Các trường hợp {action} theo quy định pháp luật?",
        ],
        "procedure": [
            "Thủ tục {action} theo {doc} gồm những bước nào?",
            "Hồ sơ cần thiết để {action} gồm những gì?",
            "Quy trình {action} được thực hiện như thế nào?",
        ],
        "penalty": [
            "Mức xử phạt khi {violation} theo {doc}?",
            "Hình thức xử lý khi vi phạm {subject}?",
        ],
        "right": [
            "Quyền của {subject} theo {doc} là gì?",
            "{subject} có những quyền gì theo quy định?",
        ],
        "obligation": [
            "Nghĩa vụ của {subject} theo {doc}?",
            "{subject} phải thực hiện những nghĩa vụ gì?",
        ],
    }

    def __init__(self, settings: Settings):
        self.settings = settings

    def generate_from_chunks(
        self, chunks: List[Chunk], num_pairs: int = 1000
    ) -> List[QAPair]:
        """
        Tạo cặp Q&A từ chunks pháp luật.
        
        Strategy:
        - Mỗi chunk chứa 1 Điều luật → tạo 1-3 câu hỏi
        - Câu hỏi dựa trên nội dung chunk
        - Câu trả lời = nội dung chunk + trích dẫn
        
        Args:
            chunks: Danh sách chunks đã indexed
            num_pairs: Số cặp Q&A cần tạo
            
        Returns:
            Danh sách QAPair
        """
        qa_pairs = []
        random.shuffle(chunks)

        for chunk in chunks:
            if len(qa_pairs) >= num_pairs:
                break

            # Tạo Q&A từ chunk
            pairs = self._generate_from_single_chunk(chunk)
            qa_pairs.extend(pairs)

        # Shuffle và limit
        random.shuffle(qa_pairs)
        return qa_pairs[:num_pairs]

    def format_for_training(
        self, qa_pairs: List[QAPair], format_type: str = "chatml"
    ) -> List[InstructionSample]:
        """
        Format Q&A pairs thành instruction samples cho fine-tuning.
        
        Args:
            qa_pairs: Danh sách Q&A
            format_type: 'chatml' hoặc 'alpaca'
            
        Returns:
            Danh sách InstructionSample
        """
        system_prompt = (
            "Bạn là trợ lý pháp luật Việt Nam. Hãy trả lời câu hỏi dựa trên "
            "kiến thức pháp luật, trích dẫn điều khoản cụ thể khi có thể. "
            "Sử dụng ngôn ngữ dễ hiểu và chính xác."
        )

        samples = []
        for qa in qa_pairs:
            sample = InstructionSample(
                system_prompt=system_prompt,
                user_message=qa.question,
                assistant_message=qa.answer,
            )
            samples.append(sample)

        return samples

    def _generate_from_single_chunk(self, chunk: Chunk) -> List[QAPair]:
        """Generate Q&A pairs from a single chunk."""
        pairs = []
        content = chunk.content
        doc_ref = chunk.document_title or chunk.document_number

        # Detect content type and generate appropriate questions
        if chunk.article_number:
            article_ref = f"Điều {chunk.article_number}"
        else:
            article_ref = "quy định"

        # Simple Q&A: "Nội dung Điều X quy định gì?"
        question = f"Nội dung {article_ref} {doc_ref} quy định gì?"
        answer = f"Theo {article_ref} {doc_ref}:\n\n{content}"

        pairs.append(QAPair(
            id=str(uuid.uuid4()),
            question=question,
            answer=answer,
            context=content,
            source_document=chunk.document_number,
            category=chunk.category,
            source="generated",
        ))

        # Detect keywords for more specific questions
        content_lower = content.lower()

        if any(kw in content_lower for kw in ["điều kiện", "phải có", "đủ"]):
            q = f"Điều kiện theo {article_ref} {doc_ref} là gì?"
            pairs.append(QAPair(
                id=str(uuid.uuid4()),
                question=q,
                answer=answer,
                context=content,
                source_document=chunk.document_number,
                category=chunk.category,
                source="generated",
            ))

        if any(kw in content_lower for kw in ["quyền", "được"]):
            q = f"Quyền được quy định tại {article_ref} {doc_ref}?"
            pairs.append(QAPair(
                id=str(uuid.uuid4()),
                question=q,
                answer=answer,
                context=content,
                source_document=chunk.document_number,
                category=chunk.category,
                source="generated",
            ))

        if any(kw in content_lower for kw in ["cấm", "không được", "vi phạm", "phạt"]):
            q = f"Những hành vi bị cấm theo {article_ref} {doc_ref}?"
            pairs.append(QAPair(
                id=str(uuid.uuid4()),
                question=q,
                answer=answer,
                context=content,
                source_document=chunk.document_number,
                category=chunk.category,
                source="generated",
            ))

        return pairs[:3]  # Max 3 per chunk
