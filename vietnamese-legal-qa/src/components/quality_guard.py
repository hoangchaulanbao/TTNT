"""Quality Guard - Kiểm soát chất lượng và xử lý edge cases."""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.models.data_models import AmbiguityResult, ValidityWarning, DocumentMetadata
from src.config.settings import Settings


class QualityGuard:
    """Kiểm soát chất lượng: out-of-scope, ambiguity, validity, feedback."""

    # Legal keywords for scope detection
    LEGAL_KEYWORDS = [
        "luật", "pháp luật", "quy định", "điều", "khoản", "điểm",
        "nghị định", "thông tư", "quyết định", "nghị quyết",
        "quyền", "nghĩa vụ", "trách nhiệm", "xử phạt", "vi phạm",
        "hợp đồng", "tài sản", "thừa kế", "ly hôn", "kết hôn",
        "lao động", "bảo hiểm", "thuế", "doanh nghiệp", "đầu tư",
        "hình sự", "dân sự", "hành chính", "tố tụng",
        "bồi thường", "khiếu nại", "tố cáo", "tranh chấp",
        "giấy phép", "đăng ký", "thủ tục", "hồ sơ",
        "đất đai", "nhà ở", "xây dựng", "môi trường",
        "sở hữu trí tuệ", "bản quyền", "thương hiệu",
        "an ninh", "quốc phòng", "giao thông",
    ]

    # Non-legal topics
    NON_LEGAL_KEYWORDS = [
        "nấu ăn", "công thức", "thời tiết", "bóng đá", "phim",
        "nhạc", "game", "code", "lập trình", "python", "java",
        "toán", "vật lý", "hóa học", "y tế", "bệnh",
        "du lịch", "ẩm thực", "thể thao", "giải trí",
    ]

    def __init__(self, settings: Settings):
        self.settings = settings
        self.qg_config = settings.quality_guard
        self.feedback_dir = Path(settings.data.get("feedback_dir", "data/feedback"))
        self.feedback_dir.mkdir(parents=True, exist_ok=True)

    def is_legal_question(self, query: str) -> bool:
        """
        Kiểm tra câu hỏi có thuộc phạm vi pháp luật không.
        
        Args:
            query: Câu hỏi người dùng
            
        Returns:
            True nếu là câu hỏi pháp luật
        """
        query_lower = query.lower()

        # Check for non-legal keywords first
        non_legal_score = sum(1 for kw in self.NON_LEGAL_KEYWORDS if kw in query_lower)
        if non_legal_score >= 2:
            return False

        # Check for legal keywords
        legal_score = sum(1 for kw in self.LEGAL_KEYWORDS if kw in query_lower)

        # Threshold
        threshold = self.qg_config.get("legal_confidence_threshold", 0.5)

        # Simple scoring: at least 1 legal keyword and no strong non-legal signal
        if legal_score >= 1 and non_legal_score == 0:
            return True

        # Ambiguous case: check if it could be legal-adjacent
        if legal_score == 0 and non_legal_score == 0:
            # Could be a follow-up question without explicit legal terms
            # Allow it (will be handled by conversation context)
            return True

        return legal_score > non_legal_score

    def is_ambiguous_question(self, query: str) -> AmbiguityResult:
        """
        Phát hiện câu hỏi mơ hồ.
        
        Args:
            query: Câu hỏi
            
        Returns:
            AmbiguityResult
        """
        min_words = self.qg_config.get("ambiguity_min_words", 5)
        words = query.split()

        # Too short
        if len(words) < min_words:
            # Check if it has specific legal terms
            has_specific = any(
                term in query.lower()
                for term in ["điều", "khoản", "số", "nghị định", "thông tư"]
            )
            if not has_specific:
                options = self._generate_clarification_options(query)
                if options:
                    return AmbiguityResult(
                        is_ambiguous=True,
                        reason="Câu hỏi quá ngắn, cần thêm thông tin cụ thể",
                        clarification_options=options,
                    )

        # Broad terms without qualifier
        broad_terms = {
            "hợp đồng": ["hợp đồng dân sự", "hợp đồng lao động", "hợp đồng thương mại"],
            "thuế": ["thuế thu nhập cá nhân", "thuế doanh nghiệp", "thuế VAT"],
            "xử phạt": ["xử phạt hành chính", "xử phạt hình sự", "xử phạt giao thông"],
            "đăng ký": ["đăng ký kinh doanh", "đăng ký kết hôn", "đăng ký đất đai"],
            "bồi thường": ["bồi thường thiệt hại", "bồi thường lao động", "bồi thường hợp đồng"],
        }

        query_lower = query.lower()
        for broad_term, specifics in broad_terms.items():
            if broad_term in query_lower:
                # Check if already specific
                is_specific = any(s in query_lower for s in specifics)
                if not is_specific and len(words) < 8:
                    return AmbiguityResult(
                        is_ambiguous=True,
                        reason=f"Thuật ngữ '{broad_term}' có nhiều nghĩa trong pháp luật",
                        clarification_options=[
                            f"Bạn muốn hỏi về {s}?" for s in specifics[:4]
                        ],
                    )

        return AmbiguityResult(is_ambiguous=False)

    def check_document_validity(self, metadata: dict) -> Optional[ValidityWarning]:
        """
        Kiểm tra văn bản có thể đã hết hiệu lực.
        
        Args:
            metadata: Document metadata dict
            
        Returns:
            ValidityWarning nếu cần cảnh báo, None nếu OK
        """
        warning_years = self.qg_config.get("validity_warning_years", 3)

        issued_date_str = metadata.get("issued_date", "")
        if not issued_date_str:
            return None

        try:
            if isinstance(issued_date_str, str):
                issued_date = datetime.fromisoformat(issued_date_str)
            else:
                issued_date = issued_date_str
        except (ValueError, TypeError):
            return None

        age_years = (datetime.now() - issued_date).days / 365.25

        if age_years > warning_years:
            doc_number = metadata.get("document_number", "Không rõ")
            return ValidityWarning(
                document_number=doc_number,
                issued_date=issued_date,
                age_years=age_years,
                warning_text=(
                    f"⚠️ Văn bản {doc_number} ban hành ngày "
                    f"{issued_date.strftime('%d/%m/%Y')} "
                    f"(cách đây {age_years:.0f} năm). "
                    f"Có thể đã được sửa đổi, bổ sung. "
                    f"Vui lòng kiểm tra phiên bản mới nhất."
                ),
                severity="warning" if age_years > 5 else "info",
            )

        return None

    def generate_rejection_message(self, query: str) -> str:
        """Generate polite rejection message with context-aware suggestions."""
        # Detect what the user might have intended
        query_lower = query.lower()
        
        # Try to find a legal angle
        legal_angles = {
            "nấu ăn": "an toàn thực phẩm, giấy phép kinh doanh ẩm thực",
            "bệnh": "quyền khám chữa bệnh, bảo hiểm y tế, trách nhiệm y tế",
            "code": "sở hữu trí tuệ phần mềm, hợp đồng công nghệ",
            "lập trình": "bản quyền phần mềm, hợp đồng lao động IT",
            "game": "quy định về trò chơi điện tử, bảo vệ trẻ em trên mạng",
            "du lịch": "quyền của khách du lịch, giấy phép lữ hành",
            "y tế": "quyền bệnh nhân, trách nhiệm bồi thường y tế, BHYT",
            "thể thao": "quy định về hoạt động thể thao, hợp đồng vận động viên",
        }

        suggestion_topic = None
        for keyword, legal_topic in legal_angles.items():
            if keyword in query_lower:
                suggestion_topic = legal_topic
                break

        base_message = (
            "Xin lỗi, tôi là trợ lý pháp luật và chỉ có thể hỗ trợ các câu hỏi "
            "liên quan đến pháp luật Việt Nam.\n\n"
        )

        if suggestion_topic:
            base_message += (
                f"💡 **Gợi ý**: Nếu bạn quan tâm đến khía cạnh pháp lý liên quan, "
                f"tôi có thể giúp về: **{suggestion_topic}**\n\n"
            )

        base_message += (
            "📌 **Một số chủ đề tôi có thể hỗ trợ:**\n"
            "• Luật lao động (hợp đồng, sa thải, BHXH, lương)\n"
            "• Luật dân sự (hợp đồng, tài sản, thừa kế, bồi thường)\n"
            "• Luật hôn nhân gia đình (kết hôn, ly hôn, nuôi con)\n"
            "• Luật doanh nghiệp (thành lập, thuế, giải thể)\n"
            "• Luật đất đai (sổ đỏ, chuyển nhượng, thu hồi)\n"
            "• Luật hình sự (tội phạm, hình phạt, tố tụng)\n\n"
            "Bạn có câu hỏi pháp lý nào cần tư vấn không?"
        )

        return base_message

    def record_feedback(
        self, question: str, answer: str, rating: int, comment: str = ""
    ) -> None:
        """
        Lưu feedback từ người dùng.
        
        Args:
            question: Câu hỏi
            answer: Câu trả lời
            rating: 0 (bad) hoặc 1 (good)
            comment: Nhận xét (optional)
        """
        feedback = {
            "question": question,
            "answer": answer[:500],  # Truncate
            "rating": rating,
            "comment": comment,
            "timestamp": datetime.now().isoformat(),
        }

        # Append to feedback file
        feedback_file = self.feedback_dir / "feedback_log.jsonl"
        with open(feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback, ensure_ascii=False) + "\n")

    def get_feedback_stats(self) -> dict:
        """Get feedback statistics."""
        feedback_file = self.feedback_dir / "feedback_log.jsonl"
        if not feedback_file.exists():
            return {"total": 0, "positive": 0, "negative": 0, "satisfaction_rate": 0}

        total = 0
        positive = 0
        with open(feedback_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    total += 1
                    if data.get("rating", 0) == 1:
                        positive += 1

        return {
            "total": total,
            "positive": positive,
            "negative": total - positive,
            "satisfaction_rate": positive / total if total > 0 else 0,
        }

    def _generate_clarification_options(self, query: str) -> List[str]:
        """Generate clarification options for ambiguous queries."""
        query_lower = query.lower()

        # Common broad topics
        if "luật" in query_lower or "pháp luật" in query_lower:
            return [
                "Luật dân sự (hợp đồng, tài sản, thừa kế)",
                "Luật lao động (hợp đồng lao động, BHXH)",
                "Luật doanh nghiệp (thành lập, thuế)",
                "Luật hôn nhân gia đình",
            ]

        return []
