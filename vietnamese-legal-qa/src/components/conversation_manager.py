"""Conversation Manager - Quản lý lịch sử hội thoại multi-turn."""

import re
import uuid
from datetime import datetime
from typing import List, Optional, Dict

from src.models.data_models import ConversationTurn, RankedDocument
from src.config.settings import Settings


class ConversationManager:
    """Quản lý conversation history, topic detection, reference resolution."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.conv_config = settings.conversation
        self.max_turns = self.conv_config.get("max_history_turns", 10)
        self.lookback = self.conv_config.get("reference_lookback_turns", 3)

        # Sessions storage (in-memory for demo)
        self._sessions: Dict[str, List[ConversationTurn]] = {}

    def new_session(self) -> str:
        """Create a new conversation session."""
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = []
        return session_id

    def add_turn(
        self,
        session_id: str,
        user_message: str,
        bot_response: str,
        sources: List[RankedDocument] = None,
        suggestions: List[str] = None,
    ) -> None:
        """
        Thêm một lượt hội thoại vào history.
        
        Args:
            session_id: ID phiên hội thoại
            user_message: Tin nhắn người dùng
            bot_response: Phản hồi bot
            sources: Nguồn tham chiếu
            suggestions: Câu hỏi gợi ý
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = []

        turn = ConversationTurn(
            turn_id=str(uuid.uuid4()),
            user_message=user_message,
            bot_response=bot_response,
            sources=sources or [],
            suggestions=suggestions or [],
            timestamp=datetime.now(),
        )

        self._sessions[session_id].append(turn)

        # Trim to max turns
        if len(self._sessions[session_id]) > self.max_turns:
            self._sessions[session_id] = self._sessions[session_id][-self.max_turns:]

    def get_history(self, session_id: str, max_turns: int = None) -> List[ConversationTurn]:
        """Get conversation history for a session."""
        if session_id not in self._sessions:
            return []

        history = self._sessions[session_id]
        if max_turns:
            return history[-max_turns:]
        return history

    def resolve_references(self, query: str, history: List[ConversationTurn]) -> str:
        """
        Giải quyết tham chiếu ngầm trong query.
        
        Ví dụ:
        - "Vậy nếu hết hạn thì sao?" → "Vậy nếu hết hạn hợp đồng lao động thì sao?"
        - "Điều đó áp dụng cho ai?" → "Điều [X] áp dụng cho ai?"
        
        Args:
            query: Query có thể chứa tham chiếu
            history: Lịch sử hội thoại
            
        Returns:
            Query đã resolve
        """
        if not history:
            return query

        # Reference patterns in Vietnamese
        reference_patterns = [
            r"\b(nó|điều đó|vấn đề đó|trường hợp này|ở trên|như vậy)\b",
            r"\b(vậy|thế)\b(?!\s+nào)",  # "vậy" but not "vậy nào"
        ]

        has_reference = any(
            re.search(pattern, query, re.IGNORECASE)
            for pattern in reference_patterns
        )

        # Check if query lacks subject/context
        short_query = len(query.split()) < 8
        no_legal_term = not any(
            term in query.lower()
            for term in ["luật", "điều", "khoản", "nghị định", "thông tư",
                        "hợp đồng", "quyền", "nghĩa vụ", "xử phạt"]
        )

        if has_reference or (short_query and no_legal_term):
            # Get context from recent turns
            recent = history[-self.lookback:]
            context_topic = self._extract_topic(recent)

            if context_topic:
                # Prepend context
                resolved = f"[Về {context_topic}] {query}"
                return resolved

        return query

    def summarize_session(self, session_id: str) -> str:
        """
        Tóm tắt phiên hội thoại.
        
        Args:
            session_id: ID phiên
            
        Returns:
            Bản tóm tắt có cấu trúc
        """
        history = self.get_history(session_id)

        if not history:
            return "Chưa có nội dung hội thoại."

        summary_parts = ["📋 **Tóm tắt phiên tư vấn:**\n"]

        # Group by topic (simple: each turn is a point)
        referenced_docs = set()

        for i, turn in enumerate(history, 1):
            # Truncate long responses
            response_summary = turn.bot_response[:200]
            if len(turn.bot_response) > 200:
                response_summary += "..."

            summary_parts.append(f"{i}. **{turn.user_message}**")
            summary_parts.append(f"   → {response_summary}\n")

            # Collect referenced documents
            for source in turn.sources:
                doc_info = source.metadata.get("document_number", "")
                if doc_info:
                    referenced_docs.add(doc_info)

        # Add referenced documents
        if referenced_docs:
            summary_parts.append("\n📎 **Văn bản tham chiếu:**")
            for doc in sorted(referenced_docs):
                summary_parts.append(f"   - {doc}")

        return "\n".join(summary_parts)

    def record_feedback(
        self, session_id: str, turn_id: str, rating: int, comment: str = ""
    ) -> None:
        """Record user feedback for a turn."""
        history = self.get_history(session_id)
        for turn in history:
            if turn.turn_id == turn_id:
                turn.feedback = rating
                turn.feedback_comment = comment
                break

    def clear_history(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        if session_id in self._sessions:
            self._sessions[session_id] = []

    def _extract_topic(self, turns: List[ConversationTurn]) -> Optional[str]:
        """Extract the main topic from recent turns."""
        if not turns:
            return None

        # Get the most recent user message with legal content
        for turn in reversed(turns):
            msg = turn.user_message.lower()
            # Look for legal topic keywords
            topics = {
                "hợp đồng lao động": ["hợp đồng lao động", "người lao động", "sa thải"],
                "hôn nhân gia đình": ["ly hôn", "kết hôn", "tài sản chung", "vợ chồng"],
                "doanh nghiệp": ["công ty", "doanh nghiệp", "đăng ký kinh doanh"],
                "thừa kế": ["thừa kế", "di chúc", "tài sản"],
                "đất đai": ["đất đai", "sổ đỏ", "quyền sử dụng đất"],
                "hình sự": ["tội", "hình phạt", "truy tố"],
            }

            for topic, keywords in topics.items():
                if any(kw in msg for kw in keywords):
                    return topic

        # Fallback: return key phrase from last message
        last_msg = turns[-1].user_message
        words = last_msg.split()
        if len(words) > 3:
            return " ".join(words[:5])

        return None
