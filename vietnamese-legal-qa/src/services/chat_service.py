"""Chat Service - Orchestrate pipeline hỏi đáp end-to-end."""

import time
from typing import List, Optional

from src.models.data_models import ChatResponse, ConversationTurn, RankedDocument
from src.components.retriever import Retriever
from src.components.conversation_manager import ConversationManager
from src.components.llm_engine import LLMEngine
from src.components.quality_guard import QualityGuard
from src.config.settings import Settings


class ChatService:
    """Core service: orchestrate toàn bộ pipeline hỏi đáp."""

    def __init__(
        self,
        settings: Settings,
        retriever: Retriever,
        conversation_manager: ConversationManager,
        llm_engine: LLMEngine,
        quality_guard: QualityGuard,
    ):
        self.settings = settings
        self.retriever = retriever
        self.conversation_manager = conversation_manager
        self.llm_engine = llm_engine
        self.quality_guard = quality_guard

    def chat(
        self,
        user_message: str,
        session_id: str,
        model_key: str = "qwen",
        embedding_model: str = "multilingual_e5",
    ) -> ChatResponse:
        """
        Pipeline hỏi đáp end-to-end.
        
        Flow:
        1. Quality check (out-of-scope, ambiguity)
        2. Context resolution (references, topic detection)
        3. Retrieval (search + rerank)
        4. Validity check
        5. Generation
        6. Post-processing (suggestions, save turn)
        
        Args:
            user_message: Tin nhắn người dùng
            session_id: ID phiên hội thoại
            model_key: Model LLM sử dụng
            embedding_model: Model embedding cho retrieval
            
        Returns:
            ChatResponse
        """
        start_time = time.time()

        # === STEP 1: QUALITY CHECK ===
        # Check out-of-scope
        if not self.quality_guard.is_legal_question(user_message):
            rejection_msg = self.quality_guard.generate_rejection_message(user_message)
            return ChatResponse(
                answer=rejection_msg,
                is_rejection=True,
                inference_time=time.time() - start_time,
                model_name=model_key,
            )

        # Check ambiguity
        ambiguity = self.quality_guard.is_ambiguous_question(user_message)
        if ambiguity.is_ambiguous:
            clarification_msg = (
                f"Câu hỏi của bạn khá rộng. Để tôi hỗ trợ tốt nhất, "
                f"bạn muốn tìm hiểu về lĩnh vực nào?\n\n"
            )
            for i, opt in enumerate(ambiguity.clarification_options, 1):
                clarification_msg += f"{i}. {opt}\n"
            clarification_msg += (
                f"\nBạn có thể chọn một trong các lựa chọn trên, "
                f"hoặc mô tả cụ thể hơn vấn đề bạn quan tâm."
            )
            return ChatResponse(
                answer=clarification_msg,
                is_clarification=True,
                clarification_options=ambiguity.clarification_options,
                inference_time=time.time() - start_time,
                model_name=model_key,
            )

        # === STEP 2: CONTEXT RESOLUTION ===
        history = self.conversation_manager.get_history(session_id)
        resolved_query = self.conversation_manager.resolve_references(user_message, history)

        # === STEP 3: RETRIEVAL ===
        ranked_docs = self.retriever.retrieve(
            query=resolved_query,
            conversation_history=history,
            embedding_model=embedding_model,
        )

        # === STEP 4: VALIDITY CHECK ===
        validity_warnings = []
        for doc in ranked_docs:
            warning = self.quality_guard.check_document_validity(doc.metadata)
            if warning:
                validity_warnings.append(warning.warning_text)
                doc.validity_warning = warning.warning_text

        # === STEP 5: GENERATION ===
        context_texts = [doc.content for doc in ranked_docs]
        answer = self.llm_engine.generate(
            query=resolved_query,
            context=context_texts,
            model_key=model_key,
            history=history,
        )

        # === STEP 6: POST-PROCESSING ===
        suggestions = self.llm_engine.suggest_related_questions(
            user_message, answer, context_texts
        )

        # Add complexity disclaimer if answer is long or topic is sensitive
        sensitive_topics = ["hình sự", "tội", "bắt giam", "khởi tố", "tử hình"]
        if any(topic in answer.lower() for topic in sensitive_topics):
            answer += (
                "\n\n⚠️ **Lưu ý**: Đây là vấn đề pháp lý phức tạp. "
                "Khuyến nghị bạn tham khảo ý kiến luật sư chuyên nghiệp "
                "để được tư vấn cụ thể cho trường hợp của mình."
            )

        # Save turn
        self.conversation_manager.add_turn(
            session_id=session_id,
            user_message=user_message,
            bot_response=answer,
            sources=ranked_docs,
            suggestions=suggestions,
        )

        inference_time = time.time() - start_time

        return ChatResponse(
            answer=answer,
            sources=ranked_docs,
            suggestions=suggestions,
            validity_warnings=validity_warnings,
            inference_time=inference_time,
            model_name=model_key,
        )

    def chat_without_rag(
        self, user_message: str, session_id: str, model_key: str = "qwen"
    ) -> ChatResponse:
        """
        Generate câu trả lời KHÔNG có RAG (cho so sánh).
        
        Args:
            user_message: Câu hỏi
            session_id: Session ID
            model_key: Model key
            
        Returns:
            ChatResponse (no sources)
        """
        start_time = time.time()

        answer = self.llm_engine.generate_without_rag(user_message, model_key)

        inference_time = time.time() - start_time

        return ChatResponse(
            answer=answer,
            sources=[],
            suggestions=[],
            inference_time=inference_time,
            model_name=model_key,
        )

    def get_session_summary(self, session_id: str) -> str:
        """Get summary of conversation session."""
        return self.conversation_manager.summarize_session(session_id)

    def submit_feedback(
        self, session_id: str, turn_id: str, rating: int, comment: str = ""
    ) -> None:
        """Submit feedback for a response."""
        self.conversation_manager.record_feedback(session_id, turn_id, rating, comment)

        # Also record in quality guard for stats
        history = self.conversation_manager.get_history(session_id)
        for turn in history:
            if turn.turn_id == turn_id:
                self.quality_guard.record_feedback(
                    turn.user_message, turn.bot_response, rating, comment
                )
                break

    def new_session(self) -> str:
        """Create a new conversation session."""
        return self.conversation_manager.new_session()
