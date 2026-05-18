"""Gradio App - Giao diện demo cho hệ thống hỏi đáp pháp luật."""

import gradio as gr
from typing import List, Tuple, Optional

from src.services.chat_service import ChatService
from src.models.data_models import ChatResponse


def create_app(chat_service: ChatService) -> gr.Blocks:
    """
    Tạo Gradio app với các tabs:
    - Chat: Hỏi đáp pháp luật
    - Comparison: So sánh RAG vs Non-RAG
    - Settings: Chọn model, embedding
    """

    # State
    current_session = {"id": chat_service.new_session()}

    def respond(
        message: str,
        history: List[Tuple[str, str]],
        model_choice: str,
        embedding_choice: str,
    ) -> Tuple[List[Tuple[str, str]], str, str, str]:
        """Handle user message and return response."""
        if not message.strip():
            return history, "", "", ""

        # Map display names to keys
        model_map = {
            "Vistral-7B": "vistral",
            "PhoGPT-7.5B": "phogpt",
            "Qwen2.5-7B": "qwen",
        }
        embedding_map = {
            "PhoBERT": "phobert",
            "Multilingual-E5": "multilingual_e5",
        }

        model_key = model_map.get(model_choice, "qwen")
        emb_key = embedding_map.get(embedding_choice, "multilingual_e5")

        # Get response
        response = chat_service.chat(
            user_message=message,
            session_id=current_session["id"],
            model_key=model_key,
            embedding_model=emb_key,
        )

        # Format response
        answer = response.answer

        # Add validity warnings
        if response.validity_warnings:
            warnings = "\n".join(response.validity_warnings)
            answer = f"{warnings}\n\n{answer}"

        # Format sources
        sources_text = ""
        if response.sources:
            sources_text = "📎 **Nguồn tham chiếu:**\n"
            for i, src in enumerate(response.sources[:5], 1):
                breadcrumb = src.breadcrumb or src.metadata.get("breadcrumb", "")
                doc_num = src.metadata.get("document_number", "")
                sources_text += f"{i}. {breadcrumb} ({doc_num})\n"

        # Format suggestions
        suggestions_text = ""
        if response.suggestions:
            suggestions_text = "💡 **Câu hỏi liên quan:**\n"
            for s in response.suggestions:
                suggestions_text += f"• {s}\n"

        # Add to history
        full_response = answer
        if sources_text:
            full_response += f"\n\n{sources_text}"
        if suggestions_text:
            full_response += f"\n{suggestions_text}"

        # Add inference time
        full_response += f"\n\n⏱️ _Thời gian: {response.inference_time:.2f}s | Model: {model_choice}_"

        history.append((message, full_response))

        return history, "", sources_text, suggestions_text

    def respond_comparison(
        message: str,
        model_choice: str,
    ) -> Tuple[str, str, str]:
        """Compare RAG vs Non-RAG responses."""
        if not message.strip():
            return "", "", ""

        model_map = {
            "Vistral-7B": "vistral",
            "PhoGPT-7.5B": "phogpt",
            "Qwen2.5-7B": "qwen",
        }
        model_key = model_map.get(model_choice, "qwen")

        # RAG response
        rag_response = chat_service.chat(
            user_message=message,
            session_id=current_session["id"],
            model_key=model_key,
        )

        # Non-RAG response
        no_rag_response = chat_service.chat_without_rag(
            user_message=message,
            session_id=current_session["id"],
            model_key=model_key,
        )

        rag_text = f"🔵 **Có RAG** (⏱️ {rag_response.inference_time:.2f}s)\n\n{rag_response.answer}"
        no_rag_text = f"🔴 **Không RAG** (⏱️ {no_rag_response.inference_time:.2f}s)\n\n{no_rag_response.answer}"

        # Sources
        sources = ""
        if rag_response.sources:
            sources = "📎 Retrieved documents:\n"
            for i, src in enumerate(rag_response.sources[:3], 1):
                sources += f"{i}. [{src.relevance_score:.3f}] {src.breadcrumb}\n"

        return rag_text, no_rag_text, sources

    def new_session():
        """Start a new conversation session."""
        current_session["id"] = chat_service.new_session()
        return [], "", ""

    def get_summary():
        """Get session summary."""
        return chat_service.get_session_summary(current_session["id"])

    def submit_feedback(rating: str):
        """Submit feedback."""
        history = chat_service.conversation_manager.get_history(current_session["id"])
        if history:
            last_turn = history[-1]
            r = 1 if rating == "👍 Hữu ích" else 0
            chat_service.submit_feedback(current_session["id"], last_turn.turn_id, r)
            return f"Cảm ơn phản hồi của bạn! ({'👍' if r else '👎'})"
        return ""

    # Build UI
    with gr.Blocks(
        title="Hệ thống Hỏi đáp Pháp luật Việt Nam",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            "# ⚖️ Hệ thống Hỏi đáp Pháp luật Việt Nam\n"
            "Chatbot pháp luật thông minh sử dụng RAG + Fine-tuned LLM"
        )

        with gr.Tabs():
            # === TAB 1: CHAT ===
            with gr.Tab("💬 Hỏi đáp", id="chat"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            height=500,
                            label="Hội thoại",
                            data_testid="chat-history",
                        )
                        with gr.Row():
                            msg_input = gr.Textbox(
                                placeholder="Nhập câu hỏi pháp luật...",
                                label="Câu hỏi",
                                scale=4,
                                data_testid="chat-input",
                            )
                            send_btn = gr.Button(
                                "Gửi", variant="primary", scale=1,
                                data_testid="chat-send-button",
                            )

                        with gr.Row():
                            feedback_btn_good = gr.Button("👍 Hữu ích", data_testid="feedback-good")
                            feedback_btn_bad = gr.Button("👎 Không hữu ích", data_testid="feedback-bad")
                            new_session_btn = gr.Button("🔄 Phiên mới", data_testid="new-session-button")
                            summary_btn = gr.Button("📋 Tóm tắt", data_testid="summary-button")

                        feedback_output = gr.Textbox(label="Phản hồi", interactive=False)

                    with gr.Column(scale=1):
                        model_dropdown = gr.Dropdown(
                            choices=["Qwen2.5-7B", "Vistral-7B", "PhoGPT-7.5B"],
                            value="Qwen2.5-7B",
                            label="Mô hình LLM",
                            data_testid="model-selector",
                        )
                        embedding_dropdown = gr.Dropdown(
                            choices=["Multilingual-E5", "PhoBERT"],
                            value="Multilingual-E5",
                            label="Embedding Model",
                            data_testid="embedding-selector",
                        )
                        sources_output = gr.Markdown(label="Nguồn tham chiếu")
                        suggestions_output = gr.Markdown(label="Gợi ý")

                # Events
                send_btn.click(
                    respond,
                    [msg_input, chatbot, model_dropdown, embedding_dropdown],
                    [chatbot, msg_input, sources_output, suggestions_output],
                )
                msg_input.submit(
                    respond,
                    [msg_input, chatbot, model_dropdown, embedding_dropdown],
                    [chatbot, msg_input, sources_output, suggestions_output],
                )
                new_session_btn.click(new_session, [], [chatbot, sources_output, suggestions_output])
                summary_btn.click(get_summary, [], [feedback_output])
                feedback_btn_good.click(submit_feedback, [gr.State("👍 Hữu ích")], [feedback_output])
                feedback_btn_bad.click(submit_feedback, [gr.State("👎 Không hữu ích")], [feedback_output])

            # === TAB 2: COMPARISON ===
            with gr.Tab("⚖️ So sánh RAG vs Non-RAG", id="comparison"):
                gr.Markdown("### So sánh câu trả lời có RAG và không có RAG")

                with gr.Row():
                    comp_input = gr.Textbox(
                        placeholder="Nhập câu hỏi để so sánh...",
                        label="Câu hỏi",
                        scale=4,
                        data_testid="comparison-input",
                    )
                    comp_model = gr.Dropdown(
                        choices=["Qwen2.5-7B", "Vistral-7B", "PhoGPT-7.5B"],
                        value="Qwen2.5-7B",
                        label="Model",
                        scale=1,
                    )
                    comp_btn = gr.Button("So sánh", variant="primary", data_testid="comparison-button")

                with gr.Row():
                    rag_output = gr.Markdown(label="Có RAG")
                    no_rag_output = gr.Markdown(label="Không RAG")

                comp_sources = gr.Markdown(label="Retrieved Documents")

                comp_btn.click(
                    respond_comparison,
                    [comp_input, comp_model],
                    [rag_output, no_rag_output, comp_sources],
                )

            # === TAB 3: INFO ===
            with gr.Tab("ℹ️ Thông tin", id="info"):
                gr.Markdown("""
                ## Về hệ thống
                
                Hệ thống hỏi đáp pháp luật Việt Nam sử dụng:
                - **RAG (Retrieval-Augmented Generation)**: Truy xuất văn bản pháp luật liên quan
                - **Fine-tuned LLM**: Mô hình ngôn ngữ đã được tinh chỉnh trên dữ liệu pháp luật VN
                - **Semantic Chunking**: Chia văn bản theo cấu trúc Điều/Khoản/Điểm
                - **Reranking**: Sắp xếp lại kết quả tìm kiếm bằng bge-reranker-v2-m3
                
                ## Mô hình hỗ trợ
                - **Vistral-7B**: Mô hình tiếng Việt dựa trên Mistral
                - **PhoGPT-7.5B**: Mô hình tiếng Việt của VinAI
                - **Qwen2.5-7B**: Mô hình đa ngôn ngữ
                
                ## Dữ liệu
                - Corpus: 500-2000 văn bản QPPL từ vbpl.vn
                - Q&A: ALQAC 2023 + tự xây dựng (5000-10000+ cặp)
                
                ## Lưu ý
                ⚠️ Hệ thống chỉ mang tính chất tham khảo. Vui lòng tham khảo luật sư 
                chuyên nghiệp cho các vấn đề pháp lý quan trọng.
                """)

    return app


def launch_app(chat_service: ChatService, share: bool = False):
    """Launch the Gradio app."""
    app = create_app(chat_service)
    app.launch(share=share)
