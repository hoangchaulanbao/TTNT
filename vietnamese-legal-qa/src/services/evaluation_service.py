"""Evaluation Service - Orchestrate đánh giá và so sánh."""

from typing import List, Dict
from pathlib import Path

from src.components.evaluator import Evaluator
from src.services.chat_service import ChatService
from src.models.data_models import QAPair, EvaluationReport
from src.config.settings import Settings


class EvaluationService:
    """Orchestrate đánh giá RAGAS và so sánh models."""

    def __init__(self, settings: Settings, chat_service: ChatService, evaluator: Evaluator):
        self.settings = settings
        self.chat_service = chat_service
        self.evaluator = evaluator

    def run_model_comparison(
        self,
        test_pairs: List[QAPair],
        model_keys: List[str],
        session_id: str,
    ) -> Dict[str, EvaluationReport]:
        """
        So sánh hiệu suất giữa các models.
        
        Args:
            test_pairs: Test Q&A pairs
            model_keys: List of model keys to compare
            session_id: Session for chat
            
        Returns:
            Dict of model_key -> EvaluationReport
        """
        reports = {}

        for model_key in model_keys:
            print(f"\n[EvaluationService] Evaluating model: {model_key}")

            questions = []
            answers = []
            contexts_list = []
            ground_truths = []

            for pair in test_pairs:
                response = self.chat_service.chat(
                    user_message=pair.question,
                    session_id=session_id,
                    model_key=model_key,
                )

                if not response.is_rejection and not response.is_clarification:
                    questions.append(pair.question)
                    answers.append(response.answer)
                    contexts_list.append([doc.content for doc in response.sources])
                    ground_truths.append(pair.answer)

            if questions:
                report = self.evaluator.evaluate_batch(
                    questions=questions,
                    answers=answers,
                    contexts_list=contexts_list,
                    ground_truths=ground_truths,
                    model_name=model_key,
                )
                reports[model_key] = report

        return reports

    def run_rag_comparison(
        self,
        test_pairs: List[QAPair],
        model_key: str,
        session_id: str,
    ) -> Dict[str, EvaluationReport]:
        """
        So sánh RAG vs Non-RAG cho một model.
        
        Returns:
            Dict with 'rag' and 'no_rag' reports
        """
        print(f"\n[EvaluationService] RAG vs Non-RAG comparison for: {model_key}")

        # With RAG
        rag_questions, rag_answers, rag_contexts, rag_truths = [], [], [], []
        # Without RAG
        no_rag_answers = []

        for pair in test_pairs:
            # RAG response
            rag_response = self.chat_service.chat(
                user_message=pair.question,
                session_id=session_id,
                model_key=model_key,
            )

            # Non-RAG response
            no_rag_response = self.chat_service.chat_without_rag(
                user_message=pair.question,
                session_id=session_id,
                model_key=model_key,
            )

            if not rag_response.is_rejection:
                rag_questions.append(pair.question)
                rag_answers.append(rag_response.answer)
                rag_contexts.append([doc.content for doc in rag_response.sources])
                rag_truths.append(pair.answer)
                no_rag_answers.append(no_rag_response.answer)

        results = {}

        if rag_questions:
            # Evaluate RAG
            results["rag"] = self.evaluator.evaluate_batch(
                questions=rag_questions,
                answers=rag_answers,
                contexts_list=rag_contexts,
                ground_truths=rag_truths,
                model_name=f"{model_key}_with_rag",
            )

            # Evaluate Non-RAG (empty contexts)
            results["no_rag"] = self.evaluator.evaluate_batch(
                questions=rag_questions,
                answers=no_rag_answers,
                contexts_list=[[] for _ in rag_questions],
                ground_truths=rag_truths,
                model_name=f"{model_key}_without_rag",
            )

        return results

    def run_embedding_comparison(
        self,
        test_pairs: List[QAPair],
        model_key: str,
        session_id: str,
    ) -> Dict[str, EvaluationReport]:
        """So sánh giữa 2 embedding models."""
        embedding_models = ["phobert", "multilingual_e5"]
        reports = {}

        for emb_model in embedding_models:
            print(f"\n[EvaluationService] Evaluating embedding: {emb_model}")

            questions, answers, contexts_list, ground_truths = [], [], [], []

            for pair in test_pairs:
                response = self.chat_service.chat(
                    user_message=pair.question,
                    session_id=session_id,
                    model_key=model_key,
                    embedding_model=emb_model,
                )

                if not response.is_rejection:
                    questions.append(pair.question)
                    answers.append(response.answer)
                    contexts_list.append([doc.content for doc in response.sources])
                    ground_truths.append(pair.answer)

            if questions:
                reports[emb_model] = self.evaluator.evaluate_batch(
                    questions=questions,
                    answers=answers,
                    contexts_list=contexts_list,
                    ground_truths=ground_truths,
                    model_name=f"embedding_{emb_model}",
                )

        return reports

    def export_all_reports(self, reports: Dict[str, EvaluationReport]) -> None:
        """Export all reports to files."""
        for name, report in reports.items():
            self.evaluator.export_report(report, f"{name}_report.json")
