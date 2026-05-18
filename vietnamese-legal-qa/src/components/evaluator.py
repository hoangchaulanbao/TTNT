"""Evaluator - Đánh giá hệ thống bằng RAGAS framework."""

import json
from typing import List, Dict, Optional
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, context_recall, answer_relevancy

from src.models.data_models import RAGASScores, EvaluationReport, QAPair
from src.config.settings import Settings


class Evaluator:
    """Đánh giá chất lượng hệ thống bằng RAGAS."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.reports_dir = Path(settings.config.get("evaluation", {}).get(
            "reports_dir", "evaluation/reports"
        ))
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_response(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str,
    ) -> RAGASScores:
        """
        Đánh giá 1 response bằng RAGAS.
        
        Args:
            question: Câu hỏi
            answer: Câu trả lời từ hệ thống
            contexts: Retrieved contexts
            ground_truth: Câu trả lời đúng (reference)
            
        Returns:
            RAGASScores
        """
        eval_dataset = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth],
        })

        result = evaluate(
            eval_dataset,
            metrics=[faithfulness, context_recall, answer_relevancy],
        )

        return RAGASScores(
            faithfulness=result["faithfulness"],
            context_recall=result["context_recall"],
            answer_relevancy=result["answer_relevancy"],
        )

    def evaluate_batch(
        self,
        questions: List[str],
        answers: List[str],
        contexts_list: List[List[str]],
        ground_truths: List[str],
        model_name: str = "unknown",
    ) -> EvaluationReport:
        """
        Đánh giá batch responses.
        
        Args:
            questions: Danh sách câu hỏi
            answers: Danh sách câu trả lời
            contexts_list: Danh sách contexts cho mỗi câu
            ground_truths: Danh sách ground truth
            model_name: Tên model đang đánh giá
            
        Returns:
            EvaluationReport
        """
        eval_dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
            "ground_truth": ground_truths,
        })

        print(f"[Evaluator] Evaluating {len(questions)} samples for model: {model_name}...")

        result = evaluate(
            eval_dataset,
            metrics=[faithfulness, context_recall, answer_relevancy],
        )

        report = EvaluationReport(
            model_name=model_name,
            avg_faithfulness=result["faithfulness"],
            avg_context_recall=result["context_recall"],
            avg_answer_relevancy=result["answer_relevancy"],
            num_samples=len(questions),
        )

        print(f"[Evaluator] Results for {model_name}:")
        print(f"  Faithfulness: {report.avg_faithfulness:.4f}")
        print(f"  Context Recall: {report.avg_context_recall:.4f}")
        print(f"  Answer Relevancy: {report.avg_answer_relevancy:.4f}")

        return report

    def compare_models(
        self,
        test_data: List[Dict],
        model_reports: Dict[str, EvaluationReport],
    ) -> str:
        """
        So sánh hiệu suất giữa các models.
        
        Args:
            test_data: Test dataset
            model_reports: Dict of model_name -> EvaluationReport
            
        Returns:
            Formatted comparison table
        """
        lines = ["# Model Comparison Report\n"]
        lines.append("| Model | Faithfulness | Context Recall | Answer Relevancy | Samples |")
        lines.append("|---|---|---|---|---|")

        for name, report in model_reports.items():
            lines.append(
                f"| {name} | {report.avg_faithfulness:.4f} | "
                f"{report.avg_context_recall:.4f} | "
                f"{report.avg_answer_relevancy:.4f} | "
                f"{report.num_samples} |"
            )

        return "\n".join(lines)

    def compare_rag_vs_no_rag(
        self,
        rag_report: EvaluationReport,
        no_rag_report: EvaluationReport,
    ) -> str:
        """
        So sánh RAG vs Non-RAG.
        
        Returns:
            Formatted comparison
        """
        lines = ["# RAG vs Non-RAG Comparison\n"]
        lines.append("| Metric | With RAG | Without RAG | Improvement |")
        lines.append("|---|---|---|---|")

        metrics = [
            ("Faithfulness", rag_report.avg_faithfulness, no_rag_report.avg_faithfulness),
            ("Context Recall", rag_report.avg_context_recall, no_rag_report.avg_context_recall),
            ("Answer Relevancy", rag_report.avg_answer_relevancy, no_rag_report.avg_answer_relevancy),
        ]

        for name, rag_val, no_rag_val in metrics:
            improvement = rag_val - no_rag_val
            sign = "+" if improvement > 0 else ""
            lines.append(
                f"| {name} | {rag_val:.4f} | {no_rag_val:.4f} | {sign}{improvement:.4f} |"
            )

        return "\n".join(lines)

    def export_report(self, report: EvaluationReport, filename: str) -> str:
        """Export evaluation report to file."""
        output_path = self.reports_dir / filename

        report_data = {
            "model_name": report.model_name,
            "avg_faithfulness": report.avg_faithfulness,
            "avg_context_recall": report.avg_context_recall,
            "avg_answer_relevancy": report.avg_answer_relevancy,
            "num_samples": report.num_samples,
            "detailed_scores": report.detailed_scores,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        print(f"[Evaluator] Report exported to: {output_path}")
        return str(output_path)
