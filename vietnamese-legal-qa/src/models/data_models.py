"""Shared data models for the Vietnamese Legal QA system."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict


@dataclass
class DocumentMetadata:
    """Metadata for a legal document."""
    document_number: str = ""
    document_type: str = ""  # Luật, Nghị định, Thông tư, Quyết định
    title: str = ""
    issued_date: Optional[datetime] = None
    effective_date: Optional[datetime] = None
    issuing_body: str = ""
    category: str = ""
    status: str = "Không rõ"  # Còn hiệu lực, Hết hiệu lực, Không rõ


@dataclass
class LegalDocument:
    """A legal document from the corpus."""
    id: str = ""
    title: str = ""
    content: str = ""
    raw_html: Optional[str] = None
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    source_url: str = ""
    crawled_at: Optional[datetime] = None


@dataclass
class LegalStructureNode:
    """A node in the legal document structure tree."""
    level: str = ""  # PHAN, CHUONG, MUC, DIEU, KHOAN, DIEM
    number: str = ""
    title: Optional[str] = None
    content: str = ""
    children: List["LegalStructureNode"] = field(default_factory=list)
    start_pos: int = 0
    end_pos: int = 0


@dataclass
class Chunk:
    """A text chunk from a legal document."""
    id: str = ""
    content: str = ""
    token_count: int = 0
    breadcrumb: str = ""
    document_id: str = ""
    article_number: Optional[str] = None
    clause_number: Optional[str] = None
    point_number: Optional[str] = None
    document_number: str = ""
    document_title: str = ""
    issued_date: Optional[datetime] = None
    issuing_body: str = ""
    category: str = ""


@dataclass
class QAPair:
    """A question-answer pair for training/evaluation."""
    id: str = ""
    question: str = ""
    answer: str = ""
    context: Optional[str] = None
    source_document: Optional[str] = None
    category: str = ""
    source: str = ""  # alqac, generated, manual


@dataclass
class InstructionSample:
    """An instruction-following sample for fine-tuning."""
    system_prompt: str = ""
    user_message: str = ""
    assistant_message: str = ""


@dataclass
class SearchResult:
    """A result from vector similarity search."""
    chunk_id: str = ""
    content: str = ""
    similarity_score: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class RankedDocument:
    """A document after reranking."""
    chunk_id: str = ""
    content: str = ""
    relevance_score: float = 0.0
    original_score: float = 0.0
    metadata: Dict = field(default_factory=dict)
    breadcrumb: str = ""
    validity_warning: Optional[str] = None


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    turn_id: str = ""
    user_message: str = ""
    bot_response: str = ""
    sources: List[RankedDocument] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    timestamp: Optional[datetime] = None
    feedback: Optional[int] = None
    feedback_comment: Optional[str] = None
    topic: Optional[str] = None


@dataclass
class RAGASScores:
    """RAGAS evaluation scores."""
    faithfulness: float = 0.0
    context_recall: float = 0.0
    answer_relevancy: float = 0.0


@dataclass
class ChatResponse:
    """Response from the chat service."""
    answer: str = ""
    sources: List[RankedDocument] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    validity_warnings: List[str] = field(default_factory=list)
    is_rejection: bool = False
    is_clarification: bool = False
    clarification_options: List[str] = field(default_factory=list)
    inference_time: float = 0.0
    model_name: str = ""
    ragas_scores: Optional[RAGASScores] = None


@dataclass
class AmbiguityResult:
    """Result of ambiguity detection."""
    is_ambiguous: bool = False
    reason: Optional[str] = None
    clarification_options: List[str] = field(default_factory=list)


@dataclass
class ValidityWarning:
    """Warning about document validity."""
    document_number: str = ""
    issued_date: Optional[datetime] = None
    age_years: float = 0.0
    warning_text: str = ""
    severity: str = "info"  # info, warning, critical


@dataclass
class TrainingResult:
    """Result from fine-tuning training."""
    model_name: str = ""
    final_loss: float = 0.0
    training_time_minutes: float = 0.0
    epochs_completed: int = 0
    adapter_path: str = ""


@dataclass
class EvaluationReport:
    """Evaluation report with RAGAS scores."""
    model_name: str = ""
    avg_faithfulness: float = 0.0
    avg_context_recall: float = 0.0
    avg_answer_relevancy: float = 0.0
    num_samples: int = 0
    detailed_scores: List[Dict] = field(default_factory=list)
