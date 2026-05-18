"""Retriever - Orchestrate retrieval pipeline (search + rerank)."""

from typing import List, Optional

from FlagEmbedding import FlagReranker

from src.models.data_models import SearchResult, RankedDocument, ConversationTurn
from src.components.embedding_engine import EmbeddingEngine
from src.components.vector_store import VectorStoreManager
from src.config.settings import Settings


class Retriever:
    """Retrieval pipeline: query formulation → search → rerank."""

    def __init__(
        self,
        settings: Settings,
        embedding_engine: EmbeddingEngine,
        vector_store: VectorStoreManager,
    ):
        self.settings = settings
        self.embedding_engine = embedding_engine
        self.vector_store = vector_store
        self.rag_config = settings.rag.get("retrieval", {})
        self.conv_config = settings.conversation

        # Reranker
        reranker_config = settings.reranker
        self._reranker = None
        self._reranker_model_name = reranker_config.get("model_name", "BAAI/bge-reranker-v2-m3")
        self._rerank_top_k = reranker_config.get("top_k", 5)

    @property
    def reranker(self):
        """Lazy load reranker model."""
        if self._reranker is None:
            print(f"[Retriever] Loading reranker: {self._reranker_model_name}...")
            self._reranker = FlagReranker(self._reranker_model_name, use_fp16=True)
        return self._reranker

    def retrieve(
        self,
        query: str,
        conversation_history: List[ConversationTurn] = None,
        top_k: int = None,
        embedding_model: str = "multilingual_e5",
        collection_name: str = "legal_docs",
    ) -> List[RankedDocument]:
        """
        Pipeline retrieval hoàn chỉnh.
        
        Args:
            query: Câu hỏi người dùng (đã resolve references)
            conversation_history: Lịch sử hội thoại
            top_k: Số documents trả về sau rerank
            embedding_model: Model embedding sử dụng
            collection_name: ChromaDB collection
            
        Returns:
            Documents đã rerank
        """
        if top_k is None:
            top_k = self._rerank_top_k

        initial_top_k = self.rag_config.get("initial_top_k", 20)

        # Step 1: Formulate search query (context-aware)
        search_query = self.formulate_query(query, conversation_history)

        # Step 2: Embed query
        query_embedding = self.embedding_engine.embed_query(search_query, embedding_model)

        # Step 3: Initial retrieval
        search_results = self.vector_store.similarity_search(
            query_embedding=query_embedding.tolist(),
            collection_name=collection_name,
            top_k=initial_top_k,
        )

        # Filter by minimum similarity
        min_sim = self.rag_config.get("min_similarity", 0.3)
        search_results = [r for r in search_results if r.similarity_score >= min_sim]

        if not search_results:
            return []

        # Step 4: Rerank
        ranked_docs = self.rerank(query, search_results, top_k)

        return ranked_docs

    def rerank(
        self, query: str, documents: List[SearchResult], top_k: int = 5
    ) -> List[RankedDocument]:
        """
        Rerank documents bằng bge-reranker-v2-m3.
        
        Args:
            query: Query gốc
            documents: Documents từ initial retrieval
            top_k: Số documents giữ lại
            
        Returns:
            Documents đã sắp xếp theo relevance
        """
        if not documents:
            return []

        # Prepare pairs for reranker
        pairs = [[query, doc.content] for doc in documents]

        # Get reranker scores
        scores = self.reranker.compute_score(pairs, normalize=True)
        if isinstance(scores, float):
            scores = [scores]

        # Create ranked documents
        ranked = []
        for doc, score in zip(documents, scores):
            ranked.append(RankedDocument(
                chunk_id=doc.chunk_id,
                content=doc.content,
                relevance_score=float(score),
                original_score=doc.similarity_score,
                metadata=doc.metadata,
                breadcrumb=doc.metadata.get("breadcrumb", ""),
            ))

        # Sort by relevance score (descending)
        ranked.sort(key=lambda x: x.relevance_score, reverse=True)

        return ranked[:top_k]

    def formulate_query(
        self,
        user_query: str,
        history: Optional[List[ConversationTurn]] = None,
    ) -> str:
        """
        Bổ sung ngữ cảnh từ history vào query.
        
        Args:
            user_query: Query hiện tại (đã resolve references)
            history: Lịch sử hội thoại
            
        Returns:
            Query đã enriched
        """
        if not history:
            return user_query

        # Check if topic changed
        topic_changed = self.detect_topic_change(user_query, history)

        if topic_changed:
            # Fresh start - don't include old context
            return user_query

        # Include context from recent turns
        context_turns = self.conv_config.get("context_turns_for_prompt", 3)
        recent_turns = history[-context_turns:]

        # Extract key terms from recent questions
        context_terms = []
        for turn in recent_turns:
            # Get important words from previous questions
            words = turn.user_message.split()
            # Keep legal terms and nouns (simple heuristic)
            context_terms.extend([w for w in words if len(w) > 3])

        # Deduplicate and limit
        context_terms = list(dict.fromkeys(context_terms))[:10]

        if context_terms:
            enriched = f"{user_query} {' '.join(context_terms)}"
            return enriched

        return user_query

    def detect_topic_change(
        self, current_query: str, history: List[ConversationTurn]
    ) -> bool:
        """
        Phát hiện khi người dùng chuyển lĩnh vực.
        
        Args:
            current_query: Query mới
            history: Lịch sử hội thoại
            
        Returns:
            True nếu topic đã thay đổi
        """
        if not history:
            return False

        threshold = self.conv_config.get("topic_change_threshold", 0.2)

        # Get keywords from current query
        current_words = set(current_query.lower().split())

        # Get keywords from last 3 turns
        lookback = min(3, len(history))
        history_words = set()
        for turn in history[-lookback:]:
            history_words.update(turn.user_message.lower().split())

        # Remove common stop words
        stop_words = {"là", "có", "không", "của", "và", "hoặc", "trong", "theo",
                      "được", "phải", "thì", "nếu", "khi", "cho", "về", "với",
                      "này", "đó", "các", "một", "những", "tôi", "bạn"}
        current_words -= stop_words
        history_words -= stop_words

        if not current_words or not history_words:
            return False

        # Compute overlap
        overlap = len(current_words & history_words)
        max_possible = min(len(current_words), len(history_words))

        if max_possible == 0:
            return True

        overlap_ratio = overlap / max_possible

        # Explicit topic change markers
        change_markers = ["còn về", "chuyển sang", "ngoài ra", "vấn đề khác",
                         "câu hỏi khác", "chủ đề khác"]
        for marker in change_markers:
            if marker in current_query.lower():
                return True

        return overlap_ratio < threshold
