"""Data Ingestion Service - Orchestrate data collection and indexing."""

from typing import List, Optional

from src.components.data_collector import DataCollector
from src.components.text_processor import TextProcessor
from src.components.embedding_engine import EmbeddingEngine
from src.components.vector_store import VectorStoreManager
from src.models.data_models import LegalDocument, Chunk
from src.config.settings import Settings


class DataIngestionService:
    """Orchestrate: collect → chunk → embed → index."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.collector = DataCollector(settings)
        self.processor = TextProcessor(settings)
        self.embedding_engine = EmbeddingEngine(settings)
        self.vector_store = VectorStoreManager(settings)

    def ingest_from_files(
        self,
        directory: str,
        embedding_model: str = "multilingual_e5",
        collection_name: str = "legal_docs",
    ) -> dict:
        """
        Full pipeline: load files → chunk → embed → index.
        
        Args:
            directory: Thư mục chứa văn bản pháp luật (JSON)
            embedding_model: Model embedding sử dụng
            collection_name: Tên ChromaDB collection
            
        Returns:
            Ingestion statistics
        """
        # Step 1: Load documents
        documents = self.collector.load_from_files(directory)
        print(f"[Ingestion] Loaded {len(documents)} documents")

        # Step 2: Chunk documents
        all_chunks = []
        for doc in documents:
            chunks = self.processor.semantic_chunk(doc)
            all_chunks.extend(chunks)
        print(f"[Ingestion] Created {len(all_chunks)} chunks")

        # Step 3: Embed chunks
        chunk_texts = [chunk.content for chunk in all_chunks]
        embeddings = self.embedding_engine.embed_chunks(chunk_texts, embedding_model)
        print(f"[Ingestion] Embedded {len(embeddings)} chunks")

        # Step 4: Index in ChromaDB
        self.vector_store.index_chunks(all_chunks, embeddings, collection_name)
        print(f"[Ingestion] Indexed to collection '{collection_name}'")

        return {
            "documents_loaded": len(documents),
            "chunks_created": len(all_chunks),
            "embeddings_generated": len(embeddings),
            "collection_name": collection_name,
            "embedding_model": embedding_model,
        }

    def ingest_alqac(self, dataset_path: str) -> dict:
        """Load ALQAC dataset for fine-tuning."""
        qa_pairs = self.collector.load_alqac_dataset(dataset_path)
        return {
            "qa_pairs_loaded": len(qa_pairs),
            "source": "ALQAC 2023",
        }

    def rebuild_index(
        self,
        directory: str,
        embedding_model: str = "multilingual_e5",
        collection_name: str = "legal_docs",
    ) -> dict:
        """Delete existing collection and rebuild from scratch."""
        self.vector_store.delete_collection(collection_name)
        return self.ingest_from_files(directory, embedding_model, collection_name)

    def get_corpus_stats(self) -> dict:
        """Get statistics about the indexed corpus."""
        collections = self.vector_store.list_collections()
        stats = {}
        for name in collections:
            stats[name] = self.vector_store.get_collection_stats(name)
        return stats
