"""Vector Store - ChromaDB operations."""

from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.models.data_models import Chunk, SearchResult
from src.config.settings import Settings


class VectorStoreManager:
    """Quản lý ChromaDB cho lưu trữ và truy vấn vectors."""

    def __init__(self, settings: Settings):
        self.settings = settings
        persist_dir = settings.data.get("chroma_db_dir", "data/chroma_db")
        self.client = chromadb.PersistentClient(path=persist_dir)

    def get_or_create_collection(self, name: str, embedding_dim: int = 768):
        """Get or create a ChromaDB collection."""
        return self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def index_chunks(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        collection_name: str = "legal_docs",
    ) -> None:
        """
        Lưu chunks + vectors vào ChromaDB.
        
        Args:
            chunks: Danh sách chunks
            embeddings: Vectors tương ứng
            collection_name: Tên collection
        """
        collection = self.get_or_create_collection(collection_name)

        # Prepare data for ChromaDB
        ids = [chunk.id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "document_id": chunk.document_id,
                "document_number": chunk.document_number,
                "document_title": chunk.document_title,
                "article_number": chunk.article_number or "",
                "clause_number": chunk.clause_number or "",
                "breadcrumb": chunk.breadcrumb,
                "category": chunk.category,
                "issuing_body": chunk.issuing_body,
                "issued_date": chunk.issued_date.isoformat() if chunk.issued_date else "",
            }
            for chunk in chunks
        ]

        # Batch insert (ChromaDB has batch size limits)
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            collection.add(
                ids=ids[i:end],
                embeddings=[emb.tolist() if hasattr(emb, 'tolist') else emb for emb in embeddings[i:end]],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
            )

        print(f"[VectorStore] Indexed {len(chunks)} chunks to collection '{collection_name}'")

    def similarity_search(
        self,
        query_embedding: List[float],
        collection_name: str = "legal_docs",
        top_k: int = 20,
        filters: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """
        Tìm kiếm top-k chunks tương tự nhất.
        
        Args:
            query_embedding: Vector query
            collection_name: Tên collection
            top_k: Số kết quả trả về
            filters: Bộ lọc metadata (optional)
            
        Returns:
            Danh sách SearchResult với similarity scores
        """
        collection = self.get_or_create_collection(collection_name)

        query_params = {
            "query_embeddings": [query_embedding if isinstance(query_embedding, list) 
                                  else query_embedding.tolist()],
            "n_results": top_k,
        }

        if filters:
            query_params["where"] = filters

        results = collection.query(**query_params)

        search_results = []
        if results and results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                score = 1 - results["distances"][0][i]  # Convert distance to similarity
                search_results.append(SearchResult(
                    chunk_id=chunk_id,
                    content=results["documents"][0][i],
                    similarity_score=score,
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                ))

        return search_results

    def get_collection_stats(self, collection_name: str = "legal_docs") -> Dict:
        """Get statistics about a collection."""
        try:
            collection = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "count": collection.count(),
            }
        except Exception:
            return {"name": collection_name, "count": 0}

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection."""
        try:
            self.client.delete_collection(collection_name)
            print(f"[VectorStore] Deleted collection '{collection_name}'")
        except Exception as e:
            print(f"[VectorStore] Error deleting collection: {e}")

    def list_collections(self) -> List[str]:
        """List all collection names."""
        collections = self.client.list_collections()
        return [c.name for c in collections]
