"""Text Processor - Semantic chunking theo cấu trúc văn bản pháp luật."""

import re
import uuid
from typing import List, Optional, Tuple

from src.models.data_models import LegalDocument, Chunk, LegalStructureNode
from src.config.settings import Settings


class TextProcessor:
    """Xử lý và chunking văn bản pháp luật theo cấu trúc Điều/Khoản/Điểm."""

    def __init__(self, settings: Settings):
        self.settings = settings
        chunking_config = settings.rag.get("chunking", {})
        self.min_tokens = chunking_config.get("min_tokens", 100)
        self.max_tokens = chunking_config.get("max_tokens", 1000)
        self.target_tokens = chunking_config.get("target_tokens", 500)
        self.overlap_tokens = chunking_config.get("overlap_tokens", 50)

    def semantic_chunk(self, document: LegalDocument) -> List[Chunk]:
        """
        Chia văn bản theo cấu trúc pháp luật (Điều/Khoản/Điểm).
        
        Strategy:
        - Primary boundary: Mỗi Điều = 1 chunk
        - Điều dài (> max_tokens): split theo Khoản
        - Điều ngắn (< min_tokens): merge với Điều liền kề
        
        Args:
            document: Văn bản pháp luật đã xử lý
            
        Returns:
            Danh sách chunks với metadata
        """
        articles = self._split_by_articles(document.content)

        if not articles:
            # Fallback: fixed-size chunking nếu không detect được cấu trúc
            return self._fallback_chunking(document)

        chunks = []
        buffer = []  # Buffer cho merge Điều ngắn

        for article_num, article_content in articles:
            token_count = self._estimate_tokens(article_content)

            if token_count > self.max_tokens:
                # Flush buffer trước
                if buffer:
                    chunks.append(self._create_chunk_from_buffer(buffer, document))
                    buffer = []
                # Split Điều dài theo Khoản
                sub_chunks = self._split_article_by_clauses(
                    article_num, article_content, document
                )
                chunks.extend(sub_chunks)

            elif token_count < self.min_tokens:
                # Thêm vào buffer để merge
                buffer.append((article_num, article_content))
                # Flush nếu buffer đủ lớn
                if self._estimate_buffer_tokens(buffer) >= self.target_tokens:
                    chunks.append(self._create_chunk_from_buffer(buffer, document))
                    buffer = []
            else:
                # Flush buffer trước
                if buffer:
                    chunks.append(self._create_chunk_from_buffer(buffer, document))
                    buffer = []
                # Tạo chunk bình thường
                chunks.append(self._create_chunk(
                    content=article_content,
                    document=document,
                    article_number=article_num,
                ))

        # Flush remaining buffer
        if buffer:
            chunks.append(self._create_chunk_from_buffer(buffer, document))

        return chunks

    def parse_legal_structure(self, text: str) -> List[LegalStructureNode]:
        """
        Parse cấu trúc văn bản pháp luật VN.
        
        Hierarchy: PHẦN > CHƯƠNG > MỤC > ĐIỀU > KHOẢN > ĐIỂM
        """
        nodes = []
        # Split by Điều (primary unit)
        articles = self._split_by_articles(text)

        for article_num, content in articles:
            node = LegalStructureNode(
                level="DIEU",
                number=article_num,
                content=content,
                children=self._parse_clauses(content),
            )
            nodes.append(node)

        return nodes

    def _split_by_articles(self, text: str) -> List[Tuple[str, str]]:
        """Split text by Điều (Article) boundaries."""
        # Pattern: "Điều X." or "Điều X:" at start of line
        pattern = r"(?:^|\n)(Điều\s+(\d+)[\.:]?\s*(.*))"
        matches = list(re.finditer(pattern, text))

        if not matches:
            return []

        articles = []
        for i, match in enumerate(matches):
            article_num = match.group(2)
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()
            articles.append((article_num, content))

        return articles

    def _parse_clauses(self, article_text: str) -> List[LegalStructureNode]:
        """Parse Khoản (clauses) within an article."""
        clauses = []
        # Pattern: "1. ", "2. " at start of line
        pattern = r"(?:^|\n)(\d+)\.\s+"
        matches = list(re.finditer(pattern, article_text))

        for i, match in enumerate(matches):
            clause_num = match.group(1)
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(article_text)
            content = article_text[start:end].strip()
            clauses.append(LegalStructureNode(
                level="KHOAN",
                number=clause_num,
                content=content,
            ))

        return clauses

    def _split_article_by_clauses(
        self, article_num: str, content: str, document: LegalDocument
    ) -> List[Chunk]:
        """Split a long article by its clauses."""
        chunks = []
        clauses = self._parse_clauses(content)

        if not clauses:
            # No clauses found, do fixed-size split
            return self._fixed_size_split(content, document, article_num)

        for clause in clauses:
            chunk = self._create_chunk(
                content=clause.content,
                document=document,
                article_number=article_num,
                clause_number=clause.number,
            )
            chunks.append(chunk)

        return chunks

    def _create_chunk(
        self,
        content: str,
        document: LegalDocument,
        article_number: Optional[str] = None,
        clause_number: Optional[str] = None,
        point_number: Optional[str] = None,
    ) -> Chunk:
        """Create a Chunk with full metadata."""
        # Build breadcrumb
        breadcrumb_parts = [document.metadata.title or document.title]
        if article_number:
            breadcrumb_parts.append(f"Điều {article_number}")
        if clause_number:
            breadcrumb_parts.append(f"Khoản {clause_number}")
        if point_number:
            breadcrumb_parts.append(f"Điểm {point_number}")

        return Chunk(
            id=str(uuid.uuid4()),
            content=content,
            token_count=self._estimate_tokens(content),
            breadcrumb=" > ".join(breadcrumb_parts),
            document_id=document.id,
            article_number=article_number,
            clause_number=clause_number,
            point_number=point_number,
            document_number=document.metadata.document_number,
            document_title=document.title,
            issued_date=document.metadata.issued_date,
            issuing_body=document.metadata.issuing_body,
            category=document.metadata.category,
        )

    def _create_chunk_from_buffer(
        self, buffer: List[Tuple[str, str]], document: LegalDocument
    ) -> Chunk:
        """Create a chunk by merging buffered short articles."""
        combined_content = "\n\n".join(content for _, content in buffer)
        first_article = buffer[0][0]
        last_article = buffer[-1][0]

        return self._create_chunk(
            content=combined_content,
            document=document,
            article_number=f"{first_article}-{last_article}",
        )

    def _fallback_chunking(self, document: LegalDocument) -> List[Chunk]:
        """Fallback: fixed-size chunking when structure is not detected."""
        return self._fixed_size_split(document.content, document)

    def _fixed_size_split(
        self, text: str, document: LegalDocument, article_number: Optional[str] = None
    ) -> List[Chunk]:
        """Split text into fixed-size chunks with overlap."""
        chunks = []
        words = text.split()
        target_words = self.target_tokens  # Approximate: 1 token ≈ 1 word for Vietnamese

        i = 0
        while i < len(words):
            end = min(i + target_words, len(words))
            chunk_text = " ".join(words[i:end])

            chunks.append(self._create_chunk(
                content=chunk_text,
                document=document,
                article_number=article_number,
            ))

            # Move forward with overlap
            i += target_words - self.overlap_tokens

        return chunks

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (approximate: Vietnamese ~1.5 chars per token)."""
        return len(text.split())

    def _estimate_buffer_tokens(self, buffer: List[Tuple[str, str]]) -> int:
        """Estimate total tokens in buffer."""
        return sum(self._estimate_tokens(content) for _, content in buffer)
