"""Data Collector - Thu thập corpus pháp luật từ vbpl.vn và ALQAC 2023."""

import json
import time
import uuid
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.models.data_models import LegalDocument, DocumentMetadata, QAPair
from src.config.settings import Settings


class DataCollector:
    """Thu thập và xử lý dữ liệu pháp luật."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.crawl_config = settings.crawling
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Research Bot - Vietnamese Legal QA)"
        })

    def crawl_legal_documents(
        self,
        base_url: str = "https://vbpl.vn",
        category: str = "all",
        max_docs: int = 100
    ) -> List[LegalDocument]:
        """
        Crawl văn bản pháp luật từ vbpl.vn.
        
        Args:
            base_url: URL gốc của nguồn dữ liệu
            category: Lĩnh vực pháp luật (dân sự, lao động, doanh nghiệp...)
            max_docs: Số lượng văn bản tối đa cần crawl
            
        Returns:
            Danh sách LegalDocument đã xử lý
        """
        documents = []
        rate_limit = self.crawl_config.get("rate_limit_seconds", 1.5)
        max_retries = self.crawl_config.get("max_retries", 3)
        timeout = self.crawl_config.get("timeout_seconds", 30)

        print(f"[DataCollector] Crawling {category} documents from {base_url}...")
        print(f"[DataCollector] Rate limit: {rate_limit}s, Max docs: {max_docs}")

        # Note: Actual crawling logic depends on vbpl.vn's structure
        # This is a template that should be adapted to the actual website
        # For the demo, we'll provide a file-based loading alternative

        print(f"[DataCollector] Crawled {len(documents)} documents")
        return documents

    def load_from_files(self, directory: str) -> List[LegalDocument]:
        """
        Load văn bản pháp luật từ files đã download sẵn.
        
        Args:
            directory: Thư mục chứa files văn bản (JSON format)
            
        Returns:
            Danh sách LegalDocument
        """
        documents = []
        dir_path = Path(directory)

        if not dir_path.exists():
            print(f"[DataCollector] Directory not found: {directory}")
            return documents

        json_files = list(dir_path.glob("*.json"))
        print(f"[DataCollector] Loading {len(json_files)} files from {directory}...")

        for file_path in tqdm(json_files, desc="Loading documents"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                doc = LegalDocument(
                    id=str(uuid.uuid4()),
                    title=data.get("title", ""),
                    content=data.get("content", ""),
                    metadata=DocumentMetadata(
                        document_number=data.get("document_number", ""),
                        document_type=data.get("document_type", ""),
                        title=data.get("title", ""),
                        issued_date=self._parse_date(data.get("issued_date")),
                        effective_date=self._parse_date(data.get("effective_date")),
                        issuing_body=data.get("issuing_body", ""),
                        category=data.get("category", ""),
                        status=data.get("status", "Không rõ"),
                    ),
                    source_url=data.get("source_url", ""),
                    crawled_at=datetime.now(),
                )

                # Validate document
                if self._validate_document(doc):
                    documents.append(doc)

            except (json.JSONDecodeError, KeyError) as e:
                print(f"[DataCollector] Error loading {file_path}: {e}")

        print(f"[DataCollector] Loaded {len(documents)} valid documents")
        return documents

    def load_alqac_dataset(self, path: str) -> List[QAPair]:
        """
        Load bộ dữ liệu ALQAC 2023.
        
        Args:
            path: Đường dẫn đến file/thư mục ALQAC dataset
            
        Returns:
            Danh sách QAPair
        """
        qa_pairs = []
        dataset_path = Path(path)

        if not dataset_path.exists():
            print(f"[DataCollector] ALQAC dataset not found: {path}")
            return qa_pairs

        print(f"[DataCollector] Loading ALQAC 2023 from {path}...")

        # Support both single file and directory
        if dataset_path.is_file():
            files = [dataset_path]
        else:
            files = list(dataset_path.glob("*.json")) + list(dataset_path.glob("*.jsonl"))

        for file_path in files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    if file_path.suffix == ".jsonl":
                        data = [json.loads(line) for line in f if line.strip()]
                    else:
                        data = json.load(f)
                        if isinstance(data, dict):
                            data = data.get("data", data.get("items", [data]))

                for item in data:
                    qa = QAPair(
                        id=str(uuid.uuid4()),
                        question=item.get("question", ""),
                        answer=item.get("answer", ""),
                        context=item.get("context", None),
                        source_document=item.get("source", None),
                        category=item.get("category", ""),
                        source="alqac",
                    )

                    # Validate Q&A pair
                    if len(qa.question) >= 10 and len(qa.answer) >= 20:
                        qa_pairs.append(qa)

            except (json.JSONDecodeError, KeyError) as e:
                print(f"[DataCollector] Error loading {file_path}: {e}")

        # Deduplicate
        seen_questions = set()
        unique_pairs = []
        for qa in qa_pairs:
            if qa.question not in seen_questions:
                seen_questions.add(qa.question)
                unique_pairs.append(qa)

        print(f"[DataCollector] Loaded {len(unique_pairs)} unique Q&A pairs from ALQAC")
        return unique_pairs

    def preprocess_document(self, raw_html: str) -> str:
        """
        Chuẩn hóa văn bản từ HTML sang plain text.
        
        Args:
            raw_html: HTML thô
            
        Returns:
            Plain text đã chuẩn hóa
        """
        soup = BeautifulSoup(raw_html, "html.parser")

        # Remove unwanted elements
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()

        # Get text
        text = soup.get_text(separator="\n")

        # Normalize
        text = self._normalize_text(text)

        return text

    def extract_metadata(self, text: str, source_url: str = "") -> DocumentMetadata:
        """
        Trích xuất metadata từ văn bản pháp luật.
        
        Args:
            text: Nội dung văn bản
            source_url: URL nguồn
            
        Returns:
            DocumentMetadata
        """
        metadata = DocumentMetadata()

        # Extract document number
        number_pattern = r"Số[:\s]*(\d+[-/]\d{4}[-/][A-ZĐa-zđ\-]+)"
        match = re.search(number_pattern, text)
        if match:
            metadata.document_number = match.group(1)

        # Extract issued date
        date_pattern = r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})"
        match = re.search(date_pattern, text)
        if match:
            day, month, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
            try:
                metadata.issued_date = datetime(year, month, day)
            except ValueError:
                pass

        # Extract document type
        type_patterns = {
            "Luật": r"\bLUẬT\b|\bLuật\b",
            "Nghị định": r"\bNGHỊ ĐỊNH\b|\bNghị định\b",
            "Thông tư": r"\bTHÔNG TƯ\b|\bThông tư\b",
            "Quyết định": r"\bQUYẾT ĐỊNH\b|\bQuyết định\b",
        }
        for doc_type, pattern in type_patterns.items():
            if re.search(pattern, text[:500]):
                metadata.document_type = doc_type
                break

        # Extract issuing body
        body_patterns = [
            (r"QUỐC HỘI", "Quốc hội"),
            (r"CHÍNH PHỦ", "Chính phủ"),
            (r"THỦ TƯỚNG", "Thủ tướng Chính phủ"),
            (r"BỘ\s+\w+", "Bộ"),
        ]
        for pattern, body in body_patterns:
            if re.search(pattern, text[:300]):
                metadata.issuing_body = body
                break

        return metadata

    def _validate_document(self, doc: LegalDocument) -> bool:
        """Validate document meets quality criteria."""
        if len(doc.content) < 100:
            return False
        if not doc.metadata.document_number and not doc.metadata.issued_date:
            return False
        return True

    def _normalize_text(self, text: str) -> str:
        """Normalize Vietnamese text."""
        import unicodedata
        # NFC normalization for Vietnamese
        text = unicodedata.normalize("NFC", text)
        # Remove multiple spaces
        text = re.sub(r"[ \t]+", " ", text)
        # Remove excessive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Strip lines
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)
        return text.strip()

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string to datetime."""
        if not date_str:
            return None
        formats = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"]
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return None
