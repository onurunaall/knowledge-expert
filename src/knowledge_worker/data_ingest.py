"""
Load raw documents from ``knowledge_base/`` (or another directory) and return
a flat ``list[Document]`` ready for text-splitting / embedding.

Supported file types (ext → loader):
    • .md / .markdown   → UnstructuredMarkdownLoader
    • .txt              → TextLoader
    • .pdf              → UnstructuredPDFLoader

Each returned ``Document`` gets ``metadata["source"]`` containing the path
relative to the base directory.

Usage
-----
>>> from knowledge_worker.data_ingest import load_documents
>>> docs = load_documents()           # defaults to config.KNOWLEDGE_BASE_DIR
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, List, Sequence

from langchain.docstore.document import Document
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPDFLoader,
)

from .config import KNOWLEDGE_BASE_DIR

# --------------------------------------------------------------------------- #
# Loader registry                                                             #
# --------------------------------------------------------------------------- #

_LOADER_MAP = {
    ".md": UnstructuredMarkdownLoader,
    ".markdown": UnstructuredMarkdownLoader,
    ".txt": TextLoader,
    ".pdf": UnstructuredPDFLoader,
}


def _iter_files(base_path: Path) -> Iterable[Path]:
    for path in base_path.rglob("*"):
        if path.is_file():
            yield path


def _select_loader(file_path: Path):
    return _LOADER_MAP.get(file_path.suffix.lower())


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def load_documents(base_path: Path | str | None = None) -> List[Document]:
    root = Path(base_path) if base_path else KNOWLEDGE_BASE_DIR
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")

    documents: List[Document] = []

    for file_path in _iter_files(root):
        loader_cls = _select_loader(file_path)
        if loader_cls is None:
            warnings.warn(f"Skipped unsupported file type: {file_path.name}")
            continue

        try:
            loader = loader_cls(str(file_path))
            docs = loader.load()
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Failed to load {file_path.name}: {exc}")
            continue

        rel_path = file_path.relative_to(root).as_posix()
        for doc in docs:
            doc.metadata.setdefault("source", rel_path)

        documents.extend(docs)

    if not documents:
        raise FileNotFoundError(f"No supported documents found in: {root}")

    return documents


__all__: Sequence[str] = ["load_documents"]
