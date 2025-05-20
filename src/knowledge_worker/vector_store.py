"""
Build or load a persistent Chroma vector store.
"""

from __future__ import annotations

import shutil
from typing import List, Sequence

from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from .config import VECTOR_STORE_DIR, settings

_COLLECTION_NAME = "knowledge"


def _split_documents(docs: Sequence[Document]) -> List[Document]:
    splitter = CharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    return splitter.split_documents(list(docs))


def _create_store(docs: Sequence[Document]) -> Chroma:
    if VECTOR_STORE_DIR.exists():
        shutil.rmtree(VECTOR_STORE_DIR)
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    splits = _split_documents(docs)
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model, api_key=settings.openai_api_key
    )

    return Chroma.from_documents(
        splits,
        embedding=embeddings,
        persist_directory=str(VECTOR_STORE_DIR),
        collection_name=_COLLECTION_NAME,
    )


def _load_store() -> Chroma | None:
    if not (VECTOR_STORE_DIR / "index").exists():
        return None

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model, api_key=settings.openai_api_key
    )
    try:
        store = Chroma(
            collection_name=_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(VECTOR_STORE_DIR),
        )
        _ = store._collection.count()  # quick sanity check
        return store
    except Exception:  # noqa: BLE001
        return None


def get_store(
    documents: Sequence[Document] | None = None, *, force_rebuild: bool = False
) -> Chroma:
    if force_rebuild:
        if documents is None:
            raise ValueError("Must provide `documents` when force_rebuild=True")
        store = _create_store(documents)
        store.persist()
        return store

    store = _load_store()
    if store:
        return store

    if documents is None:
        raise FileNotFoundError(
            "Vector store not found. Supply `documents` to build one."
        )
    store = _create_store(documents)
    store.persist()
    return store


__all__ = ["get_store"]
