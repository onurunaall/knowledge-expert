"""
Conversation-ready RAG chain (OpenAI chat + Chroma retriever).
"""

from __future__ import annotations

from typing import List, Tuple

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

from .config import settings
from .vector_store import get_store


def _make_chain(store: Chroma) -> ConversationalRetrievalChain:
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0.0,
        api_key=settings.openai_api_key,
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=store.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        verbose=False,
    )


def get_chain(
    documents=None, *, force_rebuild: bool = False
) -> ConversationalRetrievalChain:
    store = get_store(documents, force_rebuild=force_rebuild)
    return _make_chain(store)


def chat(
    question: str,
    history: List[Tuple[str, str]],
    chain: ConversationalRetrievalChain,
) -> Tuple[str, List[Tuple[str, str]]]:
    result = chain({"question": question, "chat_history": history})
    answer = result["answer"]
    history.append((question, answer))
    return answer, history


__all__ = ["get_chain", "chat"]
