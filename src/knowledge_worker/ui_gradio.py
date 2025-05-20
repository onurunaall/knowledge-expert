"""
Simple Gradio UI wrapper around the RAG chain.
"""

from __future__ import annotations

import gradio as gr

from .data_ingest import load_documents
from .rag_chain     import chat, get_chain

_CHAIN = None  # module-level cache

def _get_chain():
    """
    Fast path: use an existing persisted store when possible;
    fall back to loading documents only when necessary.
    """
    global _CHAIN  # noqa: PLW0603
    if _CHAIN is None:
        try:
            _CHAIN = get_chain(documents=None)
        except FileNotFoundError:
            print("UI: No vector store found — building from knowledge_base/ …")
            _CHAIN = get_chain(load_documents())      # slower path
    return _CHAIN

def _respond(message: str, history: list[list[str]]) -> list[list[str]]:
    """Return updated chat history for the gr.Chatbot component."""
    chain = _get_chain()
    tuple_hist = [(u, a) for u, a in history]
    answer, tuple_hist = chat(message, tuple_hist, chain)
    return [[u, a] for u, a in tuple_hist]

def launch() -> None:
    """Open the Gradio interface (blocks until closed)."""
    with gr.Blocks(title="📚 Knowledge Expert") as demo:
        gr.Markdown("# 📚 Knowledge Expert\nAsk anything about your knowledge base.")

        chatbot = gr.Chatbot(height=400)
        msg     = gr.Textbox(label="Your question", placeholder="Type and press ENTER")

        msg.submit(fn=_respond,
                   inputs=[msg, chatbot],
                   outputs=[chatbot],
                   queue=False
                  ).then(lambda: "", None, msg)

    demo.launch()

if __name__ == "__main__":
    launch()
