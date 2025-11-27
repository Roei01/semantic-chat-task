from __future__ import annotations

from typing import Literal

from rag_service import LegalRAGService
from models.ollama_model import OllamaChatModel

def build_service(backend: Literal["ollama", "openai"] = "ollama") -> LegalRAGService:
    if backend == "ollama":
        model = OllamaChatModel(model_name="llama3")
    elif backend == "openai":
        from models.openai_model import OpenAIChatModel
        model = OpenAIChatModel(model_name="gpt-4o-mini")
    else:
        raise ValueError(f"Unknown backend: {backend}")

    return LegalRAGService(chat_model=model, top_k=5)

def main():
    service = build_service(backend="ollama")

    print("=== Legal Semantic Chat (CLI) ===")
    print("type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        print("Assistant (streaming): ", end="", flush=True)

        stream, citations = service.stream_answer(question)

        answer_text = ""
        for token in stream:
            answer_text += token
            print(token, end="", flush=True)
        print("\n")

        print("Sources:")
        for c in citations:
            print(f"  {c['id']} -> {c['filename']} ({c['source_path']})")
        print("-" * 40)

if __name__ == "__main__":
    main()
