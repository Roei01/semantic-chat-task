from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path

VECTOR_DB_DIR = Path("vectorstore")

def main():
    if not VECTOR_DB_DIR.exists():
        raise FileNotFoundError(
            f"Vector DB not found at {VECTOR_DB_DIR}. "
            f"Run build_index.py first."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=str(VECTOR_DB_DIR),
        collection_name="verdicts"
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    print("Semantic search test – type a question in Hebrew (or English).")
    query = input("שאלה: ")

    results = retriever.invoke(query)

    print("\n=== Top 3 relevant chunks ===\n")
    for i, doc in enumerate(results, start=1):
        print(f"--- Result #{i} ---")
        print("Source:", doc.metadata.get("filename"))
        print("Path  :", doc.metadata.get("source_path"))
        print("Text  :")
        print(doc.page_content[:500], "...")
        print()

if __name__ == "__main__":
    main()
