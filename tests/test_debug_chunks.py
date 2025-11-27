from rag_service import LegalRAGService
from models.ollama_model import OllamaChatModel

def debug_all_chunks_of_doc():
    model = OllamaChatModel(model_name="llama3")
    service = LegalRAGService(chat_model=model, top_k=300) # Get everything

    filename_target = "doc_9_eO5c8v1ktT5n9Cq8GLP6hpyZf0JZmfVsA7P5dbAKwjQ=.pdf"
    query = "test" # Dummy query just to get docs
    
    raw_docs = service.retriever.invoke(query)
    
    print(f"Scanning {len(raw_docs)} chunks for {filename_target}...")
    
    found_chunks = []
    for d in raw_docs:
        if d.metadata.get('filename') == filename_target:
            found_chunks.append(d)

    print(f"Found {len(found_chunks)} chunks for this document.")
    
    for i, chunk in enumerate(found_chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk.page_content)
        if "רבין" in chunk.page_content:
             print(">>> FOUND 'Rabin' HERE <<<")

if __name__ == "__main__":
    debug_all_chunks_of_doc()

