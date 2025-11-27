from rag_service import LegalRAGService
from models.ollama_model import OllamaChatModel
import json

def debug_specific_file():
    print("=== Debugging Document 9 ===")
    model = OllamaChatModel(model_name="llama3")
    service = LegalRAGService(chat_model=model)

    filename = "doc_9_eO5c8v1ktT5n9Cq8GLP6hpyZf0JZmfVsA7P5dbAKwjQ=.pdf"
    query = "רבין יצחק"
    
    print(f"Checking if '{filename}' is retrieved for query: '{query}'")
    
    service.retriever.search_kwargs['k'] = 100
    docs = service.retrieve(query)
    
    found_rank = -1
    for i, doc in enumerate(docs):
        doc_name = doc.metadata.get('filename', '')
        if doc_name == filename:
            found_rank = i + 1
            print(f"\nSUCCESS: Document found at rank {found_rank}!")
            print(f"Content preview: {doc.page_content[:300]}...")
            break
            
    if found_rank == -1:
        print(f"\nFAILURE: Document '{filename}' NOT found in top {len(docs)} results.")
        
    if found_rank != -1:
        print("\nGenerating answer based on this context...")
        full_query = "מה אתה יודע על הפסק דין ברחוב יצחק רבין 2?"
        context = f"Source: {filename}\n{docs[found_rank-1].page_content}"
        messages = service._build_messages(full_query, context)
        
        print("\n--- Messages sent to Model ---")
        
        print("\n--- Model Response ---")
        answer = model.generate(messages)
        print(answer)

if __name__ == "__main__":
    debug_specific_file()
