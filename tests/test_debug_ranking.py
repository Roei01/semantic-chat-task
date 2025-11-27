from rag_service import LegalRAGService
from models.ollama_model import OllamaChatModel

def debug_ranking():
    model = OllamaChatModel(model_name="llama3")
    service = LegalRAGService(chat_model=model, top_k=100)

    filename_target = "doc_9_eO5c8v1ktT5n9Cq8GLP6hpyZf0JZmfVsA7P5dbAKwjQ=.pdf"
    query = "יצחק רבין 2"
    query_words = [w for w in query.split() if len(w) > 2]
    
    print(f"Query words: {query_words}")

    final_docs = service.retrieve(query)
    
    print(f"\nTop 8 docs returned:")
    for i, d in enumerate(final_docs):
        fname = d.metadata.get('filename', '')
        print(f"{i+1}. {fname}")
        if fname == filename_target:
            print("   *** TARGET FOUND IN TOP 8! ***")
            
    print("\n--- Deep Dive into Scoring ---")
    raw_docs = service.retriever.invoke(query)
    print(f"Raw retrieval count: {len(raw_docs)}")
    
    target_doc = None
    for d in raw_docs:
        if d.metadata.get('filename') == filename_target:
            target_doc = d
            break
            
    if not target_doc:
        print("CRITICAL: Target doc NOT in raw retrieval from ChromaDB!")
    else:
        print("Target doc FOUND in raw retrieval.")
        content = target_doc.page_content
        print(f"Content snippet: {content[:100]}...")
        
        score = 0
        for word in query_words:
            if word in content:
                print(f"  Match: '{word}' (+2)")
                score += 2
            elif word[::-1] in content:
                print(f"  Match: '{word[::-1]}' (reversed) (+2)")
                score += 2
            else:
                print(f"  No match for '{word}'")
        print(f"Calculated Score: {score}")

if __name__ == "__main__":
    debug_ranking()
