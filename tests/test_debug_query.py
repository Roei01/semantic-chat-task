from rag_service import LegalRAGService
from models.ollama_model import OllamaChatModel

def debug_query():
    print("=== Debugging Specific Query ===")
    model = OllamaChatModel(model_name="llama3")
    service = LegalRAGService(chat_model=model)

    query = "מה אתה יודע על הפסק דין ברחוב ברח' יצחק רבין 2?"
    print(f"Query: {query}\n")

    print("Searching vector store...")
    docs = service.retrieve(query)
    
    print(f"Found {len(docs)} relevant chunks.\n")
    
    found_address = False
    for i, doc in enumerate(docs):
        content = doc.page_content
        filename = doc.metadata.get('filename', 'unknown')
        print(f"--- Chunk {i+1} (Source: {filename}) ---")
        if "רבין" in content or "יצחק" in content:
            print(f"Found keyword in chunk: {content[:300]}...")
            found_address = True
        else:
            print(f"Content snippet: {content[:100]}...")
        print("\n")

    if not found_address:
        print("WARNING: The address 'Yitzhak Rabin' was NOT found in the top chunks!")
        print("This implies an embedding/indexing issue or PDF extraction issue.")
    else:
        print("SUCCESS: Address found in retrieved chunks.")
        
    print("Generating answer with new prompt...")
    answer, _ = service.answer(query)
    print("\nModel Answer:")
    print(answer)

if __name__ == "__main__":
    debug_query()
