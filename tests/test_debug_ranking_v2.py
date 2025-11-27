from rag_service import LegalRAGService
from models.ollama_model import OllamaChatModel

def debug_ranking_logic():
    model = OllamaChatModel(model_name="llama3")
    service = LegalRAGService(chat_model=model, top_k=100)

    query = 'מה הייתה המסקנה בפסק הדין של רבוע כחול נדל"ן בע"מ?'
    
    stop_words = {
        "של", "את", "על", "כי", "זה", "או", "כל", "הוא", "היא", "גם", "בין", "רק", "אך", "אין", "יש",
        "מה", "מי", "איך", "כיצד", "מתי", "איפה", "למה", "מדוע", "האם",
        "היה", "היתה", "היו", "תהיה", "יהיה",
        "פסק", "דין", "בית", "משפט", "החלטה", "תביעה", "נתבעת", "תובעת", "נגד", "בפני", "ב"
    }
    
    raw_words = [w.strip('.,?"\'').lower() for w in query.split()]
    words = [w for w in raw_words if len(w) > 1 and w not in stop_words]
    
    print(f"Filtered Query Words: {words}")
    
    docs = service.retriever.invoke(query)
    print(f"Raw docs retrieved: {len(docs)}")
    
    top_doc = docs[0]
    
    print(f"Checking top doc: {top_doc.metadata.get('filename')}")
    content = top_doc.page_content
    
    score = 1.0
    for word in words:
        if word in content:
            print(f"Match: {word}")
            score += 2.0
        elif word[::-1] in content:
            print(f"Match Reverse: {word}")
            score += 2.0
        elif len(word) > 3 and word in content.replace(" ", ""):
            print(f"Match Partial: {word}")
            score += 0.5
            
    print(f"Total Score: {score}")

if __name__ == "__main__":
    debug_ranking_logic()
