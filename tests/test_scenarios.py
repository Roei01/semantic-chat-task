from rag_service import LegalRAGService
from models.ollama_model import OllamaChatModel

def test_questions():
    print("=== Testing 3 Scenarios ===")
    model = OllamaChatModel(model_name="llama3")
    service = LegalRAGService(chat_model=model, top_k=100)

    questions = [
        "איזה נושאים של פסקי דין יש לך ?",
        "מהו הפסק דין האחרון שלך ?",
        "מה הייתה המסקנה בפסק הדין של רבוע כחול נדל\"ן בע\"מ?"
    ]

    for q in questions:
        print(f"\n--------------------------------------------------")
        print(f"Question: {q}")
        print(f"--------------------------------------------------")
        
        docs = service.retrieve(q)
        print(f"Retrieved {len(docs)} docs. Top 3:")
        for i, d in enumerate(docs[:3]):
            print(f"  {i+1}. {d.metadata.get('filename')} (Score logic applied)")
            
        print("\nGenerating Answer...")
        answer, _ = service.answer(q)
        print(f"\nAnswer:\n{answer}\n")

if __name__ == "__main__":
    test_questions()
