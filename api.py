from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import json

from models.ollama_model import OllamaChatModel
from models.openai_model import OpenAIChatModel
from rag_service import LegalRAGService

app = FastAPI()

DOCS_DIR = Path("scraper/data")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    model_type: str = "ollama"
    model_name: Optional[str] = None

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        if req.model_type == "openai":
            try:
                model = OpenAIChatModel(model_name=req.model_name or "gpt-4o-mini")
            except RuntimeError as e:
                raise HTTPException(
                    status_code=500, 
                    detail=f"OpenAI API key not configured: {str(e)}. Please set API_GPT or OPENAI_API_KEY environment variable."
                )
        else:
            model = OllamaChatModel(model_name=req.model_name or "llama3")
            
        service = LegalRAGService(chat_model=model)
        stream, citations = service.stream_answer(req.question)
        
        async def generator():
            yield json.dumps({"type": "citations", "data": citations}) + "\n"
            for token in stream:
                yield json.dumps({"type": "token", "data": token}) + "\n"
                
        return StreamingResponse(generator(), media_type="application/x-ndjson")
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)

@app.get("/api/files/{filename}")
async def get_file(filename: str):
    file_path = DOCS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/octet-stream"
    )

@app.get("/health")
def health():
    return {"status": "ok"}

