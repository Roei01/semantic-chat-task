import os
from typing import Iterable, List
from openai import OpenAI
from .base import ChatModel, Message

class OpenAIChatModel(ChatModel):
    def __init__(self, model_name: str = "gpt-4o-mini"):
        api_key = os.getenv("API_GPT") or os.getenv("OPENAI_API_KEY")
            
        if not api_key:
            raise RuntimeError("API_GPT or OPENAI_API_KEY not set. Please set one of them in your environment: export API_GPT='your-key-here'")
            
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate(self, messages: List[Message]) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=512
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            return f"שגיאה בתקשורת עם OpenAI: {str(e)}"

    def stream(self, messages: List[Message]) -> Iterable[str]:
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True,
                temperature=0.1,
                max_tokens=512
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        except Exception as e:
            yield f"שגיאה בתקשורת עם OpenAI: {str(e)}"
