import json
import requests
from typing import Iterable, List
from .base import ChatModel, Message

class OllamaChatModel(ChatModel):
    def __init__(self, model_name: str = "llama3", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

    def generate(self, messages: List[Message]) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_ctx": 2048,
                "num_predict": 512
            }
        }
        try:
            resp = requests.post(url, json=payload, timeout=45)
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content", "")
        except requests.RequestException as e:
            return f"שגיאה בתקשורת עם המודל: {str(e)}"

    def stream(self, messages: List[Message]) -> Iterable[str]:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": 0.1,
                "num_ctx": 2048,
                "num_predict": 512
            }
        }
        try:
            with requests.post(url, json=payload, stream=True, timeout=45) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            content = data.get("message", {}).get("content", "")
                            if content:
                                yield content
                        except ValueError:
                            pass
        except requests.RequestException:
            yield "שגיאה: לא ניתן להתחבר למודל המקומי."
