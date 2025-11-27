from abc import ABC, abstractmethod
from typing import Iterable, List, Dict

Message = Dict[str, str]

class ChatModel(ABC):
    @abstractmethod
    def generate(self, messages: List[Message]) -> str:
        raise NotImplementedError

    def stream(self, messages: List[Message]) -> Iterable[str]:
        full = self.generate(messages)
        yield full
