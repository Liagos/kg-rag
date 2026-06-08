from abc import ABC, abstractmethod
from .schemas import RetrievedDocument


class BaseRetriever(ABC):

    @abstractmethod
    def retrieve(self, query: str,
                 k: int = 5,
                 filters: dict | None = None,
                 ) -> list[RetrievedDocument]:
        pass
