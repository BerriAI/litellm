from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from litellm.proxy._types import TokenCountResponse


class TokenCounterInterface(ABC):
    @abstractmethod
    async def count_tokens(
        self,
        model_to_use: str,
        messages: Optional[List[Dict[str, Any]]],
        deployment: Optional[Dict[str, Any]] = None,
        request_model: str = "",
    ) -> Optional[TokenCountResponse]:
        pass

    @abstractmethod
    def supports_provider(
        self, 
        deployment: Optional[Dict[str, Any]] = None,
        from_endpoint: bool = False
    ) -> bool:
        pass


class TokenCountingFactory:
    _counters: List[TokenCounterInterface] = []
    
    @classmethod
    def register_counter(cls, counter: TokenCounterInterface) -> None:
        cls._counters.append(counter)
    
    @classmethod
    def get_counter(
        cls, 
        deployment: Optional[Dict[str, Any]] = None,
        from_endpoint: bool = False
    ) -> Optional[TokenCounterInterface]:
        for counter in cls._counters:
            if counter.supports_provider(deployment=deployment, from_endpoint=from_endpoint):
                return counter
        return None