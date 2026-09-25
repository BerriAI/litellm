from typing import Final

from ...caching import InMemoryCache


class ServiceTraceIDCache:
    def __init__(self) -> None:
        self.cache = InMemoryCache()

    def get_cache(self, litellm_call_id: str, service_name: str) -> str | None:
        key_name: Final = f"{service_name}:{litellm_call_id}"
        response: Final = self.cache.get_cache(key=key_name)
        return response

    def set_cache(self, litellm_call_id: str, service_name: str, trace_id: str) -> None:
        key_name: Final = f"{service_name}:{litellm_call_id}"
        self.cache.set_cache(key=key_name, value=trace_id)


in_memory_trace_id_cache: Final = ServiceTraceIDCache()
