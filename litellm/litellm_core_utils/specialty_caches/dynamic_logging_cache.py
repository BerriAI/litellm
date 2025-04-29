import hashlib
import json
from typing import Any, Optional

from ...caching import InMemoryCache


class DynamicLoggingCache:
    """
    Prevent memory leaks caused by initializing new logging clients on each request.

    Relevant Issue: https://github.com/BerriAI/litellm/issues/5695
    """

    def __init__(self) -> None:
        self.cache = InMemoryCache()

    def get_cache_key(self, args: dict) -> str:
        args_str = json.dumps(args, sort_keys=True)
        cache_key = hashlib.sha256(args_str.encode("utf-8")).hexdigest()
        return cache_key

    def get_cache(self, credentials: dict, service_name: str) -> Optional[Any]:
        key_name = self.get_cache_key(
            args={**credentials, "service_name": service_name}
        )
        response = self.cache.get_cache(key=key_name)
        return response

    def set_cache(self, credentials: dict, service_name: str, logging_obj: Any) -> None:
        key_name = self.get_cache_key(
            args={**credentials, "service_name": service_name}
        )
        self.cache.set_cache(key=key_name, value=logging_obj)
        return None
