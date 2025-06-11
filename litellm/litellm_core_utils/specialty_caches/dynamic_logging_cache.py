"""
This is a cache for LangfuseLoggers.

Langfuse Python SDK initializes a thread for each client. 

This ensures we do 
1. Proper cleanup of Langfuse initialized clients.
2. Re-use created langfuse clients.
"""
import hashlib
import json
from typing import Any, Optional

import litellm
from litellm.constants import _DEFAULT_TTL_FOR_HTTPX_CLIENTS

from ...caching import InMemoryCache


class LangfuseInMemoryCache(InMemoryCache):
    """
    Ensures we do proper cleanup of Langfuse initialized clients.

    Langfuse Python SDK initializes a thread for each client, we need to call Langfuse.shutdown() to properly cleanup.

    This ensures we do proper cleanup of Langfuse initialized clients.
    """

    def _remove_key(self, key: str) -> None:
        """
        Override _remove_key in InMemoryCache to ensure we do proper cleanup of Langfuse initialized clients.

        LangfuseLoggers consume threads when initalized, this shuts them down when they are expired

        Relevant Issue: https://github.com/BerriAI/litellm/issues/11169
        """
        from litellm.integrations.langfuse.langfuse import LangFuseLogger

        if isinstance(self.cache_dict[key], LangFuseLogger):
            _created_langfuse_logger: LangFuseLogger = self.cache_dict[key]
            #########################################################
            # Clean up Langfuse initialized clients
            #########################################################
            litellm.initialized_langfuse_clients -= 1
            _created_langfuse_logger.Langfuse.flush()
            _created_langfuse_logger.Langfuse.shutdown()

        #########################################################
        # Call parent class to remove key from cache
        #########################################################
        return super()._remove_key(key)


class DynamicLoggingCache:
    """
    Prevent memory leaks caused by initializing new logging clients on each request.

    Relevant Issue: https://github.com/BerriAI/litellm/issues/5695
    """

    def __init__(self) -> None:
        self.cache = LangfuseInMemoryCache(default_ttl=_DEFAULT_TTL_FOR_HTTPX_CLIENTS)

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
