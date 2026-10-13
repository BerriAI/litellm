from typing import Final

from litellm.caching.caching import Cache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.caching import LiteLLMCacheType


def test_authenticated_cache_keys_preserve_request_and_metadata_namespaces() -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL, namespace="instance")
    caller: Final = UserAPIKeyAuth(api_key="fixture", token="hashed-key", team_id="team")

    key: Final = cache._add_namespace_to_cache_key(
        "hashed-response",
        cache={"namespace": "request"},
        metadata={"user_api_key_auth": caller, "redis_namespace": "metadata"},
    )

    assert key is not None
    assert key.startswith("instance:caller:")
    assert key.endswith(":request:metadata:hashed-response")
