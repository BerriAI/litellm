import secrets
from typing import Final

from litellm._logging import verbose_proxy_logger
from litellm.proxy.common_utils.user_api_key_cache import (
    RegistryMemoryCache,
    UserApiKeyCache,
    get_management_object_ttl,
    registry_version_cache_key,
)


def _registry_version_seed() -> int:
    return secrets.randbelow(2**62)


async def bump_registry_version(registry_key: str, user_api_key_cache: UserApiKeyCache) -> None:
    version_key: Final = registry_version_cache_key(registry_key)
    redis_cache: Final = user_api_key_cache.redis_cache
    if redis_cache is not None:
        try:
            await redis_cache.async_seed_and_increment(version_key, _registry_version_seed())
        except Exception as e:
            verbose_proxy_logger.warning(
                "Failed to update registry version %s; a stale registry may be served until the next reload: %s",
                registry_key,
                e,
            )
            return
        redis_memory: Final[RegistryMemoryCache] = user_api_key_cache.in_memory_cache_for(version_key)
        redis_memory.delete_cache(version_key)
        return
    memory: Final[RegistryMemoryCache] = user_api_key_cache.in_memory_cache_for(version_key)
    current: Final = memory.get_cache(key=version_key)
    new_version: Final = (current if isinstance(current, int) else _registry_version_seed()) + 1
    memory.set_cache(key=version_key, value=new_version, ttl=get_management_object_ttl(user_api_key_cache))


async def current_registry_version_for_load(
    registry_key: str,
    user_api_key_cache: UserApiKeyCache,
) -> int | None:
    version_key: Final = registry_version_cache_key(registry_key)
    redis_cache: Final = user_api_key_cache.redis_cache
    if redis_cache is not None:
        try:
            return await redis_cache.async_get_or_seed(version_key, _registry_version_seed())
        except Exception as e:
            verbose_proxy_logger.warning(
                "Failed to read registry version %s; this registry load will not be cached: %s",
                registry_key,
                e,
            )
            return None
    memory: Final[RegistryMemoryCache] = user_api_key_cache.in_memory_cache_for(version_key)
    current: Final = memory.get_cache(key=version_key)
    if isinstance(current, int):
        return current
    seed: Final = _registry_version_seed()
    memory.set_cache(key=version_key, value=seed, ttl=get_management_object_ttl(user_api_key_cache))
    return seed
