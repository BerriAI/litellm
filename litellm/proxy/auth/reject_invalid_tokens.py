"""
Negative cache for unknown virtual keys (reduces repeat DB work on bad ``sk-`` hashes).

:class:`InvalidVirtualKeyCache` holds:

- :meth:`InvalidVirtualKeyCache.configured_ttl_seconds` — TTL from proxy settings, else
  :data:`~litellm.constants.DEFAULT_INVALID_VIRTUAL_KEY_NEGATIVE_CACHE_TTL_SECONDS`.
- :meth:`InvalidVirtualKeyCache.check_invalid_token` — ``sk-`` format, then optional negative cache +
  ``LiteLLM_VerificationToken`` probe. Returns ``True`` if the client should get **401** (except
  malformed keys, which raise ``HTTPException``). Returns ``False`` if preflight passed—hash the raw
  key and load from ``combined_view``.
- :meth:`InvalidVirtualKeyCache.allows_db_lookup` / :meth:`InvalidVirtualKeyCache.record_miss` — lower-level helpers.

Cache keys: ``invalid_vk:{hashed_token}``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

from fastapi import HTTPException, status

from litellm.constants import DEFAULT_INVALID_VIRTUAL_KEY_NEGATIVE_CACHE_TTL_SECONDS
from litellm._logging import verbose_logger, verbose_proxy_logger
from litellm.proxy._types import hash_token

INVALID_VIRTUAL_KEY_CACHE_PREFIX = "invalid_vk:"


class InvalidVirtualKeyCache:
    """Settings + negative cache for virtual keys that are not in the DB (or not yet)."""

    _prefix = INVALID_VIRTUAL_KEY_CACHE_PREFIX

    @staticmethod
    def configured_ttl_seconds(
        general_settings: Union[Dict[str, Any], Any],
    ) -> Optional[float]:
        """
        TTL for negative-caching unknown virtual keys.

        Reads ``invalid_virtual_key_cache_ttl`` from ``general_settings`` (top-level or
        ``litellm_settings``). If unset, uses
        :data:`litellm.constants.DEFAULT_INVALID_VIRTUAL_KEY_NEGATIVE_CACHE_TTL_SECONDS`.
        A configured value of ``0`` or less disables negative caching (returns ``None``).
        """
        raw: Any = None
        if isinstance(general_settings, dict):
            raw = general_settings.get("invalid_virtual_key_cache_ttl")
            if raw is None:
                litellm_settings = general_settings.get("litellm_settings")
                if isinstance(litellm_settings, dict):
                    raw = litellm_settings.get("invalid_virtual_key_cache_ttl")
        default_ttl = float(DEFAULT_INVALID_VIRTUAL_KEY_NEGATIVE_CACHE_TTL_SECONDS)
        try:
            if raw is None:
                return default_ttl
            v = float(raw)
        except (TypeError, ValueError):
            return default_ttl
        if v <= 0:
            return None
        return v

    @classmethod
    def _cache_key(cls, hashed_token: str) -> str:
        return "{}{}".format(cls._prefix, hashed_token)

    @classmethod
    async def delete_invalid_token_cache(
        cls,
        *,
        hashed_token: str,
        user_api_key_cache: Any,
    ) -> None:
        """Clear a stale negative-cache entry after the token is created/restored."""
        try:
            await user_api_key_cache.async_delete_cache(
                key=cls._cache_key(hashed_token)
            )
        except Exception as e:
            verbose_proxy_logger.debug(
                "InvalidVirtualKeyCache.delete_invalid_token_cache: %s", e
            )

    @classmethod
    async def allows_db_lookup(
        cls,
        *,
        hashed_token: str,
        user_api_key_cache: Any,
        ttl_seconds: Optional[float],
    ) -> bool:
        """
        ``True`` if this hash may hit the database (entry not in the negative cache).

        When ``ttl_seconds`` is ``None``, negative caching is off — always ``True``.
        """
        if ttl_seconds is None:
            return True

        key = cls._cache_key(hashed_token)
        try:
            cached = await user_api_key_cache.async_get_cache(key=key)
        except Exception as e:
            verbose_proxy_logger.debug(
                "InvalidVirtualKeyCache.allows_db_lookup: cache read failed, allowing query: %s",
                e,
            )
            return True

        return cached is None

    @classmethod
    async def record_miss(
        cls,
        *,
        hashed_token: str,
        user_api_key_cache: Any,
        ttl_seconds: float,
    ) -> None:
        """Remember this hash as a failed lookup until ``ttl_seconds`` elapses."""
        if ttl_seconds <= 0:
            return
        key = cls._cache_key(hashed_token)
        try:
            await user_api_key_cache.async_set_cache(
                key=key,
                value="",
                ttl=ttl_seconds,
            )
        except Exception as e:
            verbose_proxy_logger.debug("InvalidVirtualKeyCache.record_miss: %s", e)

    @classmethod
    async def check_invalid_token(
        cls,
        *,
        api_key: Any,
        prisma_client: Any,
        user_api_key_cache: Any,
        general_settings: Any,
    ) -> bool:
        """
        Virtual-key preflight: ``sk-`` shape → (if TTL on) negative cache →
        ``litellm_verificationtoken`` row check.

        Returns ``True`` if the request should be rejected as an invalid virtual key (**401**).
        Returns ``False`` if preflight passed; caller should ``hash_token(api_key)`` then call
        ``get_key_object``.

        Malformed keys raise ``HTTPException`` (401) with masking details instead of returning bool.
        """
        if isinstance(api_key, str):
            _masked_key = (
                "{}****{}".format(api_key[:4], api_key[-4:])
                if len(api_key) > 8
                else "****"
            )
            if not api_key.startswith("sk-"):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=(
                        "LiteLLM Virtual Key expected. Received={}, expected to start with 'sk-'.".format(
                            _masked_key
                        )
                    ),
                )
        else:
            verbose_logger.warning(
                "litellm.proxy.proxy_server.user_api_key_auth(): Warning - Key is not a string. Got type={}".format(
                    type(api_key) if api_key is not None else "None"
                )
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="LiteLLM Virtual Key expected.",
            )

        return await cls.check_invalid_hashed_token(
            hashed_token=hash_token(token=api_key),
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            general_settings=general_settings,
        )

    @classmethod
    async def check_invalid_hashed_token(
        cls,
        *,
        hashed_token: str,
        prisma_client: Any,
        user_api_key_cache: Any,
        general_settings: Any,
    ) -> bool:
        """
        Hashed-token preflight for callers that no longer have the raw ``sk-`` key.

        Returns ``True`` if the request should be rejected as an invalid virtual key.
        Returns ``False`` if preflight passed; caller may load the full key object.
        """
        ttl_seconds = cls.configured_ttl_seconds(general_settings)

        if ttl_seconds is None:
            return False

        if not await cls.allows_db_lookup(
            hashed_token=hashed_token,
            user_api_key_cache=user_api_key_cache,
            ttl_seconds=ttl_seconds,
        ):
            return True

        token_probe_failed = False
        try:
            token_row = await prisma_client.db.litellm_verificationtoken.find_first(
                where={"token": hashed_token},
            )
        except Exception as e:
            verbose_proxy_logger.debug(
                "InvalidVirtualKeyCache.check_invalid_hashed_token: verification token probe failed, continuing to combined_view: %s",
                e,
            )
            token_probe_failed = True
            token_row = None

        if not token_probe_failed and token_row is None:
            await cls.record_miss(
                hashed_token=hashed_token,
                user_api_key_cache=user_api_key_cache,
                ttl_seconds=ttl_seconds,
            )
            return True

        return False
