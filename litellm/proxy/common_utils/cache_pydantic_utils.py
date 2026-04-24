"""
DualCache presents a single API for reads and writes, but the two backends behave
differently: the in-memory layer can store arbitrary Python objects (including live
``BaseModel`` instances), while Redis persists strings and therefore needs JSON-safe
payloads (``json.dumps`` on the Redis side).

Call sites therefore see cache ``value`` / ``cached`` as effectively ``Any``: the same
key may deserialize to a model on one process (memory hit) or to a ``dict`` after a
Redis round-trip. ``CacheCodec`` centralizes encode/decode at that boundary:
``CacheCodec.serialize`` before ``set``, ``CacheCodec.deserialize`` after ``get``
when you need a typed ``BaseModel``.

``dataclasses`` are not supported: only ``dict`` and Pydantic ``BaseModel`` inputs
are encoded; pass a Pydantic model or convert with e.g. ``dataclasses.asdict`` first.
"""

from __future__ import annotations

from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from litellm._logging import verbose_proxy_logger

T = TypeVar("T", bound=BaseModel)


class CacheCodec:
    """
    Encode/decode Pydantic models for DualCache (memory vs Redis safe payloads).

    Dataclasses are not supported yet (only ``dict`` and ``BaseModel``).

    Use ``serialize`` with ``model_type`` when writing so the same schema is used
    as on read (``deserialize``). Pass ``model_type`` whenever you know it
    (validates ``dict`` payloads and normalizes ``BaseModel`` instances).
    """

    @staticmethod
    def serialize(value: Any, model_type: Optional[Type[T]] = None) -> Any:
        """
        Encode a value for DualCache / Redis (``json.dumps``-safe).

        If ``model_type`` is set, the payload is validated with that model, then
        ``model_dump(mode="json", exclude_none=True)`` — symmetric with ``deserialize``.

        If the value is already an instance of ``model_type`` (or a subclass),
        ``model_validate`` is skipped to avoid an unnecessary Pydantic copy — the
        value is dumped directly.

        If ``model_type`` is omitted, any ``BaseModel`` is dumped as above; other
        values (e.g. plain ``dict``) are returned unchanged.
        """
        if model_type is not None:
            if isinstance(value, model_type):
                # Already the right type: dump directly, skip re-validation copy.
                return value.model_dump(mode="json", exclude_none=True)
            if isinstance(value, (dict, BaseModel)):
                return model_type.model_validate(value).model_dump(
                    mode="json", exclude_none=True
                )
            return value
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json", exclude_none=True)
        return value

    @staticmethod
    def deserialize(cached: Any, model_type: Type[T]) -> Optional[T]:
        """
        Decode a cache entry to ``model_type``.

        - ``None`` → ``None``
        - Already an instance of ``model_type`` (including subclasses) → returned as-is
        - ``dict`` → ``model_type.model_validate(...)``; on ``ValidationError``,
          logs a warning and returns ``None`` (treat as cache miss; avoids serving
          malformed or schema-drifted entries)
        - Any other type → ``None`` (caller should treat as cache miss or log)
        """
        if cached is None:
            return None
        if isinstance(cached, model_type):
            return cached
        if isinstance(cached, dict):
            try:
                return model_type.model_validate(cached)
            except ValidationError as e:
                verbose_proxy_logger.warning(
                    "CacheCodec.deserialize: validation failed for %s (%s)",
                    model_type.__name__,
                    e,
                )
                return None
        return None
