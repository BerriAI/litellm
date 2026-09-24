from __future__ import annotations

from collections.abc import Awaitable, Mapping
from typing import Final, Protocol, cast  # noqa: TID251  # validates dynamically loaded native callables

from litellm.rust_bridge.bindings import NativeBinding


class RustRetrieveBatch(Protocol):
    def __call__(
        self,
        batch_id: str,
        api_key: str | None,
        api_base: str | None,
        extra_headers: Mapping[str, str] | None,
        timeout_seconds: float | None,
    ) -> Mapping[str, object]:
        raise NotImplementedError


class RustAretrieveBatch(Protocol):
    def __call__(
        self,
        batch_id: str,
        api_key: str | None,
        api_base: str | None,
        extra_headers: Mapping[str, str] | None,
        timeout_seconds: float | None,
    ) -> Awaitable[Mapping[str, object]]:
        raise NotImplementedError


class RustCreateBatch(Protocol):
    def __call__(
        self,
        input_jsonl: str,
        model: str | None,
        api_key: str | None,
        api_base: str | None,
        extra_headers: Mapping[str, str] | None,
        timeout_seconds: float | None,
    ) -> Mapping[str, object]:
        raise NotImplementedError


class RustAcreateBatch(Protocol):
    def __call__(
        self,
        input_jsonl: str,
        model: str | None,
        api_key: str | None,
        api_base: str | None,
        extra_headers: Mapping[str, str] | None,
        timeout_seconds: float | None,
    ) -> Awaitable[Mapping[str, object]]:
        raise NotImplementedError


def _retrieve_binding(value: object) -> RustRetrieveBatch | None:
    if not callable(value):
        return None
    return cast("RustRetrieveBatch", value)  # cast-ok: callable validated at the native binding boundary


def _aretrieve_binding(value: object) -> RustAretrieveBatch | None:
    if not callable(value):
        return None
    return cast("RustAretrieveBatch", value)  # cast-ok: callable validated at the native binding boundary


def _create_binding(value: object) -> RustCreateBatch | None:
    if not callable(value):
        return None
    return cast("RustCreateBatch", value)  # cast-ok: callable validated at the native binding boundary


def _acreate_binding(value: object) -> RustAcreateBatch | None:
    if not callable(value):
        return None
    return cast("RustAcreateBatch", value)  # cast-ok: callable validated at the native binding boundary


NATIVE_RETRIEVE_BATCH: Final = NativeBinding("retrieve_batch", validate=_retrieve_binding)
NATIVE_ARETRIEVE_BATCH: Final = NativeBinding("aretrieve_batch", validate=_aretrieve_binding)
NATIVE_CREATE_BATCH: Final = NativeBinding("create_batch", validate=_create_binding)
NATIVE_ACREATE_BATCH: Final = NativeBinding("acreate_batch", validate=_acreate_binding)
