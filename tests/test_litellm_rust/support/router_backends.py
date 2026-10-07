"""Both router backends built from the same arguments and seed, for parity tests."""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from typing import Final, Protocol

from litellm.router_backends.python_router import PythonRouter
from litellm.router_backends.rust_router import NATIVE_ROUTER, RustRouter


class Backend(Protocol):
    async def acompletion(
        self, model: str, messages: Sequence[Mapping[str, object]], **kwargs: object
    ) -> object: ...  # kwargs-ok: Router.acompletion's surface

    def completion(
        self, model: str, messages: Sequence[Mapping[str, object]], **kwargs: object
    ) -> object: ...  # kwargs-ok: Router.completion's surface

    async def aresponses(self, **kwargs: object) -> object: ...  # kwargs-ok: Router.aresponses' surface

    async def aanthropic_messages(
        self, **kwargs: object
    ) -> object: ...  # kwargs-ok: Router.aanthropic_messages' surface


def _lists(arguments: Mapping[str, object]) -> dict[str, object]:  # mutable-ok: Router(...) keyword arguments
    return {key: list(value) if isinstance(value, tuple) else value for key, value in arguments.items()}


def python_backend(arguments: Mapping[str, object], seed: int) -> Backend:
    random.seed(seed)
    return PythonRouter(**_lists(arguments))  # pyright: ignore[reportReturnType]  # PythonRouter's untyped signatures


def rust_backend(arguments: Mapping[str, object], seed: int) -> Backend:
    native: Final = NATIVE_ROUTER.load()
    assert native is not None
    return RustRouter(_lists(arguments), native, seed)


BACKENDS: Final[tuple[Callable[[Mapping[str, object], int], Backend], ...]] = (python_backend, rust_backend)


def router_headers(response: object) -> Mapping[str, object]:
    hidden: Final = getattr(response, "_hidden_params", {})
    headers: Final = hidden.get("additional_headers", {}) if isinstance(hidden, Mapping) else {}
    return {key: value for key, value in headers.items() if key.startswith("x-litellm-")}
