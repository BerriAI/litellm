from __future__ import annotations

from typing import Final

import pytest

from litellm.rust_bridge import _native
from litellm.rust_bridge.configuration import RouteName
from litellm.rust_bridge.route import NativeRoute
from litellm.rust_bridge.runtime import BridgeErrorContext, FallbackMode, invoke

pytestmark = pytest.mark.requires_rust_extension

UNIMPLEMENTED: Final = (
    RouteName.MESSAGES,
    RouteName.CHAT_COMPLETIONS,
    RouteName.TRANSCRIPTION,
    RouteName.EMBEDDINGS,
    RouteName.RERANK,
    RouteName.IMAGE_GENERATION,
    RouteName.IMAGE_EDIT,
    RouteName.SPEECH,
    RouteName.MODERATION,
    RouteName.RESPONSES,
)


class UntouchedInput:
    def __getattribute__(self, name: str) -> object:
        raise AssertionError(f"unimplemented route inspected {name}")


@pytest.mark.parametrize("route_name", UNIMPLEMENTED)
@pytest.mark.parametrize("asynchronous", (False, True))
def test_unimplemented_lifecycle_declines_without_input_reads(route_name: RouteName, asynchronous: bool) -> None:
    route: Final = NativeRoute(route_name)
    native: Final = route.select(route.lifecycle())
    assert native is not None
    request: Final = UntouchedInput()
    with pytest.raises(_native.RustBridgeDeclined, match=f"^{route_name.value} native lifecycle is not implemented$"):
        native(request, (request,), {"callback": request, "file": request}, asynchronous)


@pytest.mark.parametrize("route_name", UNIMPLEMENTED)
def test_stub_decline_enters_fallback_once(route_name: RouteName) -> None:
    route: Final = NativeRoute(route_name)
    native: Final = route.select(route.lifecycle())
    assert native is not None
    fallback_results: Final = iter(("python result",))
    result: Final = invoke(
        native_call=lambda: native(UntouchedInput(), (), {}, False),
        fallback=lambda: next(fallback_results),
        adapt=str,
        mode=FallbackMode.PYTHON,
        context=BridgeErrorContext(route=route_name.value, model="unused", provider="unused"),
    )
    assert result == "python result"
    with pytest.raises(StopIteration):
        next(fallback_results)
