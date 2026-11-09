from collections.abc import Callable
from typing import Final, Protocol, cast

import pytest

from litellm._auth import runtime, rust_core
from litellm._auth.decision import AuthCore
from litellm._auth.python_core import PythonCore
from litellm._auth.shadow import ShadowCore
from litellm._auth.credential import PresentedKeys, SelectedKey, WebSocketDenial, WebSocketKeys


class CredentialSelector(Protocol):
    def select_api_key(self, keys: PresentedKeys) -> SelectedKey | None: ...
    def select_websocket_api_key(self, keys: WebSocketKeys) -> str | WebSocketDenial: ...
    def scheme_credential(self, value: str) -> str | None: ...
    def litellm_header_token(self, value: str) -> str: ...


@pytest.fixture(params=(PythonCore, rust_core.RustCore, ShadowCore), ids=("python", "rust", "shadow"))
def core(request: pytest.FixtureRequest) -> AuthCore:
    factory: Final = cast(Callable[[], AuthCore], request.param)
    return factory()


@pytest.fixture(params=("python", "rust", "shadow"))
def credentials(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> CredentialSelector:
    monkeypatch.setenv("LITELLM_AUTH_CORE", cast(str, request.param))
    return runtime
