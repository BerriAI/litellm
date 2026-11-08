import os
from typing import Final, Literal, TypeAlias

from litellm._auth import credential
from litellm._auth.credential import PresentedKeys, SelectedKey, WebSocketDenial, WebSocketKeys
from litellm._auth.decision import AuthCore
from litellm._auth.python_core import PythonCore

Backend: TypeAlias = Literal["python", "rust", "shadow"]


def backend() -> Backend:
    selected: Final = os.environ.get("LITELLM_AUTH_CORE", "rust")
    match selected:
        case "python" | "rust" | "shadow":
            return selected
        case _:
            raise ValueError("LITELLM_AUTH_CORE must be python, rust or shadow")


def get_auth_core(selected: Backend | None = None) -> AuthCore:
    match selected if selected is not None else backend():
        case "python":
            return PythonCore()
        case "rust":
            from litellm._auth.rust_core import RustCore

            return RustCore()
        case "shadow":
            from litellm._auth.shadow import ShadowCore

            return ShadowCore()


def scheme_credential(value: str) -> str | None:
    selected: Final = backend()
    if selected == "python":
        return credential.scheme_credential(value)
    from litellm._auth import rust_core

    result: Final = rust_core.scheme_credential(value)
    if selected == "shadow":
        from litellm._auth.shadow import compare

        compare(result, credential.scheme_credential(value))
    return result


def litellm_header_token(value: str) -> str:
    selected: Final = backend()
    if selected == "python":
        return credential.litellm_header_token(value)
    from litellm._auth import rust_core

    result: Final = rust_core.litellm_header_token(value)
    if selected == "shadow":
        from litellm._auth.shadow import compare

        compare(result, credential.litellm_header_token(value))
    return result


def select_api_key(keys: PresentedKeys) -> SelectedKey | None:
    selected: Final = backend()
    if selected == "python":
        return credential.select_api_key(keys)
    from litellm._auth import rust_core

    result: Final = rust_core.select_api_key(keys)
    if selected == "shadow":
        from litellm._auth.shadow import compare

        compare(result, credential.select_api_key(keys))
    return result


def select_websocket_api_key(keys: WebSocketKeys) -> str | WebSocketDenial:
    selected: Final = backend()
    if selected == "python":
        return credential.select_websocket_api_key(keys)
    from litellm._auth import rust_core

    result: Final = rust_core.select_websocket_api_key(keys)
    if selected == "shadow":
        from litellm._auth.shadow import compare

        compare(result, credential.select_websocket_api_key(keys))
    return result
