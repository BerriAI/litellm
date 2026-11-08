from litellm._auth import rust_core
from litellm._auth.credential import PresentedKeys, SelectedKey, WebSocketDenial, WebSocketKeys
from litellm._auth.decision import AuthCore


def get_auth_core() -> AuthCore:
    return rust_core.RustCore()


def scheme_credential(value: str) -> str | None:
    return rust_core.scheme_credential(value)


def litellm_header_token(value: str) -> str:
    return rust_core.litellm_header_token(value)


def select_api_key(keys: PresentedKeys) -> SelectedKey | None:
    return rust_core.select_api_key(keys)


def select_websocket_api_key(keys: WebSocketKeys) -> str | WebSocketDenial:
    return rust_core.select_websocket_api_key(keys)
