from typing import Final

from litellm.rust_bridge.loader import get_native_bridge
from litellm.rust_bridge.provenance import (
    NATIVE_NON_RESPONSE_ENTRYPOINTS,
    NATIVE_RESPONSE_ENTRYPOINTS,
)

MARKER_CONTRACT_ENTRYPOINTS: Final = frozenset(
    {
        "achat_completions",
        "amessages",
        "aocr",
        "atranscription",
        "chat_completions",
        "messages",
        "ocr",
        "transcription",
    }
)


def test_native_route_catalogue_matches_extension_exports() -> None:
    native: Final = get_native_bridge()
    assert native is not None
    route_exports: Final = frozenset(
        name
        for name in dir(native)
        if not name.startswith("_") and name[0].islower()
    )

    assert route_exports == NATIVE_RESPONSE_ENTRYPOINTS | NATIVE_NON_RESPONSE_ENTRYPOINTS


def test_every_response_entrypoint_has_marker_contract_coverage() -> None:
    assert MARKER_CONTRACT_ENTRYPOINTS == NATIVE_RESPONSE_ENTRYPOINTS
