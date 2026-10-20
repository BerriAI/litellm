"""With the Rust bridge opted in, the proxy's async /v1/messages path still stamps ``safeguard_results`` on native
providers in both modes."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final

import pytest
from integration._support.process import graceful_stop_seconds
from integration._support.safeguards_rig import (
    NATIVE_MODEL,
    auto_mode_body,
    flagged,
    json_body,
    native_backend,
    native_message,
    nonce,
    one_classifier_call,
    safeguards_rig,
    stream_results,
    tool_id,
    tool_use_block,
    tool_use_ids,
)

pytestmark = pytest.mark.timeout(2 * graceful_stop_seconds() + 240)


def rust_environment(wire_url: str) -> Mapping[str, str]:
    return {"LITELLM_RUST": "1"}


def test_rust_opt_in_keeps_the_verdict_on_native_messages(tmp_path: Path) -> None:
    with safeguards_rig(tmp_path, environment=rust_environment) as rig:
        request_nonce: Final = nonce()
        identity: Final = tool_id(request_nonce)
        rig.register(
            request_nonce, native_backend(native_message(identity, content=(tool_use_block(identity),)), stream=False)
        )
        body: Final = json_body(rig.post(auto_mode_body(NATIVE_MODEL, request_nonce)))
        assert tool_use_ids(body) == (identity,), body
        assert body["safeguard_results"] == flagged((identity,)), body
        assert len(one_classifier_call(rig, request_nonce)[1]) == 1
        stream_nonce: Final = nonce()
        stream_identity: Final = tool_id(stream_nonce)
        rig.register(
            stream_nonce,
            native_backend(native_message(stream_identity, content=(tool_use_block(stream_identity),)), stream=True),
        )
        status, raw = rig.stream(auto_mode_body(NATIVE_MODEL, stream_nonce))
        assert status == 200, raw
        assert stream_results(raw) == flagged((stream_identity,)), raw
        assert len(one_classifier_call(rig, stream_nonce)[1]) == 1
