"""The ``safeguards_classifier_model`` setting: unset, empty, null, or not a string leaves the reply exactly as it was
before the feature, and a name the proxy does not serve answers ``unsupported`` with one warning instead of failing the
request."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
from integration._support.client import eventually
from integration._support.process import graceful_stop_seconds
from integration._support.safeguards_rig import (
    CHAT_MODEL,
    OMITTED,
    UNSUPPORTED,
    Omitted,
    Rig,
    auto_mode_body,
    chat_backend,
    json_body,
    nonce,
    safeguards_rig,
    tool_id,
    tool_use_ids,
)
from pydantic import JsonValue

pytestmark = pytest.mark.timeout(2 * graceful_stop_seconds() + 240)

UNKNOWN_CLASSIFIER: Final = "no-such-deployment"
NOT_SERVED_WARNING: Final = f"safeguards classifier {UNKNOWN_CLASSIFIER} is not served by this proxy"


def classify(rig: Rig) -> tuple[dict[str, JsonValue], str]:
    request_nonce: Final = nonce()
    rig.register(request_nonce, chat_backend(tool_id(request_nonce), stream=False))
    return json_body(rig.post(auto_mode_body(CHAT_MODEL, request_nonce))), request_nonce


def assert_reply_unchanged(rig: Rig) -> None:
    body, request_nonce = classify(rig)
    assert "safeguard_results" not in body, body
    assert len(tool_use_ids(body)) == 1, body
    received: Final = rig.received(request_nonce)
    assert rig.classifier_calls(received) == (), [request.target for request in received]
    assert len(rig.backend_calls(received)) == 1, [request.target for request in received]


@pytest.mark.parametrize(
    ("variant", "setting"), (("missing", OMITTED), ("empty", ""), ("null", None)), ids=("missing", "empty", "null")
)
def test_unset_classifier_model_leaves_the_reply_unchanged(
    tmp_path: Path, variant: str, setting: JsonValue | Omitted
) -> None:
    with safeguards_rig(tmp_path, classifier_model=setting) as rig:
        assert_reply_unchanged(rig)


def test_integer_classifier_model_setting_leaves_the_reply_unchanged(tmp_path: Path) -> None:
    with safeguards_rig(tmp_path, classifier_model=7) as rig:
        assert_reply_unchanged(rig)
        assert rig.gateway.client.get("/health/liveliness").status_code == 200


def test_unknown_classifier_model_name_is_unsupported_with_one_warning(tmp_path: Path) -> None:
    with safeguards_rig(tmp_path, classifier_model=UNKNOWN_CLASSIFIER) as rig:
        log_offset: Final = rig.owned.log.stat().st_size
        body, request_nonce = classify(rig)
        assert body["safeguard_results"] == list(UNSUPPORTED), body
        assert len(tool_use_ids(body)) == 1, body
        received: Final = rig.received(request_nonce)
        assert rig.classifier_calls(received) == (), [request.target for request in received]
        tail: Final = eventually(
            lambda: rig.owned.log.read_bytes()[log_offset:].decode(errors="replace"),
            lambda text: NOT_SERVED_WARNING in text,
            seconds=30,
        )
        assert tail.count(NOT_SERVED_WARNING) == 1, tail
        assert "Traceback" not in tail, tail
