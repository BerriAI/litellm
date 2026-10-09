import json
from typing import Final

import pytest

from litellm.constants import SERVER_STREAMING_CLASSIFICATION_KEY, SERVER_STREAMING_CLASSIFICATION_MARKER
from litellm.llms.pass_through.guardrail_translation.handler import PassThroughEndpointHandler


def test_full_payload_guardrail_text_excludes_the_server_streaming_marker():
    body: Final = {"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]}

    text: Final = PassThroughEndpointHandler()._extract_text_for_guardrail(
        {**body, SERVER_STREAMING_CLASSIFICATION_KEY: SERVER_STREAMING_CLASSIFICATION_MARKER},
        None,
    )

    assert json.loads(text) == body, text


@pytest.mark.parametrize(
    "marker",
    [
        SERVER_STREAMING_CLASSIFICATION_MARKER,
        json.loads(json.dumps(SERVER_STREAMING_CLASSIFICATION_MARKER)),
    ],
    ids=["enum", "json-string"],
)
def test_full_payload_guardrail_text_scans_caller_value_but_not_the_marker(marker: str):
    text: Final = PassThroughEndpointHandler()._extract_text_for_guardrail(
        {
            "model": "m",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
            SERVER_STREAMING_CLASSIFICATION_KEY: "BLOCKME caller content",
        },
        None,
    )

    assert "BLOCKME caller content" in text, text

    marker_text: Final = PassThroughEndpointHandler()._extract_text_for_guardrail(
        {
            "model": "m",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
            SERVER_STREAMING_CLASSIFICATION_KEY: marker,
        },
        None,
    )

    assert SERVER_STREAMING_CLASSIFICATION_KEY not in json.loads(marker_text), marker_text
