import json
from typing import Final

from litellm.constants import SERVER_STREAMING_CLASSIFICATION_KEY, SERVER_STREAMING_CLASSIFICATION_MARKER
from litellm.llms.pass_through.guardrail_translation.handler import PassThroughEndpointHandler


def test_full_payload_guardrail_text_excludes_the_server_streaming_marker():
    body: Final = {"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]}

    text: Final = PassThroughEndpointHandler()._extract_text_for_guardrail(
        {**body, SERVER_STREAMING_CLASSIFICATION_KEY: SERVER_STREAMING_CLASSIFICATION_MARKER},
        None,
    )

    assert json.loads(text) == body, text
