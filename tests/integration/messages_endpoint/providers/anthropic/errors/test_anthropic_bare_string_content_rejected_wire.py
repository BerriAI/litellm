import json
import time
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server


@pytest.mark.covers("other.provider_wire.anthropic.bare_string_content_item_is_client_error")
@pytest.mark.parametrize(
    "text", [pytest.param("what type of file is this?", id="type_word"), pytest.param("hello", id="plain")]
)
def test_anthropic_bare_string_content_item_is_rejected_as_client_error_before_the_wire(
    gateway: Gateway, text: str
) -> None:
    def respond(request: Request) -> Reply:
        raise AssertionError(f"upstream must not be reached: {request.target}")

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="anthropic/claude-sonnet-4-5-20250929", api_base=wire.url, api_key="synthetic-anthropic-key"
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "max_tokens": 16, "timeout": 5, "messages": [{"role": "system", "content": [text]}]},
        )
        assert response.status_code == 400, response.text
        assert wire.drain() == ()
