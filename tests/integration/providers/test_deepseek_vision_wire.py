from typing import Final

import httpx
import pytest
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway, object_value

_VISION_MODEL: Final = "deepseek-v4-flash-vision-exp"
_API_KEY: Final = "synthetic-deepseek-key"
_VISION_CONTENT: Final[JsonValue] = [
    {"type": "text", "text": "what is in this image?"},
    {"type": "image_url", "image_url": {"url": "https://example.com/pic.png"}},
]


@pytest.mark.covers("other.provider_wire.deepseek.vision_image_content_list_reaches_provider")
def test_deepseek_vision_forwards_image_url_content_list_instead_of_collapsing_to_text(gateway: Gateway) -> None:
    with gateway.scenario() as scenario, httpx.Client(base_url=gateway.upstream_url, trust_env=False) as upstream:
        upstream.get("/__observations").raise_for_status()
        model: Final = scenario.model(
            model=f"deepseek/{_VISION_MODEL}",
            api_key=_API_KEY,
            model_info={"mode": "chat", "supports_vision": True},
        )
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": _VISION_CONTENT}]},
        )
        assert response.status_code == 200, response.text
        observations: Final = JSON_OBJECT.validate_json(upstream.get("/__observations").content)["requests"]
        assert isinstance(observations, list)
        assert len(observations) == 1, response.text
        observed: Final = object_value(observations[0])
        assert observed["path"] == "/v1/chat/completions", response.text
        assert observed["authorization"] == f"Bearer {_API_KEY}", response.text
        body: Final = object_value(observed["body"])
        assert body["model"] == _VISION_MODEL, response.text
        assert body["messages"] == [{"role": "user", "content": _VISION_CONTENT}], response.text
