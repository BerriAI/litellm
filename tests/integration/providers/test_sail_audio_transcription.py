from typing import Final

import httpx
import pytest

from tests.integration._support.client import JSON_OBJECT, Gateway

WAV_HEADER: Final = (
    b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00"
    b"\x40\x1f\x00\x00\x80\x3e\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
)


def test_audio_transcription_on_a_sail_deployment_is_rejected_before_the_upstream_sees_it(gateway: Gateway) -> None:
    with gateway.scenario() as scenario, httpx.Client(base_url=gateway.upstream_url, trust_env=False) as upstream:
        upstream.get("/__observations").raise_for_status()
        model: Final = scenario.model(model="sail/x")
        response: Final = gateway.request_multipart(
            "/v1/audio/transcriptions", {"model": model}, {"file": ("control.wav", WAV_HEADER, "audio/wav")}
        )
        # The status code for an unsupported transcription provider is generic
        # LiteLLM behavior tracked in LIT-8650 (500 today, 400 once it lands);
        # what Sail owns is that the request is rejected before the upstream sees it.
        assert response.status_code >= 400, response.text
        assert "error" in JSON_OBJECT.validate_json(response.content), response.text
        observations: Final = JSON_OBJECT.validate_json(upstream.get("/__observations").content)["requests"]
        assert observations == [], observations
