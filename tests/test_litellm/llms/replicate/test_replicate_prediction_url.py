"""Tests for Replicate get_prediction_url same-origin guard."""

import httpx
import pytest

from litellm.llms.replicate.chat.transformation import ReplicateConfig
from litellm.llms.replicate.common_utils import ReplicateError


def _response(urls_get: str | None, request_url: str = "https://api.replicate.com/v1/models/x/predictions"):
    body = {"urls": {"get": urls_get}} if urls_get is not None else {"urls": {}}
    request = httpx.Request("POST", request_url)
    return httpx.Response(200, json=body, request=request)


class TestReplicatePredictionUrlSameOrigin:
    def test_accepts_same_origin_get_url(self):
        url = ReplicateConfig().get_prediction_url(
            _response("https://api.replicate.com/v1/predictions/abc123")
        )
        assert url == "https://api.replicate.com/v1/predictions/abc123"

    def test_rejects_off_origin_get_url(self):
        with pytest.raises(ReplicateError, match="Rejected prediction URL"):
            ReplicateConfig().get_prediction_url(
                _response("https://evil.example/steal-token")
            )

    def test_rejects_http_scheme_mismatch(self):
        with pytest.raises(ReplicateError, match="Rejected prediction URL"):
            ReplicateConfig().get_prediction_url(
                _response("http://api.replicate.com/v1/predictions/abc123")
            )

    def test_missing_get_url_still_400(self):
        with pytest.raises(ReplicateError, match="prediction url is None"):
            ReplicateConfig().get_prediction_url(_response(None))
