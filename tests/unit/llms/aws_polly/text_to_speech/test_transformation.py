import json
from typing import Final

import pytest

from litellm.llms.aws_polly.text_to_speech.transformation import AWSPollyTextToSpeechConfig


def test_installed_botocore_signs_the_speech_request() -> None:
    headers, body = AWSPollyTextToSpeechConfig()._sign_polly_request(
        request_body={"Text": "ping", "VoiceId": "Joanna"},
        endpoint_url="https://polly.us-west-2.amazonaws.com/v1/speech",
        litellm_params={
            "aws_access_key_id": "test-key",
            "aws_secret_access_key": "test-secret",
            "aws_region_name": "us-west-2",
        },
    )
    authorization: Final = headers["Authorization"]
    assert "/us-west-2/polly/aws4_request" in authorization
    assert json.loads(body) == {"Text": "ping", "VoiceId": "Joanna"}


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("standard", "standard"),
        ("Long-Form", "long-form"),
        ("aws_polly/generative", "generative"),
        ("aws_polly", "neural"),
        ("unknown", "neural"),
    ],
)
def test_extract_engine_from_model(model: str, expected: str) -> None:
    assert AWSPollyTextToSpeechConfig()._extract_engine_from_model(model) == expected
