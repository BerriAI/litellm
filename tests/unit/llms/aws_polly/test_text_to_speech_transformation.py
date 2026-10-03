import json
import sys
from typing import Final

import pytest

from litellm.llms.aws_polly.text_to_speech.transformation import AWSPollyTextToSpeechConfig


def test_polly_signing_preserves_payload_with_installed_aws_extra() -> None:
    payload: Final[dict[str, object]] = {"Text": "héllo", "VoiceId": "Joanna"}
    headers, body = AWSPollyTextToSpeechConfig()._sign_polly_request(
        payload, "https://polly.us-east-1.amazonaws.com/v1/speech",
        {"aws_access_key_id": "test-key", "aws_secret_access_key": "test-secret",
         "aws_session_token": "test-session", "aws_region_name": "us-east-1"},
    )
    assert headers["Authorization"].startswith("AWS4-HMAC-SHA256 Credential=test-key/")
    assert "/us-east-1/polly/aws4_request" in headers["Authorization"]
    assert headers["X-Amz-Security-Token"] == "test-session"
    assert json.loads(body) == payload


@pytest.mark.parametrize("missing", ("boto3", "botocore"))
def test_polly_signing_without_aws_extra_explains_installation(missing: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, missing, None)
    with pytest.raises(ImportError, match=r'Install AWS support with pip install "litellm\[aws\]"') as caught:
        AWSPollyTextToSpeechConfig()._sign_polly_request(
            {"Text": "hello", "VoiceId": "Joanna"}, "https://polly.us-east-1.amazonaws.com/v1/speech", {},
        )
    assert isinstance(caught.value.__cause__, ModuleNotFoundError)
    assert caught.value.__cause__.name == missing
