import json
from typing import Final

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
