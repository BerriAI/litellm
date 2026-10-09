import json
from collections.abc import Mapping
from typing import Final

import pytest
from botocore.credentials import Credentials
from pydantic import TypeAdapter

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.decisions.transformation import systemone_request_to_ir
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.strands_decider.decisions.transformation import (
    AGENTCORE_SESSION_HEADER,
    StrandsDeciderDecisionsConfig,
)
from litellm.types.decisions import DecisionsRequestBody
from litellm.types.llms.bedrock import AwsAuthParams

_ARN: Final = "arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/qa_decider-AbC123xyz0"
_BODY: Final[Mapping[str, object]] = {
    "model": "strands-decider-2B-hobson-v21",
    "state": "My order arrived broken.",
    "questions": {"is_complaint": {"type": "noul", "instructions": "Is this a complaint?"}},
}
_IR_REQUEST: Final = systemone_request_to_ir(
    TypeAdapter(DecisionsRequestBody).validate_python({"state": _BODY["state"], "questions": _BODY["questions"]})
)


class _RecordingAWS(BaseAWSLLM):
    def __init__(self) -> None:
        super().__init__()
        self.resolved: tuple[tuple[AwsAuthParams, str | None], ...] = ()

    def resolve_credentials(self, auth_params: AwsAuthParams, aws_region_name: str | None) -> Credentials:
        self.resolved = (*self.resolved, (auth_params, aws_region_name))
        return Credentials(access_key="AKIDEXAMPLE", secret_key="example-secret")


def _sign(
    config: StrandsDeciderDecisionsConfig,
    api_base: str,
    headers: Mapping[str, str],
    api_key: str | None = None,
    litellm_params: Mapping[str, object] | None = None,
) -> tuple[Mapping[str, str], bytes | None]:
    return config.sign_request(
        headers=headers,
        url=config.get_complete_url(api_base=api_base, model="strands-decider-2B-hobson-v21"),
        api_base=api_base,
        body=_BODY,
        api_key=api_key,
        litellm_params=litellm_params or {},
    )


@pytest.mark.parametrize(
    ("arn", "expected_url"),
    [
        (
            _ARN,
            "https://bedrock-agentcore.us-west-2.amazonaws.com/runtimes/"
            "arn%3Aaws%3Abedrock-agentcore%3Aus-west-2%3A123456789012%3Aruntime%2Fqa_decider-AbC123xyz0/invocations",
        ),
        (
            "arn:aws-cn:bedrock-agentcore:cn-north-1:123456789012:runtime/qa_decider-AbC123xyz0",
            "https://bedrock-agentcore.cn-north-1.amazonaws.com.cn/runtimes/"
            "arn%3Aaws-cn%3Abedrock-agentcore%3Acn-north-1%3A123456789012%3Aruntime%2Fqa_decider-AbC123xyz0"
            "/invocations",
        ),
        (
            "arn:aws-us-gov:bedrock-agentcore:us-gov-west-1:123456789012:runtime/qa_decider-AbC123xyz0",
            "https://bedrock-agentcore.us-gov-west-1.amazonaws.com/runtimes/"
            "arn%3Aaws-us-gov%3Abedrock-agentcore%3Aus-gov-west-1%3A123456789012%3Aruntime%2Fqa_decider-AbC123xyz0"
            "/invocations",
        ),
    ],
)
def test_runtime_arn_targets_the_regional_invoke_agent_runtime_url(arn: str, expected_url: str) -> None:
    assert StrandsDeciderDecisionsConfig().get_complete_url(api_base=arn, model="any") == expected_url


@pytest.mark.parametrize(
    "api_base",
    [
        "https://strands.example",
        "arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/qa_decider-AbC123xyz0/runtime-endpoint/DEFAULT",
        "arn:aws:bedrock-agentcore:evil.example/x:123456789012:runtime/qa_decider-AbC123xyz0",
        "arn:aws:bedrock:us-west-2:123456789012:runtime/qa_decider-AbC123xyz0",
    ],
)
def test_other_api_bases_keep_the_plain_systemone_request(api_base: str) -> None:
    aws: Final = _RecordingAWS()
    config: Final = StrandsDeciderDecisionsConfig(aws=aws)
    headers: Final = {"Content-Type": "application/json"}

    assert config.get_complete_url(api_base=api_base, model="any") == f"{api_base}/v1/systemone"
    assert _sign(config, api_base, headers) == (headers, None)
    assert aws.resolved == ()


def test_sign_request_signs_the_body_for_the_arn_region_with_deployment_credentials() -> None:
    aws: Final = _RecordingAWS()
    config: Final = StrandsDeciderDecisionsConfig(aws=aws)

    signed_headers, signed_body = _sign(
        config,
        _ARN,
        {"Content-Type": "application/json"},
        litellm_params={"aws_profile_name": "qa", "aws_role_name": "decider-role", "aws_region_name": "us-east-1"},
    )

    assert signed_body == json.dumps(_BODY).encode()
    assert len(aws.resolved) == 1
    auth_params, region = aws.resolved[0]
    assert (auth_params.aws_profile_name, auth_params.aws_role_name, region) == ("qa", "decider-role", "us-west-2")
    authorization: Final = signed_headers["Authorization"]
    assert authorization.startswith("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/")
    assert "/us-west-2/bedrock-agentcore/aws4_request" in authorization
    assert AGENTCORE_SESSION_HEADER.lower() in authorization


def test_default_session_id_is_stable_per_runtime_and_within_the_agentcore_length_bounds() -> None:
    other_arn: Final = _ARN.replace("AbC123xyz0", "ZyX987cba1")

    first, _ = _sign(StrandsDeciderDecisionsConfig(aws=_RecordingAWS()), _ARN, {})
    again, _ = _sign(StrandsDeciderDecisionsConfig(aws=_RecordingAWS()), _ARN, {})
    other, _ = _sign(StrandsDeciderDecisionsConfig(aws=_RecordingAWS()), other_arn, {})

    session_id: Final = first[AGENTCORE_SESSION_HEADER]
    assert session_id == again[AGENTCORE_SESSION_HEADER]
    assert session_id != other[AGENTCORE_SESSION_HEADER]
    assert 33 <= len(session_id) <= 256


def test_caller_session_id_header_wins_over_the_default() -> None:
    caller_header: Final = AGENTCORE_SESSION_HEADER.lower()

    signed_headers, _ = _sign(
        StrandsDeciderDecisionsConfig(aws=_RecordingAWS()),
        _ARN,
        {caller_header: "caller-session-0123456789abcdef0123456789"},
    )

    session_headers: Final = [name for name in signed_headers if name.lower() == caller_header]
    assert session_headers == [caller_header]
    assert signed_headers[caller_header] == "caller-session-0123456789abcdef0123456789"


def test_api_key_on_a_runtime_arn_sends_the_bearer_token_without_sigv4() -> None:
    aws: Final = _RecordingAWS()

    signed_headers, signed_body = _sign(
        StrandsDeciderDecisionsConfig(aws=aws), _ARN, {"Authorization": "Bearer runtime-jwt"}, api_key="runtime-jwt"
    )

    assert signed_body is None
    assert signed_headers["Authorization"] == "Bearer runtime-jwt"
    assert AGENTCORE_SESSION_HEADER in signed_headers
    assert aws.resolved == ()


@pytest.mark.parametrize(
    ("code", "status_code"),
    [("bad_request", 400), ("invalid_request", 400), ("loading", 503), ("inference_error", 500), ("new_code", 500)],
)
def test_runtime_error_envelope_raises_with_the_mapped_status(code: str, status_code: int) -> None:
    with pytest.raises(BaseLLMException) as raised:
        StrandsDeciderDecisionsConfig().parse_response(
            {"error": {"code": code, "message": "at most 16 questions"}}, _IR_REQUEST
        )

    assert raised.value.status_code == status_code
    assert code in raised.value.message
    assert "at most 16 questions" in raised.value.message
