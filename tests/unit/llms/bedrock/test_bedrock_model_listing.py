import hashlib
import hmac
import re
from collections.abc import Callable, Mapping
from functools import reduce
from typing import Final
from urllib.parse import parse_qsl, quote

import httpx
import pytest

from litellm.llms.bedrock.common_utils import BedrockError, BedrockModelInfo
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.types.router import LiteLLM_Params

ACCESS_KEY: Final = "AKIAIOSFODNN7EXAMPLE"
SECRET_KEY: Final = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
REGION: Final = "eu-central-1"
BEARER_TOKEN: Final = "bedrock-api-key-for-tests"
SECOND_PAGE_TOKEN: Final = "page2+token/with=reserved chars"

ON_DEMAND_MODELS: Final = (
    {"modelId": "anthropic.claude-3-haiku-20240307-v1:0", "inferenceTypesSupported": ["ON_DEMAND"]},
    {"modelId": "amazon.nova-micro-v1:0", "inferenceTypesSupported": ["ON_DEMAND"]},
)
PROFILE_ONLY_MODEL: Final = {
    "modelId": "anthropic.claude-opus-4-5-20251101-v1:0",
    "inferenceTypesSupported": ["INFERENCE_PROFILE"],
}
PROFILE_PAGES: Final = {
    None: {
        "inferenceProfileSummaries": [
            {"inferenceProfileId": "global.anthropic.claude-opus-4-5-20251101-v1:0", "status": "ACTIVE"},
            {"inferenceProfileId": "eu.anthropic.claude-3-haiku-20240307-v1:0", "status": "INACTIVE"},
        ],
        "nextToken": SECOND_PAGE_TOKEN,
    },
    SECOND_PAGE_TOKEN: {
        "inferenceProfileSummaries": [{"inferenceProfileId": "eu.amazon.nova-micro-v1:0", "status": "ACTIVE"}],
    },
}
EXPECTED_MODELS: Final = [
    "amazon.nova-micro-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
    "eu.amazon.nova-micro-v1:0",
    "global.anthropic.claude-opus-4-5-20251101-v1:0",
]
SIGV4_AUTHORIZATION: Final = re.compile(
    rf"^AWS4-HMAC-SHA256 Credential={ACCESS_KEY}/\d{{8}}/{REGION}/bedrock/aws4_request, "
    r"SignedHeaders=([^,]+), Signature=([0-9a-f]{64})$"
)


@pytest.fixture(autouse=True)
def no_ambient_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)


def _sigv4_signature_as_aws_computes_it(request: httpx.Request, signed_headers: tuple[str, ...]) -> str:
    amz_date: Final = request.headers["x-amz-date"]
    canonical_query: Final = "&".join(
        f"{quote(name, safe='-_.~')}={quote(value, safe='-_.~')}"
        for name, value in sorted(parse_qsl(request.url.query.decode(), keep_blank_values=True))
    )
    canonical_headers: Final = "".join(f"{name}:{request.headers[name].strip()}\n" for name in signed_headers)
    canonical_request: Final = "\n".join(
        (
            request.method,
            request.url.path,
            canonical_query,
            canonical_headers,
            ";".join(signed_headers),
            hashlib.sha256(request.content).hexdigest(),
        )
    )
    scope: Final = f"{amz_date[:8]}/{REGION}/bedrock/aws4_request"
    string_to_sign: Final = "\n".join(
        ("AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical_request.encode()).hexdigest())
    )
    signing_key: Final = reduce(
        lambda key, part: hmac.new(key, part.encode(), hashlib.sha256).digest(),
        (amz_date[:8], REGION, "bedrock", "aws4_request"),
        f"AWS4{SECRET_KEY}".encode(),
    )
    return hmac.new(signing_key, string_to_sign.encode(), hashlib.sha256).hexdigest()


def _authorized(request: httpx.Request, bearer_token: str | None) -> bool:
    authorization: Final = request.headers.get("authorization", "")
    if bearer_token is not None:
        return authorization == f"Bearer {bearer_token}"
    signed: Final = SIGV4_AUTHORIZATION.match(authorization)
    if signed is None or "x-amz-date" not in request.headers:
        return False
    signed_headers: Final = tuple(signed.group(1).split(";"))
    if "host" not in signed_headers or "x-amz-date" not in signed_headers:
        return False
    return signed.group(2) == _sigv4_signature_as_aws_computes_it(request, signed_headers)


def _control_plane(bearer_token: str | None = None) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host != f"bedrock.{REGION}.amazonaws.com":
            return httpx.Response(404, json={"message": f"no Bedrock at {request.url.host}"})
        if not _authorized(request, bearer_token):
            return httpx.Response(403, json={"message": "not signed for this account, region and service"})
        if request.url.path == "/foundation-models":
            on_demand_only: Final = request.url.params.get("byInferenceType") == "ON_DEMAND"
            summaries: Final = ON_DEMAND_MODELS if on_demand_only else (*ON_DEMAND_MODELS, PROFILE_ONLY_MODEL)
            return httpx.Response(200, json={"modelSummaries": list(summaries)})
        if request.url.path == "/inference-profiles":
            if request.url.params.get("typeEquals") != "SYSTEM_DEFINED":
                return httpx.Response(400, json={"message": "application profiles are not listable here"})
            return httpx.Response(200, json=PROFILE_PAGES[request.url.params.get("nextToken")])
        return httpx.Response(404, json={"message": f"unknown path {request.url.path}"})

    return handler


def _bedrock(handler: Callable[[httpx.Request], httpx.Response]) -> BedrockModelInfo:
    return BedrockModelInfo(client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(handler))))


def _deployment(region: str = REGION, api_key: str | None = None) -> Mapping[str, object]:
    return LiteLLM_Params(
        model="bedrock/*",
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        aws_region_name=region,
        api_key=api_key,
    ).model_dump(exclude_none=True)


def test_lists_active_profiles_and_on_demand_models_signed_for_the_deployment() -> None:
    assert _bedrock(_control_plane()).discover_models(_deployment()) == EXPECTED_MODELS


def test_lists_in_the_deployment_region_not_the_ambient_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_REGION_NAME", REGION)

    with pytest.raises(BedrockError, match=r"us-west-2.*404"):
        _bedrock(_control_plane()).discover_models(_deployment(region="us-west-2"))


def test_bearer_token_api_key_replaces_sigv4() -> None:
    bedrock: Final = _bedrock(_control_plane(bearer_token=BEARER_TOKEN))

    assert bedrock.discover_models(_deployment(api_key=BEARER_TOKEN)) == EXPECTED_MODELS
    with pytest.raises(BedrockError, match="403"):
        bedrock.discover_models(_deployment())


def test_get_models_without_a_deployment_uses_ambient_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", ACCESS_KEY)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", SECRET_KEY)
    monkeypatch.setenv("AWS_REGION_NAME", REGION)

    assert _bedrock(_control_plane()).get_models() == EXPECTED_MODELS


def test_listing_failure_names_the_region_and_response() -> None:
    def throttled(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"message": "Too many requests"})

    with pytest.raises(BedrockError, match=rf"{REGION}.*429.*Too many requests"):
        _bedrock(throttled).discover_models(_deployment())
