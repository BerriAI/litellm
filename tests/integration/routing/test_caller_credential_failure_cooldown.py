"""A provider rejecting a credential one caller sent in the request body must not cool down the shared deployment.

The scripted provider answers a request that presents a credential value (an OpenAI or Azure key, an
``OpenAI-Organization`` header, an Azure AD bearer, a SigV4 access key id, a Google OAuth token) containing
``rejected`` with a 401, ``throttled`` with a 429 and ``broken`` with a 500. It answers an Azure ``api-version``
it does not serve, or an OpenAI model it does not have, with a 404. Everything else is served, so the operator's
own credentials keep working and the only thing that can stop a plain caller from being served is the router's
cooldown."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, assert_never
from urllib.parse import parse_qs, urlsplit

import httpx
import jwt
import pytest
import yaml
from integration._support.client import Gateway, Scenario, gateway_from_environment
from integration._support.process import owned_proxy
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

PROXY_CONFIG: Final = Path(__file__).resolve().parents[1] / "proxy_config.yaml"
COOLDOWN_SECONDS: Final = 300
FAILURE_MARKERS: Final = (("rejected", 401), ("throttled", 429), ("broken", 500))
SERVED: Final = "served by the shared deployment"
NO_DEPLOYMENTS: Final = "No deployments available"
SERVED_API_VERSION: Final = "2025-04-01-preview"
CALLER_API_VERSION: Final = "2024-10-21"
SERVED_API_VERSIONS: Final = frozenset({SERVED_API_VERSION, CALLER_API_VERSION})
RETIRED_API_VERSION: Final = "2023-03-15-preview"
MISSING_MODEL: Final = "openai/gpt-5.6-retired"
FAILING_ATTEMPTS: Final = 20
FALLBACK_FIELDS: Final = ("organization", "azure_ad_token", "api_version")
BEDROCK_ERRORS: Final = MappingProxyType(
    {
        401: ("The security token included in the request is invalid.", "UnrecognizedClientException"),
        429: ("Too many requests, please wait before trying again.", "ThrottlingException"),
        500: ("The server encountered an internal error.", "InternalServerException"),
    }
)
OPERATOR_KEY: Final = "sk-operator-key"
OPERATOR_ORGANIZATION: Final = "org-operator"
OPERATOR_ACCESS_KEY: Final = "AKIAOPERATORSTATIC"
OPERATOR_SECRET: Final = "operator-static-secret"
OPERATOR_PROJECT: Final = "operator-project"
CALLER_PROJECT: Final = "rejected-caller-project"
REJECTED_PROFILE: Final = "rejected-caller-profile"
AWS_REGION: Final = "us-east-1"
OPT_IN_FIELDS: Final = frozenset({"api_key", "azure_ad_token", "aws_profile_name", "vertex_credentials"})


Endpoint = Literal["chat", "messages"]


def primary_group(field: str) -> str:
    return f"caller-cooldown-primary-{field}"


def secondary_group(field: str) -> str:
    return f"caller-cooldown-secondary-{field}"


class Backend(Enum):
    OPENAI = "openai"
    AZURE = "azure"
    BEDROCK = "bedrock"
    VERTEX = "vertex"


def _json_reply(status: int, payload: Mapping[str, JsonValue], headers: Mapping[str, str] | None = None) -> Reply:
    return Reply(status=status, body=json.dumps(payload).encode(), headers=headers or {})


def _jwt_issuer(token: str) -> str | None:
    try:
        claims: Final = jwt.decode(token, options={"verify_signature": False})
    except jwt.InvalidTokenError:
        return None
    issuer: Final = claims.get("iss")
    return issuer if isinstance(issuer, str) else None


def _scripted_failure(request: Request) -> int | None:
    issuer: Final = _jwt_issuer(request.headers.get("authorization", "").removeprefix("Bearer "))
    presented: Final = tuple(
        value.lower() for value in (*request.headers.values(), *((issuer,) if issuer is not None else ()))
    )
    return next(
        (status for marker, status in FAILURE_MARKERS if any(marker in value for value in presented)),
        None,
    )


def _token_reply(request: Request) -> Reply:
    assertion: Final = parse_qs(request.body.decode())["assertion"][0]
    return _json_reply(
        200, {"access_token": f"minted-for-{_jwt_issuer(assertion)}", "token_type": "Bearer", "expires_in": 3600}
    )


def _chat_reply() -> Reply:
    return _json_reply(
        200,
        {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-5.6",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": SERVED}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
        },
    )


def _openai_error(status: int) -> Reply:
    return _json_reply(
        status, {"error": {"message": f"scripted {status}", "type": "scripted_error", "code": f"scripted_{status}"}}
    )


def _openai(request: Request) -> Reply:
    failure: Final = _scripted_failure(request)
    if failure is not None:
        return _openai_error(failure)
    if urlsplit(request.target).path != "/v1/chat/completions":
        return _json_reply(501, {"error": {"message": f"unscripted route {request.method} {request.target}"}})
    if json.loads(request.body).get("model") == MISSING_MODEL.removeprefix("openai/"):
        return _json_reply(
            404, {"error": {"message": "The model does not exist", "type": "invalid_request_error", "code": None}}
        )
    return _chat_reply()


def _azure(request: Request) -> Reply:
    versions: Final = parse_qs(urlsplit(request.target).query).get("api-version", [])
    if len(versions) != 1 or versions[0] not in SERVED_API_VERSIONS:
        return _json_reply(404, {"error": {"code": "404", "message": "Resource not found"}})
    failure: Final = _scripted_failure(request)
    return _chat_reply() if failure is None else _openai_error(failure)


def _bedrock(request: Request) -> Reply:
    failure: Final = _scripted_failure(request)
    if failure is not None:
        message, error_type = BEDROCK_ERRORS[failure]
        return _json_reply(failure, {"message": message}, {"x-amzn-errortype": error_type})
    return _json_reply(
        200,
        {
            "output": {"message": {"role": "assistant", "content": [{"text": SERVED}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 9, "outputTokens": 4, "totalTokens": 13},
            "metrics": {"latencyMs": 1},
        },
    )


def _vertex(request: Request) -> Reply:
    failure: Final = _scripted_failure(request)
    if failure is not None:
        return _json_reply(
            failure, {"error": {"code": failure, "message": f"scripted {failure}", "status": "UNAUTHENTICATED"}}
        )
    return _json_reply(
        200,
        {
            "candidates": [{"content": {"role": "model", "parts": [{"text": SERVED}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 4, "totalTokenCount": 13},
            "modelVersion": "gemini-3.8-flash",
        },
    )


def scripted_provider(request: Request) -> Reply:
    path: Final = urlsplit(request.target).path
    if path == "/_oauth/token":
        return _token_reply(request)
    if path.startswith("/model/"):
        return _bedrock(request)
    if path.endswith(":generateContent"):
        return _vertex(request)
    if path.startswith("/openai/deployments/"):
        return _azure(request)
    if path.startswith("/v1/"):
        return _openai(request)
    return _json_reply(501, {"error": {"message": f"unscripted route {request.method} {request.target}"}})


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    wire: Wire
    per_deployment_opt_in: bool
    operator_vertex_credentials: str
    caller_vertex_credentials: str

    def params(self, backend: Backend) -> Mapping[str, JsonValue]:
        match backend:
            case Backend.OPENAI:
                return {
                    "model": "openai/gpt-5.6",
                    "api_base": f"{self.wire.url}/v1",
                    "api_key": OPERATOR_KEY,
                    "organization": OPERATOR_ORGANIZATION,
                }
            case Backend.AZURE:
                return {
                    "model": "azure/gpt-5.6",
                    "api_base": self.wire.url,
                    "api_key": OPERATOR_KEY,
                    "api_version": SERVED_API_VERSION,
                }
            case Backend.BEDROCK:
                return {
                    "model": "bedrock/converse/anthropic.claude-sonnet-5",
                    "api_base": self.wire.url,
                    "api_key": None,
                    "aws_access_key_id": OPERATOR_ACCESS_KEY,
                    "aws_secret_access_key": OPERATOR_SECRET,
                    "aws_region_name": AWS_REGION,
                }
            case Backend.VERTEX:
                return {
                    "model": "vertex_ai/gemini-3.8-flash",
                    "api_base": self.wire.url,
                    "api_key": None,
                    "vertex_project": OPERATOR_PROJECT,
                    "vertex_location": "us-central1",
                    "vertex_credentials": self.operator_vertex_credentials,
                }
            case _:
                assert_never(backend)

    def deployment(
        self,
        scenario: Scenario,
        backend: Backend,
        caller_fields: Iterable[str] = (),
        *,
        name: str | None = None,
        **overrides: JsonValue,
    ) -> str:
        opt_ins: Final = sorted(OPT_IN_FIELDS.intersection(caller_fields)) if self.per_deployment_opt_in else []
        opt_in_params: Final[Mapping[str, JsonValue]] = (
            {"configurable_clientside_auth_params": list(opt_ins)} if opt_ins else {}
        )
        return scenario.model(name=name, **{**self.params(backend), **overrides, **opt_in_params})

    def chat(
        self,
        model: str,
        key: str,
        *,
        prompt: str | None = None,
        query: Mapping[str, str] | None = None,
        **fields: JsonValue,
    ) -> httpx.Response:
        content: Final = prompt if prompt is not None else f"question {uuid.uuid4().hex}"
        return self.gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": content}], **fields},
            key=key,
            params=query,
        )

    def stream(self, endpoint: Endpoint, model: str, key: str, prompt: str, **fields: JsonValue) -> httpx.Response:
        messages: Final[list[JsonValue]] = [{"role": "user", "content": prompt}]
        match endpoint:
            case "chat":
                return self.gateway.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": model, "messages": messages, "stream": True, **fields},
                    key=key,
                )
            case "messages":
                return self.gateway.request(
                    "POST",
                    "/v1/messages",
                    {"model": model, "max_tokens": 16, "messages": messages, "stream": True, **fields},
                    key=key,
                )
            case _:
                assert_never(endpoint)

    def upstream_calls(self, prompt: str) -> int:
        return sum(1 for request in self.wire.drain() if prompt.encode() in request.body)


def assert_served(response: httpx.Response) -> None:
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == SERVED, response.text


def assert_stream_failed(response: httpx.Response, status: int) -> None:
    assert SERVED not in response.text, response.text
    assert response.status_code == status or (response.status_code == 200 and "error" in response.text), (
        response.status_code,
        response.text,
    )


def assert_cooled_down(response: httpx.Response) -> None:
    assert response.status_code == 429, response.text
    assert NO_DEPLOYMENTS in response.text, response.text


def _proxy_config(directory: Path, *, proxy_wide_opt_in: bool) -> Path:
    loaded: Final = yaml.safe_load(PROXY_CONFIG.read_text())
    general_settings: Final = {
        **loaded["general_settings"],
        **({"allow_client_side_credentials": True} if proxy_wide_opt_in else {}),
    }
    path: Final = directory / "caller-credential-cooldown.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                **loaded,
                "general_settings": general_settings,
                "router_settings": {
                    "num_retries": 0,
                    "cooldown_time": COOLDOWN_SECONDS,
                    "fallbacks": [{primary_group(field): [secondary_group(field)]} for field in FALLBACK_FIELDS],
                },
            }
        )
    )
    return path


def _aws_files(directory: Path) -> Mapping[str, str]:
    credentials: Final = directory / "aws-credentials"
    credentials.write_text(
        f"[{REJECTED_PROFILE}]\naws_access_key_id = AKIAREJECTEDPROFILE\naws_secret_access_key = rejected-profile-secret\n"
    )
    config: Final = directory / "aws-config"
    config.write_text(f"[profile {REJECTED_PROFILE}]\nregion = {AWS_REGION}\n")
    return {
        "AWS_SHARED_CREDENTIALS_FILE": str(credentials),
        "AWS_CONFIG_FILE": str(config),
        "AWS_EC2_METADATA_DISABLED": "true",
    }


def _booted_rig(directory: Path, *, proxy_wide_opt_in: bool) -> Iterator[Rig]:
    with ExitStack() as stack:
        root: Final = stack.enter_context(gateway_from_environment())
        wire: Final = stack.enter_context(wire_server(scripted_provider))
        config: Final = _proxy_config(directory, proxy_wide_opt_in=proxy_wide_opt_in)
        gateway: Final = stack.enter_context(owned_proxy(root, directory, _aws_files(directory), config=config))
        yield Rig(
            gateway=gateway,
            wire=wire,
            per_deployment_opt_in=not proxy_wide_opt_in,
            operator_vertex_credentials=service_account_json(OPERATOR_PROJECT, wire.url),
            caller_vertex_credentials=service_account_json(CALLER_PROJECT, wire.url),
        )


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    yield from _booted_rig(tmp_path_factory.mktemp("caller-credential-cooldown"), proxy_wide_opt_in=False)


@pytest.fixture(scope="module")
def proxy_wide_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    yield from _booted_rig(tmp_path_factory.mktemp("caller-credential-cooldown-proxy-wide"), proxy_wide_opt_in=True)


@dataclass(frozen=True, slots=True)
class CallerCredential:
    backend: Backend
    fields: Callable[[Rig], Mapping[str, JsonValue]]
    status: int


@dataclass(frozen=True, slots=True)
class Rejection:
    backend: Backend
    values: Mapping[str, JsonValue]
    status: int


AZURE_AD_TOKEN_CALLER: Final = CallerCredential(
    Backend.AZURE, lambda _: {"azure_ad_token": "rejected-caller-ad-token"}, 401
)
ORGANIZATION_CALLER: Final = CallerCredential(Backend.OPENAI, lambda _: {"organization": "org-rejected-caller"}, 401)
API_VERSION_CALLER: Final = CallerCredential(Backend.AZURE, lambda _: {"api_version": RETIRED_API_VERSION}, 404)
STATIC_KEYS_CALLER: Final = CallerCredential(
    Backend.BEDROCK,
    lambda _: {"aws_access_key_id": "AKIAREJECTEDCALLER", "aws_secret_access_key": "rejected-caller-secret"},
    401,
)

CALLER_CREDENTIALS: Final = (
    pytest.param(AZURE_AD_TOKEN_CALLER, id="azure-azure_ad_token"),
    pytest.param(ORGANIZATION_CALLER, id="openai-organization"),
    pytest.param(API_VERSION_CALLER, id="azure-api_version"),
    pytest.param(STATIC_KEYS_CALLER, id="bedrock-static-keys"),
    pytest.param(
        CallerCredential(Backend.BEDROCK, lambda _: {"aws_profile_name": REJECTED_PROFILE}, 401),
        id="bedrock-aws_profile_name",
    ),
    pytest.param(
        CallerCredential(Backend.VERTEX, lambda rig: {"vertex_credentials": rig.caller_vertex_credentials}, 401),
        id="vertex-vertex_credentials",
    ),
)

FALLBACK_CREDENTIALS: Final = MappingProxyType(
    {"organization": ORGANIZATION_CALLER, "azure_ad_token": AZURE_AD_TOKEN_CALLER, "api_version": API_VERSION_CALLER}
)

STREAMED_CREDENTIALS: Final = (
    pytest.param("chat", ORGANIZATION_CALLER, id="chat-openai-organization"),
    pytest.param("chat", AZURE_AD_TOKEN_CALLER, id="chat-azure-azure_ad_token"),
    pytest.param("chat", API_VERSION_CALLER, id="chat-azure-api_version"),
    pytest.param("chat", STATIC_KEYS_CALLER, id="chat-bedrock-static-keys"),
    pytest.param("messages", STATIC_KEYS_CALLER, id="messages-bedrock-static-keys"),
)

DEPLOYMENT_REJECTIONS: Final = (
    pytest.param(Rejection(Backend.OPENAI, {"api_key": "sk-rejected-operator"}, 401), id="openai-api_key"),
    pytest.param(Rejection(Backend.OPENAI, {"organization": "org-rejected-operator"}, 401), id="openai-organization"),
    pytest.param(Rejection(Backend.AZURE, {"api_version": RETIRED_API_VERSION}, 404), id="azure-api_version"),
    pytest.param(
        Rejection(
            Backend.BEDROCK,
            {"aws_access_key_id": "AKIAREJECTEDOPERATOR", "aws_secret_access_key": "rejected-operator-secret"},
            401,
        ),
        id="bedrock-static-keys",
    ),
)

SHARED_REJECTIONS: Final = (
    pytest.param(Rejection(Backend.OPENAI, {"organization": "org-rejected-shared"}, 401), id="openai-organization"),
    pytest.param(Rejection(Backend.AZURE, {"api_version": RETIRED_API_VERSION}, 404), id="azure-api_version"),
    pytest.param(
        Rejection(Backend.AZURE, {"azure_ad_token": "rejected-shared-ad-token"}, 401), id="azure-azure_ad_token"
    ),
    pytest.param(
        Rejection(
            Backend.BEDROCK,
            {"aws_access_key_id": "AKIAREJECTEDSHARED", "aws_secret_access_key": "rejected-shared-secret"},
            401,
        ),
        id="bedrock-static-keys",
    ),
)


def _assert_one_callers_rejection_leaves_the_deployment_serving(rig: Rig, case: CallerCredential) -> None:
    fields: Final = case.fields(rig)
    with rig.gateway.scenario() as scenario:
        model: Final = rig.deployment(scenario, case.backend, fields)
        rejected_caller: Final = scenario.key()
        other_caller: Final = scenario.key()
        assert_served(rig.chat(model, other_caller))
        rejected: Final = rig.chat(model, rejected_caller, **fields)
        assert rejected.status_code == case.status, rejected.text
        assert_served(rig.chat(model, other_caller))


@pytest.mark.parametrize("case", CALLER_CREDENTIALS)
def test_one_callers_rejected_credential_leaves_the_deployment_serving_other_callers(
    rig: Rig, case: CallerCredential
) -> None:
    """regression: a caller's own bad credential, allowed with no opt-in or by the deployment's
    configurable_clientside_auth_params, fails only that caller, and the next other caller is served instead of
    getting 429 from a cooled down deployment."""
    _assert_one_callers_rejection_leaves_the_deployment_serving(rig, case)


@pytest.mark.parametrize("case", CALLER_CREDENTIALS)
def test_one_callers_rejected_credential_leaves_the_deployment_serving_other_callers_with_proxy_wide_opt_in(
    proxy_wide_rig: Rig, case: CallerCredential
) -> None:
    """regression: with general_settings.allow_client_side_credentials on, a caller's own bad credential fails only
    that caller, and the next other caller is served instead of getting 429 from a cooled down deployment."""
    _assert_one_callers_rejection_leaves_the_deployment_serving(proxy_wide_rig, case)


@pytest.mark.parametrize("rejection", DEPLOYMENT_REJECTIONS)
def test_a_rejected_deployment_credential_still_cools_the_deployment_down(rig: Rig, rejection: Rejection) -> None:
    """guard: when the provider rejects the deployment's own configured credential, the next caller gets 429."""
    with rig.gateway.scenario() as scenario:
        model: Final = rig.deployment(scenario, rejection.backend, **rejection.values)
        first: Final = rig.chat(model, scenario.key())
        assert first.status_code == rejection.status, first.text
        assert_cooled_down(rig.chat(model, scenario.key()))


@pytest.mark.parametrize("rejection", SHARED_REJECTIONS)
def test_a_caller_resending_the_deployments_own_rejected_value_still_cools_it_down(
    rig: Rig, rejection: Rejection
) -> None:
    """guard: a caller sending the exact value the deployment already has gets the rejection, and since the
    deployment's own credential is what failed, the next other caller gets 429."""
    with rig.gateway.scenario() as scenario:
        model: Final = rig.deployment(scenario, rejection.backend, rejection.values, **rejection.values)
        echoed: Final = rig.chat(model, scenario.key(), **rejection.values)
        assert echoed.status_code == rejection.status, echoed.text
        assert_cooled_down(rig.chat(model, scenario.key()))


def test_a_credential_field_the_provider_ignores_does_not_spare_a_rejected_deployment(rig: Rig) -> None:
    """guard: a caller sending an AWS region to an OpenAI deployment whose own key is rejected still cools the
    deployment down, so the next other caller gets 429."""
    fields: Final[Mapping[str, JsonValue]] = {"aws_region_name": "us-west-2"}
    with rig.gateway.scenario() as scenario:
        model: Final = rig.deployment(scenario, Backend.OPENAI, fields, api_key="sk-rejected-operator")
        rejected: Final = rig.chat(model, scenario.key(), **fields)
        assert rejected.status_code == 401, rejected.text
        assert_cooled_down(rig.chat(model, scenario.key()))


def test_a_callers_rejected_api_key_fails_only_that_caller(rig: Rig) -> None:
    """guard: a caller's own rejected api_key keeps failing that caller with 401 while other callers stay served."""
    fields: Final[Mapping[str, JsonValue]] = {"api_key": "sk-rejected-caller"}
    with rig.gateway.scenario() as scenario:
        model: Final = rig.deployment(scenario, Backend.OPENAI, fields)
        rejected_caller: Final = scenario.key()
        other_caller: Final = scenario.key()
        assert_served(rig.chat(model, other_caller))
        first: Final = rig.chat(model, rejected_caller, **fields)
        assert first.status_code == 401, first.text
        assert_served(rig.chat(model, other_caller))
        again: Final = rig.chat(model, rejected_caller, **fields)
        assert again.status_code == 401, again.text


@pytest.mark.parametrize("field", FALLBACK_FIELDS)
def test_a_callers_rejected_credential_through_a_fallback_leaves_both_groups_serving(rig: Rig, field: str) -> None:
    """regression: a caller's own bad field fails on the primary group and again on its configured fallback group,
    the caller gets the error, and the next plain request to either group is served instead of getting 429."""
    case: Final = FALLBACK_CREDENTIALS[field]
    fields: Final = case.fields(rig)
    prompt: Final = f"fallback question {uuid.uuid4().hex}"
    with rig.gateway.scenario() as scenario:
        primary: Final = rig.deployment(scenario, case.backend, fields, name=primary_group(field))
        secondary: Final = rig.deployment(scenario, case.backend, fields, name=secondary_group(field))
        rejected: Final = rig.chat(primary, scenario.key(), prompt=prompt, **fields)
        assert rejected.status_code == case.status, rejected.text
        assert rig.upstream_calls(prompt) == 2, "the caller's request did not reach both the primary and the fallback"
        other_caller: Final = scenario.key()
        assert_served(rig.chat(secondary, other_caller))
        assert_served(rig.chat(primary, other_caller))


@pytest.mark.parametrize(("endpoint", "case"), STREAMED_CREDENTIALS)
def test_a_callers_rejected_credential_on_a_stream_leaves_the_deployment_serving_other_callers(
    rig: Rig, endpoint: Endpoint, case: CallerCredential
) -> None:
    """regression: a streaming request carrying a caller's own bad field fails only that caller, and the next plain
    caller is served instead of getting 429 from a cooled down deployment."""
    fields: Final = case.fields(rig)
    prompt: Final = f"stream question {uuid.uuid4().hex}"
    with rig.gateway.scenario() as scenario:
        model: Final = rig.deployment(scenario, case.backend, fields)
        other_caller: Final = scenario.key()
        assert_served(rig.chat(model, other_caller))
        assert_stream_failed(rig.stream(endpoint, model, scenario.key(), prompt, **fields), case.status)
        assert rig.upstream_calls(prompt) >= 1, "the caller's stream never reached the provider"
        assert_served(rig.chat(model, other_caller))


def test_a_callers_own_throttled_aws_keys_leave_the_deployment_serving_other_callers(rig: Rig) -> None:
    """regression: a caller's own AWS keys answered with 429 by their own account fail only that caller, and the next
    plain caller is served instead of getting 429. The group has two deployments because a lone deployment is never
    cooled down for a 429, so the caller is throttled once per deployment."""
    fields: Final[Mapping[str, JsonValue]] = {
        "aws_access_key_id": "AKIATHROTTLEDCALLER",
        "aws_secret_access_key": "throttled-caller-secret",
    }
    group: Final = f"integration-{uuid.uuid4().hex}"
    with rig.gateway.scenario() as scenario:
        for index in range(2):
            rig.deployment(
                scenario, Backend.BEDROCK, fields, name=group, aws_access_key_id=f"{OPERATOR_ACCESS_KEY}{index}"
            )
        other_caller: Final = scenario.key()
        throttled_caller: Final = scenario.key()
        assert_served(rig.chat(group, other_caller))
        throttled: Final = tuple(rig.chat(group, throttled_caller, **fields) for _ in range(2))
        assert [response.status_code for response in throttled] == [429, 429], [r.text for r in throttled]
        assert all(NO_DEPLOYMENTS not in response.text for response in throttled), [r.text for r in throttled]
        assert_served(rig.chat(group, other_caller))


@pytest.mark.parametrize("delivery", ("body", "query"))
def test_a_routine_api_version_does_not_spare_a_deployment_whose_own_key_is_rejected(rig: Rig, delivery: str) -> None:
    """guard: a caller sending a served api_version that differs from the deployment's, in the body or as the
    api-version query parameter the Azure SDK adds, still cools down a deployment whose own key gets a 401."""
    with rig.gateway.scenario() as scenario:
        model: Final = rig.deployment(scenario, Backend.AZURE, ("api_version",), api_key="sk-rejected-operator")
        caller: Final = scenario.key()
        first: Final = (
            rig.chat(model, caller, api_version=CALLER_API_VERSION)
            if delivery == "body"
            else rig.chat(model, caller, query={"api-version": CALLER_API_VERSION})
        )
        assert first.status_code == 401, first.text
        assert_cooled_down(rig.chat(model, scenario.key()))


def _plain_caller_after_failures(
    rig: Rig, scenario: Scenario, operator_key: str, fields: Mapping[str, JsonValue]
) -> tuple[int, bool]:
    group: Final = f"integration-{uuid.uuid4().hex}"
    for index in range(2):
        rig.deployment(scenario, Backend.AZURE, fields, name=group, api_key=f"{operator_key}-{index}", max_retries=0)
    caller: Final = scenario.key()
    statuses: Final = tuple(rig.chat(group, caller, **fields).status_code for _ in range(FAILING_ATTEMPTS))
    assert 200 not in statuses, statuses
    plain: Final = rig.chat(group, scenario.key())
    return plain.status_code, NO_DEPLOYMENTS in plain.text


@pytest.mark.parametrize(
    "operator_key", (pytest.param("sk-throttled-operator", id="429"), pytest.param("sk-broken-operator", id="500"))
)
def test_a_routine_api_version_leaves_the_deployments_own_429_and_500_cooldowns_as_they_are(
    rig: Rig, operator_key: str
) -> None:
    """guard: when a two-deployment group's own keys keep getting 429 or 500, a caller sending a served api_version
    sees the same outcome for the next plain caller as a caller sending no api_version, and that outcome is a
    cooled down group."""
    with rig.gateway.scenario() as scenario:
        without_field: Final = _plain_caller_after_failures(rig, scenario, operator_key, {})
        with_field: Final = _plain_caller_after_failures(
            rig, scenario, operator_key, {"api_version": CALLER_API_VERSION}
        )
    assert with_field == without_field
    assert without_field == (429, True), "the control group never cooled down, so this comparison proves nothing"


def test_a_routine_organization_does_not_spare_a_deployment_whose_model_is_missing(rig: Rig) -> None:
    """guard: a caller sending a valid organization to a deployment whose model the provider answers with 404 still
    cools the deployment down, as it would without the organization."""
    fields: Final[Mapping[str, JsonValue]] = {"organization": "org-caller-valid"}
    with rig.gateway.scenario() as scenario:
        model: Final = rig.deployment(scenario, Backend.OPENAI, fields, model=MISSING_MODEL)
        first: Final = rig.chat(model, scenario.key(), **fields)
        assert first.status_code == 404, first.text
        assert_cooled_down(rig.chat(model, scenario.key()))
