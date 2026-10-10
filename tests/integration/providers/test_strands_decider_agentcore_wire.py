import hashlib
import json
import os
import re
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Final
from urllib.parse import quote, unquote

import httpx
import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, gateway_from_environment
from integration._support.database import read_rows
from integration._support.forward_proxy import ForwardProxy, tunnelling_forward_proxy
from integration._support.process import owned_proxy
from integration._support.sigv4 import encoded_path, signature
from integration._support.tls import server_context, write_self_signed_cert
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm.llms.strands_decider.decisions.transformation import StrandsDeciderDecisionsConfig

_ACCOUNT: Final = "123456789012"
_ACCESS_KEY: Final = "AKIAAGENTCOREAUDIT01"
_SECRET_KEY: Final = "synthetic-agentcore-audit-secret"
_SIGNING: Final[Mapping[str, JsonValue]] = {"aws_access_key_id": _ACCESS_KEY, "aws_secret_access_key": _SECRET_KEY}
_JWT: Final = "synthetic-agentcore-jwt"
_ENV_JWT: Final = "synthetic-agentcore-env-jwt"
_US_EAST: Final = "us-east-1"
_EU_WEST: Final = "eu-west-1"
_MODEL: Final = "strands_decider/systemone-decider"
_BODY_MODEL: Final = "systemone-decider"
_SESSION_HEADER: Final = "x-amzn-bedrock-agentcore-runtime-session-id"
_CONFIGURED_SESSION: Final = "agentcore-audit-configured-session-0000000001"
_STATE: Final[dict[str, JsonValue]] = {"ticket": "Refund requested twice for one order", "queue": "payments"}
_QUESTIONS: Final[dict[str, JsonValue]] = {
    "duplicate": {"type": "noul", "instructions": "Is this a duplicate refund request?"},
    "urgency": {"type": "choice", "criteria": {"low": "can wait", "high": "money at risk"}},
}
_SDK_QUESTIONS: Final = TypeAdapter(dict[str, dict[str, object]]).validate_python(_QUESTIONS)
_ANSWERS: Final[dict[str, JsonValue]] = {
    "duplicate": {"type": "noul", "noul": 0.88},
    "urgency": {"type": "choice", "choice": "high", "confidence": 0.75, "probabilities": {"low": 0.25, "high": 0.75}},
}
_USAGE: Final[dict[str, JsonValue]] = {"input_tokens": 211, "output_tokens": 2}
_UPSTREAM_BODY: Final[dict[str, JsonValue]] = {"model": _BODY_MODEL, "state": _STATE, "questions": _QUESTIONS}
_INPUT: Final = "Refund requested twice for one order"
_OPENAI_QUESTIONS: Final[list[JsonValue]] = [
    {"type": "predicate", "name": "duplicate", "instructions": "Is this a duplicate refund request?"},
    {
        "type": "choice",
        "name": "urgency",
        "instructions": "How urgent is it?",
        "choices": [{"value": "low", "description": "can wait"}, {"value": "high", "description": "money at risk"}],
    },
]
_OPENAI_UPSTREAM_BODY: Final[dict[str, JsonValue]] = {
    "model": _BODY_MODEL,
    "state": _INPUT,
    "questions": {
        "duplicate": {"type": "noul", "instructions": "Is this a duplicate refund request?"},
        "urgency": {
            "type": "choice",
            "instructions": "How urgent is it?",
            "criteria": {"low": "can wait", "high": "money at risk"},
        },
    },
}
_OPENAI_ANSWER: Final[dict[str, JsonValue]] = {
    "model": _BODY_MODEL,
    "answers": [
        {"type": "predicate", "name": "duplicate", "probability": 0.88},
        {
            "type": "choice",
            "name": "urgency",
            "choice": "high",
            "probabilities": [{"value": "low", "probability": 0.25}, {"value": "high", "probability": 0.75}],
            "confidence": 0.75,
        },
    ],
    "usage": {
        "input_tokens": 211,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens": 2,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 213,
    },
}


@dataclass(frozen=True, slots=True)
class _Route:
    path: str
    body: Mapping[str, JsonValue]
    upstream: Mapping[str, JsonValue]
    answer: Mapping[str, JsonValue]


_SYSTEMONE: Final = _Route(
    "/v1/systemone",
    {"state": _STATE, "questions": _QUESTIONS},
    _UPSTREAM_BODY,
    {"model": _BODY_MODEL, "answers": _ANSWERS, "usage": _USAGE},
)
_OPENAI: Final = _Route(
    "/v1/decisions", {"input": _INPUT, "questions": _OPENAI_QUESTIONS}, _OPENAI_UPSTREAM_BODY, _OPENAI_ANSWER
)
_ROUTES: Final = (_SYSTEMONE, _OPENAI, replace(_SYSTEMONE, path="/systemone"), replace(_OPENAI, path="/decisions"))
_SPEND_QUERY: Final = (
    "SELECT status, call_type, custom_llm_provider, model_group, api_base, prompt_tokens, completion_tokens "
    'FROM "LiteLLM_SpendLogs" WHERE request_id = %s'
)
_RUNTIME_TARGET: Final = re.compile(
    r"\A/runtimes/arn:aws:bedrock-agentcore:[a-z0-9-]+:[0-9]{12}:runtime/(?P<runtime>[A-Za-z0-9_-]+)/invocations\Z"
)
_ERROR_STATUS: Final = {"bad_request": 400, "invalid_request": 400, "loading": 503, "inference_error": 500}
_HTTP_ERRORS: Final = {
    403: "AccessDeniedException",
    404: "ResourceNotFoundException",
    429: "ThrottlingException",
}
_MALFORMED_SESSIONS: Final[Mapping[str, JsonValue]] = {"int": 1234567, "list": ["a" * 40], "empty": ""}
_LONG_SESSION: Final = "s" * 5120
_NEAR_MISS_ARNS: Final = {
    "endpoint": f"arn:aws:bedrock-agentcore:us-east-1:{_ACCOUNT}:runtime/audit_qualified/runtime-endpoint/DEFAULT",
    "account": "arn:aws:bedrock-agentcore:us-east-1:12345:runtime/audit_short_account",
    "service": f"arn:aws:bedrock:us-east-1:{_ACCOUNT}:runtime/audit_other_service",
    "host": f"arn:aws:bedrock-agentcore:us-east-1.example.com:{_ACCOUNT}:runtime/audit_crafted_host",
}
_DEPLOYMENT_HEADERS: Final[dict[str, JsonValue]] = {
    "x-amz-target": "AmazonBedrockAgentCore.Forged",
    "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": "deployment-forged-session-00000000000001",
    "Host": "bedrock-agentcore.us-west-2.amazonaws.com",
    "x-audit-trace": "deployment-trace",
}
_SIGNED: Final = "agentcore-audit-signed"
_REGION: Final = "agentcore-audit-region"
_SESSION: Final = "agentcore-audit-session"
_LONG: Final = "agentcore-audit-session-long"
_BEARER: Final = "agentcore-audit-bearer"
_HEADERS: Final = "agentcore-audit-headers"
_OPT_IN: Final = "agentcore-audit-opt-in"
_HEALTH: Final = "agentcore-audit-health"
_DROPPED: Final = "agentcore-audit-dropped"
_OUTAGE: Final = "agentcore-audit-outage"
_ENV: Final = "agentcore-audit-env"
_UNSIGNED: Final = "agentcore-audit-unsigned"
_SCRUBBED_PREFIXES: Final = ("AWS_", "STRANDS_DECIDER_")
_SCRUBBED_NAMES: Final = frozenset(
    {"HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY", "SSL_VERIFY", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"}
)
_load_yaml: Final[Callable[[str], object]] = yaml.safe_load
_JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_OWNED_PROXY_CELL_SECONDS: Final = 2 * max(30.0, float(os.environ.get("INTEGRATION_PROXY_READY_SECONDS") or 70)) + 120
pytestmark: Final = pytest.mark.timeout(_OWNED_PROXY_CELL_SECONDS)


def _host(region: str) -> str:
    return f"bedrock-agentcore.{region}.amazonaws.com"


def _arn(runtime: str, region: str = _US_EAST) -> str:
    return f"arn:aws:bedrock-agentcore:{region}:{_ACCOUNT}:runtime/{runtime}"


def _invocations_url(runtime: str, region: str = _US_EAST) -> str:
    return f"https://{_host(region)}/runtimes/{quote(_arn(runtime, region), safe='')}/invocations"


def _default_session(runtime: str, region: str = _US_EAST) -> str:
    return f"litellm-decider-{hashlib.sha256(_arn(runtime, region).encode()).hexdigest()}"


def _runtime_of(request: Request) -> str | None:
    match: Final = _RUNTIME_TARGET.match(unquote(request.target))
    return None if match is None else match["runtime"]


def _json_reply(body: Mapping[str, JsonValue], status: int = 200, **headers: str) -> Reply:
    return Reply(status=status, body=json.dumps(body).encode(), headers=headers)


def _respond(outage: threading.Event, request: Request) -> Reply:
    runtime: Final = _runtime_of(request) or ""
    if runtime == "audit_dropped" or (runtime == "audit_outage" and outage.is_set()):
        return Reply(drop_connection=True)
    if runtime.startswith("audit_error_"):
        code: Final = runtime.removeprefix("audit_error_")
        return _json_reply({"error": {"code": code, "message": f"runtime refused with {code}"}})
    if runtime.startswith("audit_status_"):
        status: Final = int(runtime.removeprefix("audit_status_"))
        return _json_reply(
            {"message": f"scripted {_HTTP_ERRORS[status]}"}, status, **{"x-amzn-errortype": _HTTP_ERRORS[status]}
        )
    if not runtime:
        return _json_reply({"message": f"unknown target {request.target}"}, 404)
    return _json_reply({"model": _BODY_MODEL, "answers": _ANSWERS, "usage": _USAGE})


@dataclass(frozen=True, slots=True)
class _Runtime:
    wire: Wire
    tunnel: ForwardProxy
    cert: Path
    outage: threading.Event

    def reset(self) -> None:
        self.outage.clear()
        self.wire.drain()
        self.tunnel.drain()

    def calls(self) -> tuple[Request, ...]:
        return self.wire.drain()


def _calls_to(calls: Sequence[Request], runtime: str) -> tuple[Request, ...]:
    return tuple(call for call in calls if _runtime_of(call) == runtime)


def _deployment(name: str, **litellm_params: JsonValue) -> dict[str, JsonValue]:
    return {
        "model_name": name,
        "litellm_params": {"model": _MODEL, **litellm_params},
        "model_info": {"id": name, "mode": "evaluation"},
    }


def _signed_deployment(name: str, runtime: str, region: str = _US_EAST, **extra: JsonValue) -> dict[str, JsonValue]:
    return _deployment(name, api_base=_arn(runtime, region), **_SIGNING, **extra)


def _deployments() -> list[dict[str, JsonValue]]:
    return [
        _signed_deployment(_SIGNED, "audit_signed", aws_region_name=_US_EAST),
        _signed_deployment(_REGION, "audit_region", _EU_WEST, aws_region_name=_US_EAST),
        _signed_deployment(_SESSION, "audit_session", agentcore_runtime_session_id=_CONFIGURED_SESSION),
        _signed_deployment(_LONG, "audit_session_long", agentcore_runtime_session_id=_LONG_SESSION),
        _deployment(_BEARER, api_base=_arn("audit_bearer"), api_key=_JWT),
        _signed_deployment(_HEADERS, "audit_headers", extra_headers=_DEPLOYMENT_HEADERS),
        _signed_deployment(
            _OPT_IN, "audit_opt_in", configurable_clientside_auth_params=["agentcore_runtime_session_id"]
        ),
        _signed_deployment(_HEALTH, "audit_health"),
        _signed_deployment(_DROPPED, "audit_dropped"),
        _signed_deployment(_OUTAGE, "audit_outage"),
        _deployment(_ENV, **_SIGNING),
        _deployment(_UNSIGNED, api_base=_arn("audit_unsigned")),
        *(_signed_deployment(f"agentcore-audit-error-{code}", f"audit_error_{code}") for code in _ERROR_STATUS),
        *(_signed_deployment(f"agentcore-audit-status-{status}", f"audit_status_{status}") for status in _HTTP_ERRORS),
        *(
            _signed_deployment(
                f"agentcore-audit-session-{label}", f"audit_session_{label}", agentcore_runtime_session_id=value
            )
            for label, value in _MALFORMED_SESSIONS.items()
        ),
        *(
            _deployment(f"agentcore-audit-near-miss-{label}", api_base=value, **_SIGNING)
            for label, value in _NEAR_MISS_ARNS.items()
        ),
    ]


def _write_config(directory: Path) -> Path:
    base: Final = JSON_OBJECT.validate_python(_load_yaml(Path("tests/integration/proxy_config.yaml").read_text()))
    configuration: Final = {
        **base,
        "model_list": _deployments(),
        "router_settings": {"disable_cooldowns": True, "num_retries": 0},
    }
    path: Final = directory / "agentcore.yaml"
    path.write_text(yaml.safe_dump(configuration))
    return path


def _scrubbed() -> tuple[str, ...]:
    return tuple(
        name for name in os.environ if name.upper().startswith(_SCRUBBED_PREFIXES) or name.upper() in _SCRUBBED_NAMES
    )


def _isolated_aws(directory: Path) -> dict[str, str]:
    return {
        "AWS_EC2_METADATA_DISABLED": "true",
        "AWS_SHARED_CREDENTIALS_FILE": str(directory / "absent-credentials"),
        "AWS_CONFIG_FILE": str(directory / "absent-config"),
    }


def _proxy_environment(runtime: _Runtime, directory: Path) -> dict[str, str]:
    return {
        "HTTPS_PROXY": runtime.tunnel.url,
        "NO_PROXY": "127.0.0.1,localhost",
        "SSL_CERT_FILE": str(runtime.cert),
        "STRANDS_DECIDER_API_BASE": _arn("audit_env"),
        **_isolated_aws(directory),
    }


@pytest.fixture(scope="module")
def module_gateway() -> Iterator[Gateway]:
    with gateway_from_environment() as value:
        yield value


@pytest.fixture(scope="module")
def runtime(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Runtime]:
    cert, key = write_self_signed_cert(tmp_path_factory.mktemp("agentcore-cert"), (_host(_US_EAST), _host(_EU_WEST)))
    outage: Final = threading.Event()
    with wire_server(partial(_respond, outage), tls=server_context(cert, key)) as wire:
        with tunnelling_forward_proxy(int(wire.url.rsplit(":", 1)[1])) as tunnel:
            yield _Runtime(wire, tunnel, cert, outage)


@pytest.fixture(scope="module")
def agentcore(
    module_gateway: Gateway, runtime: _Runtime, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("agentcore-proxy")
    with owned_proxy(
        module_gateway,
        directory,
        _proxy_environment(runtime, directory),
        config=_write_config(directory),
        remove_environment=_scrubbed(),
        workers=2,
    ) as candidate:
        yield candidate


def _decide(gateway: Gateway, model: str, route: _Route = _SYSTEMONE, **extra: JsonValue) -> httpx.Response:
    return gateway.request("POST", route.path, {"model": model, **route.body, **extra})


def _authorization_fields(request: Request) -> dict[str, str]:
    authorization: Final = request.headers.get("authorization", "")
    assert authorization.startswith("AWS4-HMAC-SHA256 "), dict(request.headers)
    return dict(part.split("=", 1) for part in authorization.removeprefix("AWS4-HMAC-SHA256 ").split(", "))


def _assert_signed(request: Request, region: str = _US_EAST) -> None:
    fields: Final = _authorization_fields(request)
    access_key, scope = fields["Credential"].split("/", 1)
    assert access_key == _ACCESS_KEY, fields
    assert scope == f"{request.headers['x-amz-date'][:8]}/{region}/bedrock-agentcore/aws4_request", fields
    signed: Final = fields["SignedHeaders"]
    assert {"host", "x-amz-date", _SESSION_HEADER}.issubset(signed.split(";")), signed
    expected: Final = signature(
        "POST", encoded_path(request.target), request.headers, signed, request.body, _SECRET_KEY, scope
    )
    assert fields["Signature"] == expected[1], (request.target, fields)


def _assert_answered(response: httpx.Response, model: str, route: _Route = _SYSTEMONE) -> None:
    assert response.status_code == 200, response.text
    assert response.json() == route.answer, response.text
    assert response.headers["x-litellm-model-group"] == model, dict(response.headers)


def _assert_runtime_target(request: Request, runtime: str, region: str = _US_EAST) -> None:
    assert request.method == "POST", request.method
    assert request.target == f"/runtimes/{quote(_arn(runtime, region), safe='')}/invocations", request.target
    assert request.headers["host"] == _host(region), dict(request.headers)


def _assert_runtime_call(request: Request, runtime: str, region: str = _US_EAST, route: _Route = _SYSTEMONE) -> None:
    _assert_runtime_target(request, runtime, region)
    assert json.loads(request.body) == route.upstream, request.body


def _canonical(body: JsonValue | Mapping[str, JsonValue]) -> str:
    return json.dumps(body, sort_keys=True)


def _spend_row(call_id: str) -> dict[str, JsonValue]:
    rows: Final = eventually(lambda: read_rows(_SPEND_QUERY, (call_id,)), lambda found: len(found) == 1, seconds=70)
    return rows[0]


def _spend_rows(call_ids: Sequence[str]) -> list[dict[str, JsonValue]]:
    query: Final = (
        "SELECT request_id, status FROM \"LiteLLM_SpendLogs\" WHERE request_id = ANY(string_to_array(%s, ','))"
    )
    return eventually(
        lambda: read_rows(query, (",".join(call_ids),)), lambda found: len(found) >= len(call_ids), seconds=70
    )


def _connect_targets(runtime: _Runtime) -> frozenset[str]:
    return frozenset(runtime.tunnel.targets())


def test_an_arn_deployment_answers_on_every_decisions_route_signed_for_bedrock_agentcore(
    agentcore: Gateway, runtime: _Runtime
) -> None:
    runtime.reset()
    responses: Final = tuple(_decide(agentcore, _SIGNED, route) for route in _ROUTES)
    for response, route in zip(responses, _ROUTES, strict=True):
        _assert_answered(response, _SIGNED, route)
    assert _connect_targets(runtime) == {f"{_host(_US_EAST)}:443"}
    calls: Final = runtime.calls()
    assert len(calls) == len(_ROUTES), calls
    for call, route in zip(calls, _ROUTES, strict=True):
        _assert_runtime_call(call, "audit_signed", route=route)
        assert call.headers[_SESSION_HEADER] == _default_session("audit_signed"), dict(call.headers)
        _assert_signed(call)
    for response in responses:
        row = _spend_row(response.headers["x-litellm-call-id"])
        assert row == {
            "status": "success",
            "call_type": "asystemone",
            "custom_llm_provider": "strands_decider",
            "model_group": _SIGNED,
            "api_base": _invocations_url("audit_signed"),
            "prompt_tokens": 211,
            "completion_tokens": 2,
        }, row


def test_the_arn_region_picks_the_host_and_signing_scope_over_aws_region_name(
    agentcore: Gateway, runtime: _Runtime
) -> None:
    runtime.reset()
    _assert_answered(_decide(agentcore, _REGION), _REGION)
    assert _connect_targets(runtime) == {f"{_host(_EU_WEST)}:443"}
    (call,) = runtime.calls()
    _assert_runtime_call(call, "audit_region", _EU_WEST)
    _assert_signed(call, _EU_WEST)


def test_a_configured_session_id_is_sent_as_written_and_signed(agentcore: Gateway, runtime: _Runtime) -> None:
    runtime.reset()
    _assert_answered(_decide(agentcore, _SESSION), _SESSION)
    _assert_answered(_decide(agentcore, _LONG), _LONG)
    calls: Final = runtime.calls()
    (configured,) = _calls_to(calls, "audit_session")
    (long,) = _calls_to(calls, "audit_session_long")
    assert configured.headers[_SESSION_HEADER] == _CONFIGURED_SESSION, dict(configured.headers)
    assert long.headers[_SESSION_HEADER] == _LONG_SESSION
    _assert_signed(configured)
    _assert_signed(long)


def test_a_deployment_api_key_sends_bearer_and_the_session_without_sigv4(agentcore: Gateway, runtime: _Runtime) -> None:
    runtime.reset()
    _assert_answered(_decide(agentcore, _BEARER), _BEARER)
    (call,) = runtime.calls()
    _assert_runtime_call(call, "audit_bearer")
    assert call.headers["authorization"] == f"Bearer {_JWT}", dict(call.headers)
    assert call.headers[_SESSION_HEADER] == _default_session("audit_bearer"), dict(call.headers)
    assert "x-amz-date" not in call.headers, dict(call.headers)


def test_reserved_extra_headers_are_dropped_and_the_rest_are_kept(agentcore: Gateway, runtime: _Runtime) -> None:
    runtime.reset()
    _assert_answered(_decide(agentcore, _HEADERS), _HEADERS)
    caller_headers: Final[dict[str, JsonValue]] = {
        "x-amz-security-token": "caller-forged-token",
        "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": "caller-forged-session-0000000000000000001",
        "x-audit-caller": "caller-trace",
    }
    _assert_answered(_decide(agentcore, _HEADERS, extra_headers=caller_headers), _HEADERS)
    deployment, caller = runtime.calls()
    for call in (deployment, caller):
        _assert_runtime_call(call, "audit_headers")
        assert call.headers[_SESSION_HEADER] == _default_session("audit_headers"), dict(call.headers)
        assert "x-amz-target" not in call.headers, dict(call.headers)
        assert "x-amz-security-token" not in call.headers, dict(call.headers)
        _assert_signed(call)
    assert deployment.headers["x-audit-trace"] == "deployment-trace", dict(deployment.headers)
    assert caller.headers["x-audit-caller"] == "caller-trace", dict(caller.headers)


def test_an_opted_in_caller_session_id_becomes_the_header_and_stays_out_of_the_body(
    agentcore: Gateway, runtime: _Runtime
) -> None:
    runtime.reset()
    chosen: Final = f"agentcore-audit-caller-session-{uuid.uuid4().hex}"
    _assert_answered(_decide(agentcore, _OPT_IN, agentcore_runtime_session_id=chosen), _OPT_IN)
    _assert_answered(_decide(agentcore, _OPT_IN), _OPT_IN)
    first, second = runtime.calls()
    for call in (first, second):
        _assert_runtime_call(call, "audit_opt_in")
        _assert_signed(call)
    assert first.headers[_SESSION_HEADER] == chosen, dict(first.headers)
    assert second.headers[_SESSION_HEADER] == _default_session("audit_opt_in"), dict(second.headers)


def test_an_arn_deployment_without_api_base_takes_the_runtime_from_the_environment(
    agentcore: Gateway, runtime: _Runtime
) -> None:
    runtime.reset()
    response: Final = _decide(agentcore, _ENV)
    _assert_answered(response, _ENV)
    (call,) = runtime.calls()
    _assert_runtime_call(call, "audit_env")
    _assert_signed(call)
    assert _spend_row(response.headers["x-litellm-call-id"])["api_base"] == _invocations_url("audit_env")


def test_an_arn_deployment_added_through_the_management_api_answers_signed(
    agentcore: Gateway, runtime: _Runtime
) -> None:
    runtime.reset()
    with agentcore.scenario() as scenario:
        model: Final = scenario.model(
            model_info=None,
            model=_MODEL,
            api_base=_arn("audit_stored"),
            api_key=None,
            aws_access_key_id=_ACCESS_KEY,
            aws_secret_access_key=_SECRET_KEY,
        )
        _assert_answered(_decide(agentcore, model), model)
    (call,) = runtime.calls()
    _assert_runtime_call(call, "audit_stored")
    _assert_signed(call)


def test_health_and_test_connection_probe_the_runtime_signed(agentcore: Gateway, runtime: _Runtime) -> None:
    runtime.reset()
    health: Final = agentcore.request("GET", "/health", params={"model": _HEALTH})
    assert health.status_code == 200, health.text
    assert (health.json()["healthy_count"], health.json()["unhealthy_count"]) == (1, 0), health.text
    (probe,) = runtime.calls()
    assert probe.target == f"/runtimes/{quote(_arn('audit_health'), safe='')}/invocations", probe.target
    _assert_signed(probe)
    connection: Final = agentcore.request(
        "POST",
        "/health/test_connection",
        {
            "litellm_params": {"model": _MODEL, "api_base": _arn("audit_test_connection"), **_SIGNING},
            "mode": "evaluation",
        },
    )
    assert connection.status_code == 200, connection.text
    assert connection.json()["status"] == "success", connection.text
    (tested,) = runtime.calls()
    assert _runtime_of(tested) == "audit_test_connection", tested.target
    _assert_signed(tested)


@pytest.mark.parametrize(("code", "status"), tuple(_ERROR_STATUS.items()), ids=tuple(_ERROR_STATUS))
def test_a_runtime_error_reply_maps_to_its_status_with_the_runtime_message(
    agentcore: Gateway, runtime: _Runtime, code: str, status: int
) -> None:
    runtime.reset()
    model: Final = f"agentcore-audit-error-{code}"
    response: Final = _decide(agentcore, model)
    assert response.status_code == status, response.text
    assert f"Strands Decider runtime error '{code}': runtime refused with {code}" in response.text, response.text
    assert len(_calls_to(runtime.calls(), f"audit_error_{code}")) == 1
    row: Final = _spend_row(response.headers["x-litellm-call-id"])
    assert (row["status"], row["model_group"]) == ("failure", model), row


@pytest.mark.parametrize("status", tuple(_HTTP_ERRORS), ids=tuple(str(status) for status in _HTTP_ERRORS))
def test_a_runtime_http_error_keeps_its_status_and_message(agentcore: Gateway, runtime: _Runtime, status: int) -> None:
    runtime.reset()
    model: Final = f"agentcore-audit-status-{status}"
    response: Final = _decide(agentcore, model)
    assert response.status_code == status, response.text
    assert f"scripted {_HTTP_ERRORS[status]}" in response.text, response.text
    assert len(_calls_to(runtime.calls(), f"audit_status_{status}")) == 1
    assert _spend_row(response.headers["x-litellm-call-id"])["status"] == "failure"


@pytest.mark.parametrize("label", tuple(_MALFORMED_SESSIONS))
def test_a_malformed_configured_session_id_fails_only_its_own_deployment(
    agentcore: Gateway, runtime: _Runtime, label: str
) -> None:
    runtime.reset()
    model: Final = f"agentcore-audit-session-{label}"
    response: Final = _decide(agentcore, model)
    calls: Final = _calls_to(runtime.calls(), f"audit_session_{label}")
    if label == "empty":
        _assert_answered(response, model)
        (call,) = calls
        assert call.headers[_SESSION_HEADER] == "", dict(call.headers)
    else:
        assert response.status_code == 500, response.text
        assert "agentcore_runtime_session_id" in response.text, response.text
        assert calls == ()
    _assert_answered(_decide(agentcore, _SIGNED), _SIGNED)


def test_a_deployment_without_aws_credentials_fails_before_dialing_the_runtime(
    agentcore: Gateway, runtime: _Runtime
) -> None:
    runtime.reset()
    response: Final = _decide(agentcore, _UNSIGNED)
    assert response.status_code == 500, response.text
    assert "credentials" in response.text.lower(), response.text
    assert runtime.calls() == ()
    assert _connect_targets(runtime) == frozenset()
    _assert_answered(_decide(agentcore, _SIGNED), _SIGNED)


@pytest.mark.parametrize("label", tuple(_NEAR_MISS_ARNS))
def test_a_near_miss_arn_keeps_the_plain_api_base_path(agentcore: Gateway, runtime: _Runtime, label: str) -> None:
    runtime.reset()
    response: Final = _decide(agentcore, f"agentcore-audit-near-miss-{label}")
    assert 500 <= response.status_code < 600, response.text
    assert "litellm.APIConnectionError" in response.text, response.text
    assert _connect_targets(runtime) == frozenset()
    assert runtime.calls() == ()


def test_a_dropped_connection_is_a_connection_error_after_one_attempt(agentcore: Gateway, runtime: _Runtime) -> None:
    runtime.reset()
    response: Final = _decide(agentcore, _DROPPED)
    assert response.status_code == 500, response.text
    assert "litellm.APIConnectionError" in response.text, response.text
    assert len(_calls_to(runtime.calls(), "audit_dropped")) == 1
    assert _spend_row(response.headers["x-litellm-call-id"])["status"] == "failure"


def _burst_routes(size: int) -> tuple[_Route, ...]:
    return tuple(_ROUTES[index % len(_ROUTES)] for index in range(size))


def _burst(gateway: Gateway, model: str, size: int) -> tuple[httpx.Response, ...]:
    with ThreadPoolExecutor(max_workers=size) as pool:
        return tuple(pool.map(partial(_decide, gateway, model), _burst_routes(size)))


def _health_counts(gateway: Gateway, model: str) -> tuple[int, int, int]:
    response: Final = gateway.request("GET", "/health", params={"model": model})
    body: Final = JSON_OBJECT.validate_json(response.content)
    healthy: Final = body["healthy_count"]
    unhealthy: Final = body["unhealthy_count"]
    assert isinstance(healthy, int) and isinstance(unhealthy, int), body
    return response.status_code, healthy, unhealthy


def test_a_runtime_outage_mid_burst_fails_each_call_once_and_recovers_signed(
    agentcore: Gateway, runtime: _Runtime
) -> None:
    runtime.reset()
    runtime.outage.set()
    failed: Final = _burst(agentcore, _OUTAGE, 10)
    assert [response.status_code for response in failed] == [500] * 10, [response.text for response in failed]
    assert _health_counts(agentcore, _OUTAGE) == (503, 0, 1)
    runtime.outage.clear()
    runtime.calls()
    served: Final = _burst(agentcore, _OUTAGE, 20)
    for response, route in zip(served, _burst_routes(20), strict=True):
        _assert_answered(response, _OUTAGE, route)
    calls: Final = runtime.calls()
    assert len(calls) == 20, calls
    for call in calls:
        _assert_runtime_target(call, "audit_outage")
        _assert_signed(call)
    assert sorted(_canonical(_JSON.validate_json(call.body)) for call in calls) == sorted(
        _canonical(route.upstream) for route in _burst_routes(20)
    )
    assert _health_counts(agentcore, _OUTAGE) == (200, 1, 0)
    expected: Final = {
        **{response.headers["x-litellm-call-id"]: "failure" for response in failed},
        **{response.headers["x-litellm-call-id"]: "success" for response in served},
    }
    assert len(expected) == 30, expected
    rows: Final = _spend_rows(tuple(expected))
    assert sorted((str(row["request_id"]), row["status"]) for row in rows) == sorted(expected.items()), rows


async def test_sdk_sync_and_async_calls_reach_the_runtime_signed(
    runtime: _Runtime, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runtime.reset()
    for name in _scrubbed():
        monkeypatch.delenv(name)
    for name, value in {
        "HTTPS_PROXY": runtime.tunnel.url,
        "NO_PROXY": "127.0.0.1,localhost",
        "SSL_CERT_FILE": str(runtime.cert),
        **_isolated_aws(tmp_path),
    }.items():
        monkeypatch.setenv(name, value)
    litellm.in_memory_llm_clients_cache.flush_cache()
    try:
        synchronous: Final = litellm.decisions(
            model=_MODEL,
            state=_STATE,
            questions=_SDK_QUESTIONS,
            api_base=_arn("audit_sdk"),
            aws_access_key_id=_ACCESS_KEY,
            aws_secret_access_key=_SECRET_KEY,
        )
        asynchronous: Final = await litellm.asystemone(
            model=_MODEL,
            state=_STATE,
            questions=_SDK_QUESTIONS,
            api_base=_arn("audit_sdk"),
            aws_access_key_id=_ACCESS_KEY,
            aws_secret_access_key=_SECRET_KEY,
        )
        monkeypatch.setenv("STRANDS_DECIDER_API_KEY", _ENV_JWT)
        bearer: Final = await litellm.asystemone(
            model=_MODEL, state=_STATE, questions=_SDK_QUESTIONS, api_base=_arn("audit_sdk_env_key")
        )
    finally:
        litellm.in_memory_llm_clients_cache.flush_cache()
    for response in (synchronous, asynchronous, bearer):
        assert response.model_dump(mode="json") == {"model": _BODY_MODEL, "answers": _ANSWERS, "usage": _USAGE}
    calls: Final = runtime.calls()
    signed: Final = _calls_to(calls, "audit_sdk")
    assert len(signed) == 2, calls
    for call in signed:
        _assert_runtime_call(call, "audit_sdk")
        assert call.headers[_SESSION_HEADER] == _default_session("audit_sdk"), dict(call.headers)
        _assert_signed(call)
    (env_key,) = _calls_to(calls, "audit_sdk_env_key")
    assert env_key.headers["authorization"] == f"Bearer {_ENV_JWT}", dict(env_key.headers)
    assert "x-amz-date" not in env_key.headers, dict(env_key.headers)


def _answering(request: Request) -> Reply:
    return _json_reply({"model": _BODY_MODEL, "answers": _ANSWERS, "usage": _USAGE})


def _envelope(request: Request) -> Reply:
    return _json_reply({"error": {"code": "bad_request", "message": "plain server refused"}})


def _plain_model(scenario: Scenario, wire: Wire, **extra: JsonValue) -> str:
    return scenario.model(model_info=None, model=_MODEL, api_base=wire.url, api_key=None, **extra)


def test_a_request_body_session_id_is_refused_on_every_route_without_an_upstream_call(gateway: Gateway) -> None:
    with wire_server(_answering) as wire, gateway.scenario() as scenario:
        model: Final = _plain_model(scenario, wire)
        responses: Final = tuple(
            _decide(gateway, model, route, agentcore_runtime_session_id=f"caller-{uuid.uuid4().hex}")
            for route in _ROUTES
        )
        for response in responses:
            assert response.status_code == 401, response.text
            assert "agentcore_runtime_session_id is not allowed in request body" in response.text, response.text
        assert wire.drain() == ()


def test_a_plain_api_base_opted_into_the_session_param_sends_no_session_header_or_body_field(
    gateway: Gateway,
) -> None:
    with wire_server(_answering) as wire, gateway.scenario() as scenario:
        model: Final = _plain_model(
            scenario, wire, configurable_clientside_auth_params=["agentcore_runtime_session_id"]
        )
        _assert_answered(_decide(gateway, model, agentcore_runtime_session_id=f"caller-{uuid.uuid4().hex}"), model)
        (call,) = wire.drain()
        assert call.target == "/v1/systemone", call.target
        assert _SESSION_HEADER not in call.headers, dict(call.headers)
        assert json.loads(call.body) == _UPSTREAM_BODY, call.body


def test_a_plain_api_base_error_envelope_keeps_the_unexpected_response_server_error(gateway: Gateway) -> None:
    with wire_server(_envelope) as wire, gateway.scenario() as scenario:
        model: Final = _plain_model(scenario, wire)
        response: Final = _decide(gateway, model)
        assert response.status_code == 500, response.text
        assert "Strands Decider runtime error" not in response.text, response.text
        assert len(wire.drain()) == 1


def test_a_plain_api_base_forwards_amazon_shaped_extra_headers_unchanged(gateway: Gateway) -> None:
    headers: Final[dict[str, JsonValue]] = {
        "x-amz-target": "plain-target",
        "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": "plain-session-kept",
        "x-audit-trace": "plain-trace",
    }
    with wire_server(_answering) as wire, gateway.scenario() as scenario:
        model: Final = _plain_model(scenario, wire, extra_headers=headers)
        _assert_answered(_decide(gateway, model), model)
        (call,) = wire.drain()
        assert (call.headers["x-amz-target"], call.headers[_SESSION_HEADER], call.headers["x-audit-trace"]) == (
            "plain-target",
            "plain-session-kept",
            "plain-trace",
        ), dict(call.headers)
        assert "authorization" not in call.headers, dict(call.headers)
        assert json.loads(call.body) == _UPSTREAM_BODY, call.body


def _worst_case_api_base() -> str:
    return "arn:aws" + "-a" * 131072 + "!"


def test_a_worst_case_runtime_arn_pattern_input_is_classified_in_linear_time(
    gateway: Gateway, record_property: Callable[[str, object], None]
) -> None:
    worst: Final = _worst_case_api_base()
    started: Final = time.perf_counter()
    url: Final = StrandsDeciderDecisionsConfig().get_complete_url(worst, _BODY_MODEL)
    classified: Final = time.perf_counter() - started
    record_property("classifier_seconds", f"{classified:.6f}")
    assert url == f"{worst}/v1/systemone"
    assert classified < 0.5, classified
    with wire_server(_answering) as wire, gateway.scenario() as scenario:
        model: Final = _plain_model(scenario, wire, configurable_clientside_auth_params=["api_base"])
        with ThreadPoolExecutor(max_workers=2) as pool:
            timed: Final = pool.submit(_timed_decide, gateway, model, worst)
            liveliness_started: Final = time.perf_counter()
            liveliness: Final = gateway.request("GET", "/health/liveliness")
            liveliness_seconds: Final = time.perf_counter() - liveliness_started
            response, decide_seconds = timed.result()
        record_property("decide_seconds", f"{decide_seconds:.6f}")
        record_property("liveliness_seconds", f"{liveliness_seconds:.6f}")
        assert liveliness.status_code == 200, liveliness.text
        assert 500 <= response.status_code < 600, response.text
        assert "litellm.APIConnectionError" in response.text, response.text[:2000]
        assert decide_seconds < 5, decide_seconds
        assert liveliness_seconds < 2, liveliness_seconds
        assert wire.drain() == ()


def _timed_decide(gateway: Gateway, model: str, api_base: str) -> tuple[httpx.Response, float]:
    started: Final = time.perf_counter()
    response: Final = _decide(gateway, model, api_base=api_base)
    return response, time.perf_counter() - started
