"""Native Claude on Vertex AI `/v1/messages` calls must be indistinguishable from the Python implementation.

Each case sends one `vertex_ai/claude-*` SDK call through the Python handler and through the native route, with a
service account whose token endpoint is a local server, and compares the same aspects as `test_native_parity.py`.
A case without an `api_base` posts through a caller handler that answers with the URL and headers it was given, since
the default Google host is not reachable from a test. Every known difference is a strict xfail naming the gap
"""

import contextlib
import json
import os
import tempfile
import threading
from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar, Final, cast

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from pydantic import JsonValue

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from tests.test_litellm_rust.support.callback_recorder import drain_logging
from tests.test_litellm_rust.support.isolation import isolated_callback_registries, rebound
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec

from .test_native_parity import (
    C_UNSENT_CALL_FLAG,
    _PYTHON_AMESSAGES,  # pyright: ignore[reportPrivateUsage]  # the Python baseline under comparison
    TEXT_RESPONSE,
    TEXT_STREAM,
    USER,
    Case,
    FallbackProbe,
    _consume_async,  # pyright: ignore[reportPrivateUsage]  # the shared stream reader
    _environment,  # pyright: ignore[reportPrivateUsage]  # the shared environment isolation
    _failure,  # pyright: ignore[reportPrivateUsage]  # the shared upstream error builder
    _legs,  # pyright: ignore[reportPrivateUsage]  # the shared leg runner
    _stream,  # pyright: ignore[reportPrivateUsage]  # the shared stream builder
    _stream_text,  # pyright: ignore[reportPrivateUsage]  # the shared stream reader
    assert_aspect_matches,
    fallback_probe,
)

pytestmark = pytest.mark.requires_rust_extension

MODEL: Final = "vertex_ai/claude-sonnet-4-5"
OPUS: Final = "vertex_ai/claude-opus-4-7"
PROJECT: Final = "parity-project"
LOCATION: Final = "us-east5"
_VERTEX_ENVIRONMENT: Final = (
    "VERTEXAI_PROJECT",
    "VERTEX_PROJECT",
    "VERTEXAI_LOCATION",
    "VERTEX_LOCATION",
    "VERTEXAI_CREDENTIALS",
    "VERTEX_AI_API_KEY",
    "VERTEXAI_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_CLOUD_PROJECT",
)
_SCRATCH: Final = Path(tempfile.mkdtemp(prefix="vertex-parity-"))
_PRIVATE_KEY: Final = (
    rsa.generate_private_key(public_exponent=65537, key_size=2048)
    .private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    .decode()
)


class _TokenEndpoint:
    """An OAuth token endpoint that grants on `/token` and refuses on `/denied`, counting each exchange by
    the service account that asked."""

    grants: ClassVar[dict[str, int]] = {}

    def __init__(self) -> None:
        endpoint: Final = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                endpoint.grants[self.path] = endpoint.grants.get(self.path, 0) + 1
                denied: Final = self.path.startswith("/denied")
                body: Final = json.dumps(
                    {"error": "invalid_grant", "error_description": "Invalid JWT Signature."}
                    if denied
                    else {"access_token": "ya29.parity", "expires_in": 3600, "token_type": "Bearer"}
                ).encode()
                self.send_response(400 if denied else 200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                pass

        self.server: Final = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host!s}:{port}"


TOKENS: Final = _TokenEndpoint()


def _account(name: str = "parity", path: str = "/token", **fields: JsonValue) -> Mapping[str, JsonValue]:
    return MappingProxyType(
        {
            "type": "service_account",
            "project_id": PROJECT,
            "private_key_id": "key-1",
            "private_key": _PRIVATE_KEY,
            "client_email": f"{name}@{PROJECT}.iam.gserviceaccount.com",
            "client_id": "1",
            "token_uri": f"{TOKENS.base_url}{path}/{name}",
            **fields,
        }
    )


def _account_file(name: str) -> str:
    path: Final = _SCRATCH / f"{name}.json"
    path.write_text(json.dumps(dict(_account(name))))
    return str(path)


ACCOUNT: Final = _account()


def _vx(**kwargs: object) -> Mapping[str, object]:
    return MappingProxyType(
        {
            "api_key": None,
            "vertex_project": PROJECT,
            "vertex_location": LOCATION,
            "vertex_credentials": dict(ACCOUNT),
            **kwargs,
        }
    )


def _env(**env: str) -> Mapping[str, str]:
    return MappingProxyType(env)


def _gaps(**gaps: str) -> Mapping[str, str | None]:
    return MappingProxyType(gaps)


class _GoogleEcho(AsyncHTTPHandler):
    """A caller handler standing in for the Google host: it answers with the URL and headers it was asked
    to post to, so the compared response shows where each lane sent the call."""

    def __deepcopy__(self, memo: dict[int, object]) -> "_GoogleEcho":
        return self

    async def post(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, url: str, data: object = None, headers: Mapping[str, str] | None = None, **kwargs: object
    ) -> httpx.Response:  # kwargs-ok: the test double ignores the rest of the post
        sent: Final = {
            "url": url,
            "headers": {
                name.lower(): value
                for name, value in sorted((headers or {}).items())
                if name.lower() not in ("content-length", "user-agent")
            },
            "body": json.loads(data) if isinstance(data, str | bytes) else None,
        }
        return httpx.Response(
            200,
            json={**TEXT_RESPONSE, "content": [{"type": "text", "text": json.dumps(sent, sort_keys=True)}]},
            request=httpx.Request(
                "POST", url, headers=headers, content=data if isinstance(data, str | bytes) else None
            ),
        )


def _google(**kwargs: object) -> Mapping[str, object]:
    return _vx(api_base=None, client=_GoogleEcho(), **kwargs)


WEB_SEARCH: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"type": "web_search_20250305", "name": "web_search", "max_uses": 2}
)
TOOL_SEARCH: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}
)
WEATHER_TOOL: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        "name": "get_weather",
        "description": "Current weather for a city",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    }
)
NATIVE_ADC_USES_GCLOUD: Final = (
    "application default credentials: the native lane (gcp_auth) falls back to `gcloud auth print-access-token`, "
    "which google-auth's default() never runs, so Python raises DefaultCredentialsError where the native call is sent"
)
PYTHON_DOUBLES_API_VERSION: Final = (
    "an api_base ending in /v1 or /v1beta1: Python's get_complete_vertex_url joins the default path onto it "
    "before _check_custom_proxy grafts, so it posts to /v1/v1/...; the native lane grafts once"
)
REFRESH_ERROR_TEXT: Final = (
    "a token endpoint refusal: both raise google-auth's RefreshError, Python with the endpoint's error and body, "
    "the native lane with its own token request failure text"
)
_ADC_GAP_ASPECTS: Final = (
    "request",
    "outcome",
    "hidden_params",
    "logged",
    "logged_response_headers",
    "logged_call",
    "logged_request",
    "logged_model_parameters",
    "logged_error",
)
_DOUBLED_API_VERSION: Final = _gaps(
    request=PYTHON_DOUBLES_API_VERSION,
    logged=PYTHON_DOUBLES_API_VERSION,
    logged_sent_request=PYTHON_DOUBLES_API_VERSION,
)
_GCLOUD: Final = _SCRATCH / "bin" / "gcloud"
_GCLOUD.parent.mkdir()
_GCLOUD.write_text("#!/bin/sh\necho gcloud-cli-token\n")
_GCLOUD.chmod(0o755)
_NO_ADC: Final = _env(
    PATH=f"{_GCLOUD.parent}:{os.environ.get('PATH', '')}",
    CLOUDSDK_CONFIG=str(_SCRATCH / "gcloud"),
    GCE_METADATA_HOST="127.0.0.1:9",
    GCE_METADATA_IP="127.0.0.1:9",
    NO_GCE_CHECK="True",
    HOME=str(_SCRATCH),
)

CALL_CASES: Final = (
    Case("vertex_plain", _vx(), model=MODEL),
    Case("vertex_stream", _vx(stream=True), upstream=(_stream(TEXT_STREAM),), model=MODEL),
    Case("vertex_sync", _vx(), model=MODEL, sync=True),
    Case("vertex_sync_stream", _vx(stream=True), upstream=(_stream(TEXT_STREAM),), model=MODEL, sync=True),
    Case("vertex_opus", _vx(), model=OPUS),
    Case("vertex_model_version_suffix", _vx(), model="vertex_ai/claude-sonnet-4-5@20250929"),
)

LOCATION_CASES: Final = (
    Case(
        "vertex_project_alias_params",
        _vx(
            vertex_project=None,
            vertex_location=None,
            vertex_ai_project="alias-project",
            vertex_ai_location="europe-west1",
        ),
        model=MODEL,
    ),
    Case(
        "vertex_project_and_location_from_globals",
        _vx(vertex_project=None, vertex_location=None),
        settings=MappingProxyType({"vertex_project": "global-project", "vertex_location": "us-central1"}),
        model=MODEL,
    ),
    Case(
        "vertex_project_and_location_from_environment",
        _vx(vertex_project=None, vertex_location=None),
        env=_env(VERTEXAI_PROJECT="env-project", VERTEXAI_LOCATION="asia-southeast1"),
        model=MODEL,
    ),
    Case(
        "vertex_location_from_vertex_location_environment",
        _vx(vertex_location=None),
        env=_env(VERTEX_LOCATION="europe-west4"),
        model=MODEL,
    ),
    Case("vertex_project_from_credentials", _vx(vertex_project=None), model=MODEL),
    Case("vertex_no_location_defaults", _vx(vertex_location=None), model=MODEL),
    Case("vertex_location_global", _vx(vertex_location="global"), model=MODEL),
    Case(
        "vertex_location_global_stream",
        _vx(vertex_location="global", stream=True),
        upstream=(_stream(TEXT_STREAM),),
        model=MODEL,
    ),
    Case(
        "vertex_kwargs_outrank_globals",
        _vx(),
        settings=MappingProxyType({"vertex_project": "global-project", "vertex_location": "us-central1"}),
        model=MODEL,
    ),
)

URL_CASES: Final = (
    Case("vertex_api_base_trailing_slash", _vx(api_base_suffix="/"), model=MODEL),
    Case("vertex_api_base_v1", _vx(api_base_suffix="/v1"), model=MODEL, gaps=_DOUBLED_API_VERSION),
    Case("vertex_api_base_v1beta1", _vx(api_base_suffix="/v1beta1"), model=MODEL, gaps=_DOUBLED_API_VERSION),
    Case("vertex_api_base_full_path", _vx(api_base_suffix="/custom/path"), model=MODEL),
    Case(
        "vertex_api_base_full_path_stream",
        _vx(api_base_suffix="/custom/path", stream=True),
        upstream=(_stream(TEXT_STREAM),),
        model=MODEL,
    ),
    Case(
        "vertex_api_base_v1_stream",
        _vx(api_base_suffix="/v1", stream=True),
        upstream=(_stream(TEXT_STREAM),),
        model=MODEL,
        gaps=_DOUBLED_API_VERSION,
    ),
    Case("vertex_default_url_regional", _google(), upstream=(), model=MODEL),
    Case("vertex_default_url_global", _google(vertex_location="global"), upstream=(), model=MODEL),
    Case("vertex_default_url_multi_region", _google(vertex_location="eu"), upstream=(), model=MODEL),
    Case("vertex_default_url_no_location", _google(vertex_location=None), upstream=(), model=MODEL),
    Case("vertex_default_url_invalid_location", _google(vertex_location="evil.example/"), upstream=(), model=MODEL),
)

CREDENTIAL_CASES: Final = (
    Case(
        "vertex_credentials_json_string", _vx(vertex_credentials=json.dumps(dict(_account("json-string")))), model=MODEL
    ),
    Case("vertex_credentials_file_path", _vx(vertex_credentials=_account_file("file-path")), model=MODEL),
    Case(
        "vertex_credentials_from_environment",
        _vx(vertex_credentials=None),
        env=_env(VERTEXAI_CREDENTIALS=json.dumps(dict(_account("environment")))),
        model=MODEL,
    ),
    Case(
        "vertex_credentials_from_google_application_credentials",
        _vx(vertex_credentials=None),
        env=_env(GOOGLE_APPLICATION_CREDENTIALS=_account_file("application")),
        model=MODEL,
    ),
    Case(
        "vertex_credentials_missing_application_credentials_file",
        _vx(vertex_credentials=None, vertex_project="missing-file-project"),
        upstream=(),
        env=_env(GOOGLE_APPLICATION_CREDENTIALS=str(_SCRATCH / "missing.json")),
        model=MODEL,
    ),
    Case(
        "vertex_credentials_missing_adc",
        _vx(vertex_credentials=None, vertex_project="missing-adc-project"),
        upstream=(),
        env=_NO_ADC,
        model=MODEL,
        gaps=_gaps(**dict.fromkeys(_ADC_GAP_ASPECTS, NATIVE_ADC_USES_GCLOUD)),
    ),
    Case("vertex_credentials_not_json", _vx(vertex_credentials="{not json"), upstream=(), model=MODEL),
    Case(
        "vertex_credentials_missing_file",
        _vx(vertex_credentials=str(_SCRATCH / "absent.json")),
        upstream=(),
        model=MODEL,
    ),
    Case(
        "vertex_credentials_missing_private_key",
        _vx(vertex_credentials={k: v for k, v in _account("no-key").items() if k != "private_key"}),
        upstream=(),
        model=MODEL,
    ),
    Case(
        "vertex_credentials_token_denied",
        _vx(vertex_credentials=dict(_account("denied", path="/denied"))),
        upstream=(),
        model=MODEL,
        gaps=_gaps(outcome=REFRESH_ERROR_TEXT, logged_error=REFRESH_ERROR_TEXT),
    ),
    Case("vertex_forwarded_authorization", _vx(extra_headers={"Authorization": "Bearer caller-token"}), model=MODEL),
    Case(
        "vertex_project_unresolved",
        _vx(
            vertex_project=None,
            vertex_credentials={k: v for k, v in _account("no-project").items() if k != "project_id"},
        ),
        upstream=(),
        model=MODEL,
    ),
)

VERSION_CASES: Final = (
    Case("vertex_version_capitalized_header", _vx(extra_headers={"Anthropic-Version": "2026-10-21"}), model=MODEL),
    Case("vertex_version_vertex_default", _vx(extra_headers={"anthropic-version": "vertex-2023-10-16"}), model=MODEL),
    Case("vertex_version_2023_01_01", _vx(extra_headers={"anthropic-version": "2023-01-01"}), model=MODEL),
    Case("vertex_version_blank", _vx(extra_headers={"anthropic-version": " "}), model=MODEL),
    Case(
        "vertex_version_in_provider_specific_header",
        _vx(
            provider_specific_header={
                "custom_llm_provider": "vertex_ai",
                "extra_headers": {"anthropic-version": "2026-10-21", "x-request-id": "req-1"},
            }
        ),
        model=MODEL,
    ),
    Case("vertex_version_2023_06_01", _vx(extra_headers={"anthropic-version": "2023-06-01"}), model=MODEL),
    Case("vertex_version_2026_10_21", _vx(extra_headers={"anthropic-version": "2026-10-21"}), model=MODEL),
    Case(
        "vertex_version_unsupported", _vx(extra_headers={"anthropic-version": "2099-01-01"}), upstream=(), model=MODEL
    ),
    Case(
        "vertex_version_unsupported_dropped",
        _vx(extra_headers={"anthropic-version": "2099-01-01"}, drop_params=True),
        model=MODEL,
    ),
    Case(
        "vertex_version_unsupported_dropped_globally",
        _vx(extra_headers={"anthropic-version": "2099-01-01"}),
        settings=MappingProxyType({"drop_params": True}),
        model=MODEL,
    ),
)

BETA_CASES: Final = (
    Case("vertex_beta_web_search", _vx(tools=[dict(WEB_SEARCH)]), model=MODEL),
    Case(
        "vertex_beta_tool_search", _vx(tools=[dict(TOOL_SEARCH), {**WEATHER_TOOL, "defer_loading": True}]), model=OPUS
    ),
    Case(
        "vertex_beta_context_management",
        _vx(context_management={"edits": [{"type": "clear_tool_uses_20250919"}]}),
        model=MODEL,
    ),
    Case(
        "vertex_beta_thinking", _vx(thinking={"type": "enabled", "budget_tokens": 1024}, max_tokens=2048), model=MODEL
    ),
    Case("vertex_beta_compaction", _vx(compaction={"type": "summarize"}), model=OPUS),
    Case(
        "vertex_beta_caller_header_merged",
        _vx(
            tools=[dict(WEB_SEARCH)],
            extra_headers={"anthropic-beta": "files-api-2025-04-14, web-search-2025-03-05, example-beta-2099-01-01"},
        ),
        model=MODEL,
    ),
    Case(
        "vertex_beta_caller_unknown_beta_dropped",
        _vx(extra_headers={"anthropic-beta": "example-beta-2099-01-01,context-1m-2025-08-07"}),
        model=MODEL,
    ),
    Case(
        "vertex_beta_output_config_format",
        _vx(output_config={"format": {"type": "json_schema", "schema": {"type": "object"}}}),
        model=OPUS,
    ),
    Case(
        "vertex_beta_speed_fast", _vx(speed="fast"), upstream=(), model=OPUS, gaps=_gaps(logged_call=C_UNSENT_CALL_FLAG)
    ),
    Case("vertex_thread_create", _vx(thread={"type": "create"}), upstream=(), model=MODEL),
    Case("vertex_thread_create_dropped", _vx(thread={"type": "create"}, drop_params=True), model=MODEL),
    Case(
        "vertex_thread_continue",
        _vx(thread={"type": "continue", "previous_message_id": "msg_01Previous"}, drop_params=True),
        upstream=(),
        model=MODEL,
    ),
    Case(
        "vertex_beta_thinking_display_updates",
        _vx(thinking={"type": "adaptive", "display": "updates"}),
        model=OPUS,
    ),
    Case(
        "vertex_beta_thinking_block_binding",
        _vx(thinking={"type": "enabled", "budget_tokens": 1024, "block_binding": "strict"}, max_tokens=2048),
        model=MODEL,
    ),
    Case(
        "vertex_beta_safeguards", _vx(safeguards={"dangerous_tool_use": {"action": "flag"}}), upstream=(), model=MODEL
    ),
)

BODY_CASES: Final = (
    Case(
        "vertex_cache_control_scope_stripped",
        _vx(
            system=[
                {"type": "text", "text": "Answer briefly.", "cache_control": {"type": "ephemeral", "scope": "global"}}
            ],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Capital of France?",
                            "cache_control": {"type": "ephemeral", "scope": "global"},
                        }
                    ],
                }
            ],
        ),
        model=MODEL,
    ),
    Case(
        "vertex_system_role_message",
        _vx(messages=[{"role": "system", "content": "Answer in one word."}, dict(USER)]),
        model=MODEL,
    ),
    Case(
        "vertex_mid_conversation_system_role",
        _vx(
            messages=[
                dict(USER),
                {"role": "assistant", "content": "Paris"},
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "And Germany?"},
            ]
        ),
        model=MODEL,
    ),
    Case("vertex_output_config_effort_dropped", _vx(output_config={"effort": "high"}), model=MODEL),
    Case("vertex_output_config_effort_kept", _vx(output_config={"effort": "high"}), model=OPUS),
    Case(
        "vertex_output_format_folded",
        _vx(
            output_format={
                "type": "json_schema",
                "schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        ),
        model=OPUS,
    ),
    Case("vertex_output_format_not_an_object", _vx(output_format="json"), upstream=(), model=OPUS),
    Case("vertex_output_format_list", _vx(output_format=["json"]), upstream=(), model=OPUS),
    Case("vertex_output_format_not_an_object_dropped", _vx(output_format="json", drop_params=True), model=OPUS),
    Case(
        "vertex_output_format_wins_over_output_config_format",
        _vx(
            output_format={"type": "json_schema", "schema": {"type": "object"}},
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": {"type": "string"}}},
        ),
        model=OPUS,
    ),
    Case(
        "vertex_billing_metadata_stripped",
        _vx(
            system=[
                {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.0; cc_entrypoint=cli"},
                {"type": "text", "text": "Answer briefly."},
            ]
        ),
        model=MODEL,
    ),
    Case("vertex_metadata_user_id", _vx(metadata={"user_id": "user-1"}), model=MODEL),
    Case("vertex_tools_and_tool_choice", _vx(tools=[dict(WEATHER_TOOL)], tool_choice={"type": "auto"}), model=MODEL),
)

FAILURE_CASES: Final = (
    Case("vertex_upstream_400", _vx(), upstream=(_failure(400, "invalid_request_error", "bad input"),), model=MODEL),
    Case("vertex_upstream_401", _vx(), upstream=(_failure(401, "authentication_error", "bad token"),), model=MODEL),
    Case("vertex_upstream_403", _vx(), upstream=(_failure(403, "permission_error", "denied"),), model=MODEL),
    Case("vertex_upstream_404", _vx(), upstream=(_failure(404, "not_found_error", "no such model"),), model=MODEL),
    Case(
        "vertex_upstream_429",
        _vx(),
        upstream=(_failure(429, "rate_limit_error", "slow down", **{"retry-after": "7"}),),
        model=MODEL,
    ),
    Case("vertex_upstream_500", _vx(), upstream=(_failure(500, "api_error", "internal"),), model=MODEL),
    Case("vertex_upstream_529", _vx(), upstream=(_failure(529, "overloaded_error", "Overloaded"),), model=MODEL),
    Case(
        "vertex_upstream_google_error_shape",
        _vx(),
        upstream=(
            ResponseSpec(
                body={"error": {"code": 400, "message": "Invalid project", "status": "INVALID_ARGUMENT"}}, status=400
            ),
        ),
        model=MODEL,
    ),
    Case(
        "vertex_stream_upstream_429",
        _vx(stream=True),
        upstream=(_failure(429, "rate_limit_error", "slow down"),),
        model=MODEL,
    ),
    Case(
        "vertex_upstream_500_retried",
        _vx(num_retries=1),
        upstream=(_failure(500, "api_error", "internal"), ResponseSpec(body=dict(TEXT_RESPONSE))),
        model=MODEL,
    ),
    Case("vertex_upstream_connection_refused", _vx(api_base="http://127.0.0.1:9"), upstream=(), model=MODEL),
    Case("vertex_mock_response", _vx(mock_response="Hello from the mock"), upstream=(), model=MODEL),
)

CASES: Final = (
    *CALL_CASES,
    *LOCATION_CASES,
    *URL_CASES,
    *CREDENTIAL_CASES,
    *VERSION_CASES,
    *BETA_CASES,
    *BODY_CASES,
    *FAILURE_CASES,
)


@pytest.fixture
def parity_server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.expected_requests = None
    return recording_server


@pytest.fixture
def native_messages() -> Iterator[FallbackProbe]:
    with fallback_probe() as probe:
        yield probe


@pytest.fixture(autouse=True)
def no_ambient_vertex_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _VERTEX_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(litellm, "vertex_project", None)
    monkeypatch.setattr(litellm, "vertex_location", None)


@pytest.mark.asyncio
@pytest.mark.parametrize(("case", "aspect"), tuple(param for case in CASES for param in case.params()))
async def test_native_vertex_messages_match_python(
    case: Case, aspect: str, parity_server: RecordingServer, native_messages: FallbackProbe
) -> None:
    python, native = await _legs(case, parity_server, native_messages)
    assert_aspect_matches(python, native, aspect)


@dataclass(frozen=True, slots=True)
class _Repeated:
    paths: tuple[str, ...]
    texts: tuple[str, ...]
    grants: int


async def _twice(server: RecordingServer, native: bool, name: str) -> _Repeated:
    account: Final = dict(_account(name))
    kwargs: Final = {
        "model": MODEL,
        "max_tokens": 64,
        "messages": [dict(USER)],
        "api_base": server.base_url,
        "vertex_project": PROJECT,
        "vertex_location": LOCATION,
        "vertex_credentials": account,
    }
    call: Final = litellm.anthropic_messages if native else _PYTHON_AMESSAGES
    sent_before: Final = len(server.requests)
    texts: Final[list[str]] = []
    with isolated_callback_registries(), _environment({}):
        for _ in range(2):
            result = cast(Mapping[str, object], await call(**kwargs))
            texts.append(json.dumps(result.get("content")))
    return _Repeated(
        paths=tuple(sent.path for sent in server.requests[sent_before:]),
        texts=tuple(texts),
        grants=_TokenEndpoint.grants.get(f"/token/{name}", 0),
    )


@pytest.mark.asyncio
async def test_both_lanes_exchange_a_service_account_token_as_often(
    parity_server: RecordingServer, native_messages: FallbackProbe
) -> None:
    python: Final = await _twice(parity_server, native=False, name="twice-python")
    native: Final = await _twice(parity_server, native=True, name="twice-native")

    assert native_messages.calls == 0
    assert (native.paths, native.texts) == (python.paths, python.texts)
    assert (python.grants, native.grants) == (1, 1)


async def _through_a_router(
    server: RecordingServer, native: bool, stream: bool
) -> tuple[tuple[Mapping[str, object], ...], str]:
    server.enqueue(_stream(TEXT_STREAM) if stream else ResponseSpec(body=dict(TEXT_RESPONSE)))
    deployment: Final[Mapping[str, object]] = {
        "model": MODEL,
        "api_base": server.base_url,
        "vertex_project": PROJECT,
        "vertex_location": LOCATION,
        "vertex_credentials": json.dumps(dict(_account("router"))),
    }
    sent_before: Final = len(server.requests)
    with contextlib.ExitStack() as stack:
        stack.enter_context(isolated_callback_registries())
        stack.enter_context(_environment({}))
        if not native:
            stack.enter_context(rebound(litellm, "anthropic_messages", _PYTHON_AMESSAGES))
        router: Final = litellm.Router(
            model_list=[
                {"model_name": "claude", "litellm_params": dict(deployment), "model_info": {"id": "dep-vertex"}}
            ],
            num_retries=0,
        )
        result: Final = await router.aanthropic_messages(
            model="claude", max_tokens=64, messages=[dict(USER)], stream=stream
        )
        if stream:
            chunks, failure = await _consume_async(cast(AsyncIterator[object], result))
            assert failure is None
            text = _stream_text(chunks)
        else:
            text = json.dumps(cast(Mapping[str, object], result).get("content"))
        await drain_logging()
    server.responses.clear()
    return (
        tuple(
            {"path": sent.path, "body": sent.body, "anthropic-beta": sent.headers.get("anthropic-beta")}
            for sent in server.requests[sent_before:]
        ),
        text,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["unary", "stream"])
async def test_both_lanes_serve_a_router_vertex_deployment_alike(
    stream: bool, parity_server: RecordingServer, native_messages: FallbackProbe
) -> None:
    python: Final = await _through_a_router(parity_server, native=False, stream=stream)
    native: Final = await _through_a_router(parity_server, native=True, stream=stream)

    assert native_messages.calls == 0
    assert native == python


def test_case_names_are_unique() -> None:
    names: Final = [case.name for case in CASES]
    assert len(names) == len(set(names))
