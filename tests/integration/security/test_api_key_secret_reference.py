from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    list_value,
    object_value,
    string_value,
)
from integration._support.openai_wire import chat_reply, openai_error, responses_reply
from integration._support.process import (
    OwnedProxy,
    WorkerHealthcheck,
    graceful_stop_seconds,
    owned_proxy_process,
)
from integration._support.wire import Reply, Request, wire_server
from integration.security._sinks import Recorder
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(int(2 * graceful_stop_seconds() + 120))

_CLOSE: Final = MappingProxyType({"Connection": "close"})
_FRESH: Final = httpx.Limits(max_keepalive_connections=0)
_CHAT_MODEL: Final = "openai/gpt-4o-mini"
_EMBEDDING_MODEL: Final = "openai/text-embedding-3-small"
_REALTIME_MODEL: Final = "openai/gpt-4o-realtime-preview"
_LISTED_MODEL: Final = "wif-listed-model"
_VECTOR_STORE: Final = "vs_wif_reference"
_SEARCH_PATH: Final = f"/v1/vector_stores/{_VECTOR_STORE}/search"
_BATCH_DEPLOYMENT: Final = "wif-vllm-batch"
_BEDROCK_MODEL: Final = "bedrock/converse/us.anthropic.claude-sonnet-5-5"
_PROVIDER_INPUT_TOKENS: Final = 7919
_UNSUPPORTED_REFERENCE: Final = "oidc/google/https://example.invalid"
_RUN_SECRETS_TOKEN: Final = "/run/secrets/wif-absent-token"
_SOURCES: Final = ("config_credential", "config_inline", "db_credential", "db_inline")
_UNREADABLE: Final = MappingProxyType(
    {"missing_file": "wif-missing", "outside_allowlist": "wif-outside", "unsupported_kind": "wif-unsupported"}
)
_ATTACKS: Final = (
    "client_api_base_inline",
    "client_api_base_credential",
    "body_reference",
    "body_credential_name",
    "user_config",
)
_CREDENTIAL_ROW: Final = (
    'SELECT credential_values::text AS stored, updated_at::text AS updated_at FROM "LiteLLM_CredentialsTable" '
    "WHERE credential_name = %s"
)
_MODEL_ROW: Final = (
    'SELECT litellm_params::text AS stored, updated_at::text AS updated_at FROM "LiteLLM_ProxyModelTable" '
    "WHERE model_id = %s"
)
_ENDPOINTS: Final = MappingProxyType(
    {"embeddings": ("/v1/embeddings", "wif-embedding"), "responses": ("/v1/responses", "wif-config-inline")}
)
_TEAM_WRITES: Final = MappingProxyType(
    {
        ("team_admin", "plain_key"): 200,
        ("team_admin", "names_credential"): 403,
        ("team_admin", "inline_reference"): 400,
        ("proxy_admin", "inline_reference"): 400,
    }
)
_MODEL_ROWS: Final = (
    "SELECT model_id FROM \"LiteLLM_ProxyModelTable\" WHERE model_info::jsonb ->> 'team_public_model_name' = %s"
)


@dataclass(frozen=True, slots=True)
class _Secrets:
    allowed: Path
    token: Path
    missing: Path
    outside: Path

    def current(self) -> str:
        return self.token.read_text()

    def rotate(self) -> str:
        value: Final = f"wif-token-{uuid.uuid4().hex}"
        self.token.write_text(value)
        return value

    def overrides(self) -> Mapping[str, str]:
        return MappingProxyType(
            {"LITELLM_OIDC_ALLOWED_CREDENTIAL_DIRS": str(self.allowed), "PROXY_CONFIG_RELOAD_INTERVAL_SECONDS": "2"}
        )


def _reference(path: Path) -> str:
    return f"oidc/file/{path}"


def _secrets(directory: Path) -> _Secrets:
    allowed: Final = directory / "secrets"
    outside_dir: Final = directory / "outside"
    allowed.mkdir()
    outside_dir.mkdir()
    secrets: Final = _Secrets(
        allowed=allowed, token=allowed / "token", missing=allowed / "absent-token", outside=outside_dir / "token"
    )
    secrets.rotate()
    secrets.outside.write_text(f"outside-secret-{uuid.uuid4().hex}")
    return secrets


def _json_reply(body: Mapping[str, JsonValue]) -> Reply:
    return Reply(body=json.dumps(body).encode())


def _served(request: Request) -> Reply:
    path: Final = request.target.split("?", 1)[0]
    match request.method, path:
        case "POST", "/v1/chat/completions":
            return chat_reply(f"chatcmpl-{uuid.uuid4().hex}", "gpt-4o-mini", "served", stream=False)
        case "POST", "/v1/responses":
            return responses_reply(f"resp_{uuid.uuid4().hex}", "gpt-4o-mini", "served", stream=False)
        case "POST", "/v1/responses/input_tokens":
            return _json_reply({"object": "response.input_tokens", "input_tokens": _PROVIDER_INPUT_TOKENS})
        case "POST", converse if converse.startswith("/model/") and converse.endswith("/converse"):
            return _json_reply(
                {
                    "output": {"message": {"role": "assistant", "content": [{"text": "served"}]}},
                    "stopReason": "end_turn",
                    "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
                }
            )
        case "POST", "/v1/embeddings":
            return _json_reply(
                {
                    "object": "list",
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.25, 0.5]}],
                    "model": "text-embedding-3-small",
                    "usage": {"prompt_tokens": 2, "total_tokens": 2},
                }
            )
        case "GET", "/v1/models":
            return _json_reply(
                {"object": "list", "data": [{"id": _LISTED_MODEL, "object": "model", "created": 0, "owned_by": "it"}]}
            )
        case "POST", searched if searched == _SEARCH_PATH:
            return _json_reply(
                {
                    "object": "vector_store.search_results.page",
                    "search_query": "served",
                    "data": [],
                    "has_more": False,
                    "next_page": None,
                }
            )
        case _:
            return openai_error(404)


def _provider(secrets: _Secrets) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.headers.get("authorization") != f"Bearer {secrets.current()}":
            return openai_error(401)
        return _served(request)

    return respond


def _intruder(request: Request) -> Reply:
    return chat_reply(f"chatcmpl-{uuid.uuid4().hex}", "gpt-4o-mini", "intercepted", stream=False)


def _deployment(
    name: str, params: Mapping[str, JsonValue], info: Mapping[str, JsonValue] = MappingProxyType({})
) -> dict[str, JsonValue]:
    return {"model_name": name, "litellm_params": dict(params), "model_info": dict(info)}


def _credential(name: str, api_key: str, api_base: str) -> dict[str, JsonValue]:
    return {
        "credential_name": name,
        "credential_values": {"api_key": api_key, "api_base": api_base},
        "credential_info": {"custom_llm_provider": "openai"},
    }


def _named(model: str, credential: str) -> dict[str, JsonValue]:
    return {"model": model, "litellm_credential_name": credential}


def _config(directory: Path, secrets: _Secrets, provider: str) -> Path:
    inline: Final = {"api_key": _reference(secrets.token), "api_base": provider}
    configuration: Final[dict[str, JsonValue]] = {
        "model_list": [
            _deployment("wif-config-credential", _named(_CHAT_MODEL, "wif-config")),
            _deployment("wif-config-inline", {"model": _CHAT_MODEL, **inline}),
            _deployment("wif-embedding", _named(_EMBEDDING_MODEL, "wif-config"), {"mode": "embedding"}),
            _deployment("wif-missing", _named(_CHAT_MODEL, "wif-missing")),
            _deployment("wif-outside", _named(_CHAT_MODEL, "wif-outside")),
            _deployment("wif-unsupported", _named(_CHAT_MODEL, "wif-unsupported")),
            _deployment("wif-run-secrets", _named(_CHAT_MODEL, "wif-run-secrets")),
            _deployment("wif-missing-fallback", _named(_CHAT_MODEL, "wif-missing")),
            _deployment("wif-realtime", _named(_REALTIME_MODEL, "wif-config"), {"mode": "realtime"}),
            _deployment("wif-vllm", {"model": "hosted_vllm/wif-local", **inline}, {"id": _BATCH_DEPLOYMENT}),
            _deployment("openai/*", _named("openai/*", "wif-config")),
            _deployment(
                "wif-bedrock-endpoint",
                {
                    "model": _BEDROCK_MODEL,
                    "aws_region_name": "us-east-1",
                    "api_key": _reference(secrets.token),
                    "aws_bedrock_runtime_endpoint": provider.removesuffix("/v1"),
                },
            ),
            _deployment(
                "wif-mistral-passthrough",
                {
                    "model": "mistral/mistral-small-latest",
                    "api_key": _reference(secrets.missing),
                    "use_in_pass_through": True,
                },
            ),
        ],
        "credential_list": [
            _credential("wif-config", _reference(secrets.token), provider),
            _credential("wif-missing", _reference(secrets.missing), provider),
            _credential("wif-outside", _reference(secrets.outside), provider),
            _credential("wif-unsupported", _UNSUPPORTED_REFERENCE, provider),
            _credential("wif-run-secrets", f"oidc/file/{_RUN_SECRETS_TOKEN}", provider),
        ],
        "vector_store_registry": [
            {
                "vector_store_name": "wif-vector-store",
                "litellm_params": {
                    "vector_store_id": _VECTOR_STORE,
                    "custom_llm_provider": "openai",
                    "litellm_credential_name": "wif-config",
                },
            }
        ],
        "router_settings": {
            "fallbacks": [{"wif-missing-fallback": ["wif-config-credential"]}],
            "disable_cooldowns": True,
        },
        "litellm_settings": {"check_provider_endpoint": True},
        "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY", "allow_client_side_credentials": True},
    }
    path: Final = directory / "api_key_secret_reference_config.yaml"
    path.write_text(yaml.safe_dump(configuration, sort_keys=False))
    return path


@dataclass(frozen=True, slots=True)
class _Stored:
    credential: str
    models: Mapping[str, str]
    model_ids: Mapping[str, str]


def _delete_credential(gateway: Gateway, name: str) -> None:
    response: Final = gateway.request("DELETE", f"/credentials/{name}")
    assert response.status_code in (200, 404), response.text


def _created_model(gateway: Gateway, scenario: Scenario, body: Mapping[str, JsonValue]) -> tuple[str, str]:
    created: Final = gateway.post("/model/new", body)
    identity: Final = string_value(object_value(created["model_info"])["id"])
    scenario.cleanups.callback(scenario.delete_model, identity)
    return string_value(created["model_name"]), identity


def _stored(owned: OwnedProxy, scenario: Scenario, secrets: _Secrets, provider: str) -> _Stored:
    credential: Final = f"db-oidc-{uuid.uuid4().hex}"
    owned.gateway.post("/credentials", _credential(credential, _reference(secrets.token), provider))
    scenario.cleanups.callback(_delete_credential, owned.gateway, credential)
    by_credential: Final = _created_model(
        owned.gateway,
        scenario,
        _deployment(f"db-credential-{uuid.uuid4().hex}", _named(_CHAT_MODEL, credential)),
    )
    inline: Final = _created_model(
        owned.gateway,
        scenario,
        _deployment(
            f"db-inline-{uuid.uuid4().hex}",
            {"model": _CHAT_MODEL, "api_key": _reference(secrets.token), "api_base": provider},
        ),
    )
    for name, identity in (by_credential, inline):
        eventually(partial(owned.serves_only_on_every_worker, name, identity), bool, seconds=120)
    return _Stored(
        credential=credential,
        models=MappingProxyType({"db_credential": by_credential[0], "db_inline": inline[0]}),
        model_ids=MappingProxyType({"db_credential": by_credential[1], "db_inline": inline[1]}),
    )


@dataclass(frozen=True, slots=True)
class _Rig:
    owned: OwnedProxy
    provider: Recorder
    intruder: Recorder
    provider_base: str
    intruder_base: str
    secrets: _Secrets
    stored: _Stored
    boot: tuple[Request, ...]
    boot_token: str

    def model(self, source: str) -> str:
        match source:
            case "config_credential":
                return "wif-config-credential"
            case "config_inline":
                return "wif-config-inline"
            case "db_credential" | "db_inline":
                return self.stored.models[source]
            case _:
                pytest.fail(f"unknown source {source}")


def _is_discovery(request: Request) -> bool:
    return (request.method, request.target) == ("GET", "/v1/models")


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("api_key_secret_reference").resolve()
    secrets: Final = _secrets(directory)
    with (
        gateway_from_environment() as gateway,
        wire_server(_provider(secrets)) as provider_wire,
        wire_server(_intruder) as intruder_wire,
    ):
        provider: Final = Recorder(provider_wire)
        provider_base: Final = f"{provider_wire.url}/v1"
        with owned_proxy_process(
            gateway,
            directory,
            {**secrets.overrides(), "MISTRAL_API_BASE": provider_wire.url},
            config=_config(directory, secrets, provider_base),
            workers=2,
            healthcheck=WorkerHealthcheck.LENIENT,
        ) as owned:
            with owned.gateway.scenario() as scenario:
                stored: Final = _stored(owned, scenario, secrets, provider_base)
                boot: Final = eventually(provider.requests, lambda seen: any(_is_discovery(r) for r in seen))
                yield _Rig(
                    owned=owned,
                    provider=provider,
                    intruder=Recorder(intruder_wire),
                    provider_base=provider_base,
                    intruder_base=f"{intruder_wire.url}/v1",
                    secrets=secrets,
                    stored=stored,
                    boot=boot,
                    boot_token=secrets.current(),
                )


def _marker() -> str:
    return f"wif-marker-{uuid.uuid4().hex}"


def _chat(
    gateway: Gateway,
    model: str,
    marker: str,
    extra: Mapping[str, JsonValue] = MappingProxyType({}),
    *,
    key: str | None = None,
) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": f"say {marker}"}], **extra},
        key=key,
        headers=_CLOSE,
    )


def _carrying(requests: Sequence[Request], marker: str) -> tuple[Request, ...]:
    return tuple(request for request in requests if marker in request.body.decode(errors="replace"))


def _bearers(requests: Sequence[Request]) -> tuple[str | None, ...]:
    return tuple(request.headers.get("authorization") for request in requests)


def _anywhere(requests: Sequence[Request], value: str) -> bool:
    return any(
        value in json.dumps(dict(request.headers)) + request.body.decode(errors="replace") for request in requests
    )


def _reached_on_worker(rig: _Rig, model: str, idle: int) -> tuple[int, tuple[Request, ...]]:
    marker: Final = _marker()
    gateway: Final = rig.owned.gateway
    with (
        rig.owned.suspended(idle),
        httpx.Client(base_url=gateway.client.base_url, timeout=60, trust_env=False, limits=_FRESH) as client,
    ):
        response: Final = client.post(
            "/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": f"say {marker}"}]},
            headers={"Authorization": f"Bearer {gateway.key}"},
        )
    return response.status_code, _carrying(rig.provider.requests(), marker)


def _chat_on_worker(rig: _Rig, model: str, idle: int) -> tuple[int, tuple[str | None, ...]]:
    status, reached = _reached_on_worker(rig, model, idle)
    return status, _bearers(reached)


@pytest.mark.parametrize("source", _SOURCES)
def test_admin_reference_sends_the_current_token_from_every_worker(rig: _Rig, source: str) -> None:
    first, second = sorted(eventually(rig.owned.worker_pids, lambda pids: len(pids) == 2))
    model: Final = rig.model(source)
    before: Final = rig.secrets.current()
    sent_before: Final = (_chat_on_worker(rig, model, second), _chat_on_worker(rig, model, first))
    rotated: Final = rig.secrets.rotate()
    sent_after: Final = (_chat_on_worker(rig, model, second), _chat_on_worker(rig, model, first))
    assert sent_before == ((200, (f"Bearer {before}",)),) * 2, sent_before
    assert sent_after == ((200, (f"Bearer {rotated}",)),) * 2, sent_after


@pytest.mark.parametrize("endpoint", tuple(_ENDPOINTS))
def test_other_call_types_send_the_current_token(rig: _Rig, endpoint: str) -> None:
    marker: Final = _marker()
    path, model = _ENDPOINTS[endpoint]
    response: Final = rig.owned.gateway.request(
        "POST", path, {"model": model, "input": f"say {marker}"}, headers=_CLOSE
    )
    sent: Final = _bearers(_carrying(rig.provider.requests(), marker))
    assert (response.status_code, sent) == (200, (f"Bearer {rig.secrets.current()}",)), response.text


@pytest.mark.parametrize("shape", ("credential_name", "inline"))
def test_connection_with_an_admin_reference_reaches_the_provider(rig: _Rig, shape: str) -> None:
    params: Final[dict[str, JsonValue]] = (
        _named(_CHAT_MODEL, "wif-config")
        if shape == "credential_name"
        else {"model": _CHAT_MODEL, "api_key": _reference(rig.secrets.token), "api_base": rig.provider_base}
    )
    seen: Final = len(rig.provider.requests())
    response: Final = rig.owned.gateway.request(
        "POST", "/health/test_connection", {"litellm_params": params, "mode": "chat"}, headers=_CLOSE
    )
    sent: Final = frozenset(_bearers(rig.provider.requests()[seen:]))
    assert response.status_code == 200, response.text
    assert JSON_OBJECT.validate_json(response.content)["status"] == "success", response.text
    assert sent == {f"Bearer {rig.secrets.current()}"}, sent


@pytest.mark.parametrize("case", tuple(_UNREADABLE))
def test_unreadable_reference_fails_closed_before_any_upstream_call(rig: _Rig, case: str) -> None:
    marker: Final = _marker()
    response: Final = _chat(rig.owned.gateway, _UNREADABLE[case], marker)
    outside_secret: Final = rig.secrets.outside.read_text()
    reached: Final = _carrying(rig.provider.requests(), marker)
    assert (response.status_code, reached) == (401, ()), (response.text, _bearers(reached))
    assert outside_secret not in response.text
    assert not _anywhere(rig.provider.requests(), outside_secret)


def test_unreadable_reference_falls_over_to_the_next_deployment(rig: _Rig) -> None:
    marker: Final = _marker()
    response: Final = _chat(rig.owned.gateway, "wif-missing-fallback", marker)
    sent: Final = frozenset(_bearers(_carrying(rig.provider.requests(), marker)))
    assert (response.status_code, sent) == (200, {f"Bearer {rig.secrets.current()}"}), response.text


def test_local_token_count_does_not_need_an_unreadable_reference(rig: _Rig) -> None:
    marker: Final = _marker()
    response: Final = rig.owned.gateway.request(
        "POST",
        "/utils/token_counter",
        {"model": _UNREADABLE["missing_file"], "messages": [{"role": "user", "content": f"say {marker}"}]},
        headers=_CLOSE,
    )
    assert response.status_code == 200, response.text
    counted: Final = JSON_OBJECT.validate_json(response.content)["total_tokens"]
    assert isinstance(counted, int) and counted > 0, response.text
    assert _carrying(rig.provider.requests(), marker) == ()


def test_input_token_count_with_an_unreadable_reference_fails_closed(rig: _Rig) -> None:
    marker: Final = _marker()
    response: Final = rig.owned.gateway.request(
        "POST",
        "/v1/responses/input_tokens",
        {"model": _UNREADABLE["missing_file"], "input": f"say {marker}"},
        headers=_CLOSE,
    )
    reached: Final = _carrying(rig.provider.requests(), marker)
    assert (response.status_code, reached) == (401, ()), (response.text, _bearers(reached))
    assert str(rig.secrets.allowed) not in response.text, response.text


def test_messages_with_an_unreadable_reference_hides_the_token_path(rig: _Rig) -> None:
    marker: Final = _marker()
    response: Final = rig.owned.gateway.request(
        "POST",
        "/v1/messages",
        {
            "model": _UNREADABLE["missing_file"],
            "max_tokens": 16,
            "messages": [{"role": "user", "content": f"say {marker}"}],
        },
        headers=_CLOSE,
    )
    reached: Final = _carrying(rig.provider.requests(), marker)
    assert (response.status_code, reached) == (401, ()), (response.text, _bearers(reached))
    assert str(rig.secrets.allowed) not in response.text, response.text


def test_unreadable_reference_under_run_secrets_hides_the_token_path(rig: _Rig) -> None:
    marker: Final = _marker()
    response: Final = _chat(rig.owned.gateway, "wif-run-secrets", marker)
    reached: Final = _carrying(rig.provider.requests(), marker)
    assert (response.status_code, reached) == (401, ()), (response.text, _bearers(reached))
    assert _RUN_SECRETS_TOKEN not in response.text, response.text


@pytest.mark.parametrize("source", ("config_credential", "config_inline"))
def test_provider_token_count_sends_the_current_token(rig: _Rig, source: str) -> None:
    marker: Final = _marker()
    response: Final = rig.owned.gateway.request(
        "POST", "/v1/responses/input_tokens", {"model": rig.model(source), "input": f"say {marker}"}, headers=_CLOSE
    )
    sent: Final = _bearers(_carrying(rig.provider.requests(), marker))
    assert (response.status_code, sent) == (200, (f"Bearer {rig.secrets.current()}",)), response.text
    assert JSON_OBJECT.validate_json(response.content)["input_tokens"] == _PROVIDER_INPUT_TOKENS, response.text


def test_pass_through_with_an_unreadable_reference_hides_the_token_path(rig: _Rig) -> None:
    marker: Final = _marker()
    response: Final = rig.owned.gateway.request(
        "POST",
        "/mistral/v1/chat/completions",
        {"model": "mistral-small-latest", "messages": [{"role": "user", "content": f"say {marker}"}]},
        headers=_CLOSE,
    )
    assert (response.status_code, _carrying(rig.provider.requests(), marker)) == (401, ()), response.text
    assert str(rig.secrets.allowed) not in response.text, response.text


def test_admin_configured_endpoint_receives_the_current_token(rig: _Rig) -> None:
    marker: Final = _marker()
    response: Final = _chat(rig.owned.gateway, "wif-bedrock-endpoint", marker)
    sent: Final = _bearers(_carrying(rig.provider.requests(), marker))
    assert (response.status_code, sent) == (200, (f"Bearer {rig.secrets.current()}",)), response.text


def _hosts_and_bearers(requests: Sequence[Request]) -> tuple[tuple[str | None, str | None], ...]:
    return tuple((request.headers.get("host"), request.headers.get("authorization")) for request in requests)


def test_updated_api_base_carries_the_token_to_the_new_host(rig: _Rig) -> None:
    first, second = sorted(eventually(rig.owned.worker_pids, lambda pids: len(pids) == 2))
    moved_base: Final = rig.provider_base.replace("127.0.0.1", "localhost")
    with rig.owned.gateway.scenario() as scenario:
        name, identity = _created_model(
            rig.owned.gateway,
            scenario,
            _deployment(
                f"db-moved-{uuid.uuid4().hex}",
                {"model": _CHAT_MODEL, "api_key": _reference(rig.secrets.token), "api_base": rig.provider_base},
            ),
        )
        eventually(partial(rig.owned.serves_only_on_every_worker, name, identity), bool, seconds=120)
        patched: Final = rig.owned.gateway.request(
            "PATCH", f"/model/{identity}/update", {"litellm_params": {"api_base": moved_base}}
        )
        assert patched.status_code == 200, patched.text
        expected: Final = ((200, ((httpx.URL(moved_base).netloc.decode(), f"Bearer {rig.secrets.current()}"),)),) * 2
        reached: Final = eventually(
            lambda: tuple(
                (status, _hosts_and_bearers(requests))
                for status, requests in (_reached_on_worker(rig, name, idle) for idle in (second, first))
            ),
            lambda seen: seen == expected,
            seconds=120,
            return_last_on_timeout=True,
        )
        assert reached == expected, reached


def _attack(rig: _Rig, attack: str) -> tuple[str, dict[str, JsonValue]]:
    redirected: Final[dict[str, JsonValue]] = {"api_base": rig.intruder_base}
    match attack:
        case "client_api_base_inline":
            return "wif-config-inline", redirected
        case "client_api_base_credential":
            return "wif-config-credential", redirected
        case "body_reference":
            return "wif-config-inline", {**redirected, "api_key": _reference(rig.secrets.token)}
        case "body_credential_name":
            return "wif-config-inline", {**redirected, "litellm_credential_name": "wif-config"}
        case "user_config":
            injected: Final = {"model": _CHAT_MODEL, "api_key": _reference(rig.secrets.token), **redirected}
            return "wif-injected", {
                "user_config": {"model_list": [{"model_name": "wif-injected", "litellm_params": injected}]}
            }
        case _:
            pytest.fail(f"unknown attack {attack}")


@pytest.mark.parametrize("attack", _ATTACKS)
def test_caller_cannot_send_the_admin_token_to_its_own_host(rig: _Rig, attack: str) -> None:
    marker: Final = _marker()
    model, extra = _attack(rig, attack)
    response: Final = _chat(rig.owned.gateway, model, marker, extra)
    intruded: Final = rig.intruder.requests()
    assert not _anywhere(intruded, rig.secrets.current())
    assert _carrying(intruded, marker) == (), _bearers(_carrying(intruded, marker))
    assert _carrying(rig.provider.requests(), marker) == ()
    assert response.status_code in (400, 401), response.text


def test_caller_with_its_own_key_still_reaches_its_own_host(rig: _Rig) -> None:
    marker: Final = _marker()
    own_key: Final = f"sk-caller-{uuid.uuid4().hex}"
    response: Final = _chat(
        rig.owned.gateway, "wif-config-inline", marker, {"api_base": rig.intruder_base, "api_key": own_key}
    )
    intruded: Final = rig.intruder.requests()
    assert (response.status_code, _bearers(_carrying(intruded, marker))) == (200, (f"Bearer {own_key}",)), response.text
    assert not _anywhere(intruded, rig.secrets.current())


def _team_key(scenario: Scenario, team_id: str, role: str) -> str:
    return scenario.key(user_id=scenario.member(team_id, role=role), team_id=team_id)


def _team_body(name: str, team_id: str, params: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return _deployment(name, {"model": _CHAT_MODEL, **params}, {"team_id": team_id})


def _team_write_params(rig: _Rig, shape: str) -> dict[str, JsonValue]:
    match shape:
        case "plain_key":
            return {"api_key": f"sk-team-{uuid.uuid4().hex}", "api_base": rig.provider_base}
        case "names_credential":
            return {"litellm_credential_name": "wif-config"}
        case "inline_reference":
            return {"api_key": _reference(rig.secrets.token), "api_base": rig.provider_base}
        case _:
            pytest.fail(f"unknown shape {shape}")


def _delete_written(scenario: Scenario, response: httpx.Response) -> None:
    if response.status_code == 200:
        scenario.delete_model(
            string_value(object_value(JSON_OBJECT.validate_json(response.content)["model_info"])["id"])
        )


@pytest.mark.parametrize(("writer", "shape"), tuple(_TEAM_WRITES))
def test_team_model_write_carrying_a_reference_is_refused(rig: _Rig, writer: str, shape: str) -> None:
    with rig.owned.gateway.scenario() as scenario:
        team_id: Final = scenario.team()
        name: Final = f"wif-team-{uuid.uuid4().hex}"
        key: Final = None if writer == "proxy_admin" else _team_key(scenario, team_id, "admin")
        response: Final = rig.owned.gateway.request(
            "POST", "/model/new", _team_body(name, team_id, _team_write_params(rig, shape)), key=key
        )
        stored: Final = len(rig.owned.gateway.rows(_MODEL_ROWS, (name,)))
        _delete_written(scenario, response)
        expected: Final = _TEAM_WRITES[(writer, shape)]
        assert (response.status_code, stored) == (expected, int(expected == 200)), response.text


@pytest.mark.parametrize("host", ("credential_host", "other_host"))
def test_team_row_naming_the_credential_sends_the_token_only_to_its_host(rig: _Rig, host: str) -> None:
    redirected: Final[dict[str, JsonValue]] = {} if host == "credential_host" else {"api_base": rig.intruder_base}
    with rig.owned.gateway.scenario() as scenario:
        team_id: Final = scenario.team()
        name: Final = f"wif-team-{uuid.uuid4().hex}"
        stored_name, identity = _created_model(
            rig.owned.gateway,
            scenario,
            _team_body(name, team_id, {"litellm_credential_name": "wif-config", **redirected}),
        )
        eventually(lambda: rig.owned.serves_only_on_every_worker(stored_name, identity), bool, seconds=120)
        marker: Final = _marker()
        response: Final = _chat(rig.owned.gateway, name, marker, key=_team_key(scenario, team_id, "user"))
        intruded: Final = rig.intruder.requests()
        sent: Final = _bearers(_carrying(rig.provider.requests(), marker))
        assert not _anywhere(intruded, rig.secrets.current())
        assert _carrying(intruded, marker) == ()
        expected: Final = (200, (f"Bearer {rig.secrets.current()}",)) if host == "credential_host" else (401, ())
        assert (response.status_code, sent) == expected, response.text


def test_startup_discovery_sends_the_boot_token(rig: _Rig) -> None:
    sent: Final = frozenset(_bearers(tuple(request for request in rig.boot if _is_discovery(request))))
    assert sent == {f"Bearer {rig.boot_token}"}, sent


def _side_call(rig: _Rig, seam: str, marker: str) -> tuple[httpx.Response, Callable[[Request], bool]]:
    gateway: Final = rig.owned.gateway
    match seam:
        case "realtime_health":
            return (
                gateway.request("GET", "/health", params={"model": "wif-realtime"}, headers=_CLOSE),
                lambda request: request.target.startswith("/v1/realtime"),
            )
        case "vector_store_search":
            return (
                gateway.request(
                    "POST", f"/v1/vector_stores/{_VECTOR_STORE}/search", {"query": f"say {marker}"}, headers=_CLOSE
                ),
                lambda request: marker in request.body.decode(errors="replace"),
            )
        case "batch_probe":
            return (
                gateway.request(
                    "POST",
                    "/v1/batches",
                    {
                        "input_file_id": f"file-{marker}",
                        "endpoint": "/v1/chat/completions",
                        "completion_window": "24h",
                        "model": _BATCH_DEPLOYMENT,
                    },
                    headers=_CLOSE,
                ),
                lambda request: request.target == "/v1/files",
            )
        case _:
            pytest.fail(f"unknown seam {seam}")


_SIDE_STATUS: Final = MappingProxyType({"realtime_health": 503, "vector_store_search": 200, "batch_probe": 400})


@pytest.mark.parametrize("seam", tuple(_SIDE_STATUS))
def test_calls_outside_the_client_wrapper_send_the_current_token(rig: _Rig, seam: str) -> None:
    seen: Final = len(rig.provider.requests())
    response, matches = _side_call(rig, seam, _marker())
    sent: Final = frozenset(_bearers(tuple(request for request in rig.provider.requests()[seen:] if matches(request))))
    assert (response.status_code, sent) == (_SIDE_STATUS[seam], {f"Bearer {rig.secrets.current()}"}), response.text


def test_wildcard_listing_shows_the_models_the_provider_lists(rig: _Rig) -> None:
    response: Final = rig.owned.gateway.request("GET", "/v1/models", headers=_CLOSE)
    assert response.status_code == 200, response.text
    entries: Final = list_value(JSON_OBJECT.validate_json(response.content)["data"])
    listed: Final = frozenset(string_value(object_value(entry)["id"]) for entry in entries)
    asked: Final = frozenset(_bearers(tuple(request for request in rig.provider.requests() if _is_discovery(request))))
    assert any(_LISTED_MODEL in identity for identity in listed), sorted(listed)
    assert not any(bearer.startswith("Bearer oidc/") for bearer in asked if bearer is not None), asked


def _stored_rows(rig: _Rig) -> tuple[list[dict[str, JsonValue]], ...]:
    return (
        rig.owned.gateway.rows(_CREDENTIAL_ROW, (rig.stored.credential,)),
        *(rig.owned.gateway.rows(_MODEL_ROW, (identity,)) for identity in rig.stored.model_ids.values()),
    )


def test_resolved_token_is_never_written_back(rig: _Rig) -> None:
    before: Final = _stored_rows(rig)
    calls: Final = tuple(
        _chat(rig.owned.gateway, rig.model(source), _marker()).status_code for source in ("db_credential", "db_inline")
    )
    after: Final = _stored_rows(rig)
    assert tuple(len(rows) for rows in before) == (1, 1, 1), before
    assert after == before, (calls, before, after)
