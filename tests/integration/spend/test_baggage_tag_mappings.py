import itertools
import json
import uuid
from collections.abc import Generator, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal

import httpx
import pytest
import yaml
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    eventually,
    gateway_from_environment,
    list_value,
    object_value,
)
from integration._support.database import scratch_database
from integration._support.database_relay import dropped_connection_relay, plaintext_database_url
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter
from websockets.exceptions import InvalidStatus
from websockets.sync.client import connect

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

PROXY_CONFIG: Final = Path(__file__).resolve().parents[1] / "proxy_config.yaml"
PRICED: Final[dict[str, JsonValue]] = {"input_cost_per_token": 0.01, "output_cost_per_token": 0.01}
WORKER_PROBES: Final = 6
JSON_LIST: Final = TypeAdapter(list[JsonValue])
GUARDRAIL_VERDICT: Final = b'{"action":"GUARDRAIL_INTERVENED","texts":["masked baggage prompt"]}'
GUARDRAIL_PATH: Final = "/beta/litellm_basic_guardrail_api"
NO_FIELDS: Final[Mapping[str, JsonValue]] = MappingProxyType({})

Route = Literal["chat", "chat_litellm_metadata", "messages", "responses"]
CallerTag = Literal["none", "header", "metadata_json_with_header", "litellm_metadata"]
RejectedClientTag = Literal["none", "header", "body", "metadata_json", "litellm_metadata"]


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    provider: Wire
    callbacks: Wire
    marker: str

    @property
    def chat_model(self) -> str:
        return f"chat-{self.marker}"

    @property
    def exact_tag(self) -> str:
        return f"usecase-a-{self.marker}"

    def usecase_tag(self, value: str) -> str:
        return f"usecase-{self.marker}:{value}"

    def tenant_tag(self, value: str) -> str:
        return f"tenant-{self.marker}:{value}"

    def budgeted_tag(self, value: str) -> str:
        return f"budgeted-{self.marker}:{value}"

    def team_tag(self, value: str) -> str:
        return f"team-{self.marker}:{value}"

    @property
    def guarded_tag(self) -> str:
        return f"guarded-{self.marker}:on"

    @property
    def policy_tag(self) -> str:
        return f"policy-{self.marker}:sensitive"


def _chat_reply() -> Reply:
    marker: Final = uuid.uuid4().hex
    return Reply(
        body=json.dumps(
            {
                "id": f"chatcmpl-{marker}",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": marker}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            }
        ).encode()
    )


def _responses_reply() -> Reply:
    marker: Final = uuid.uuid4().hex
    return Reply(
        body=json.dumps(
            {
                "id": f"resp_{marker}",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "gpt-4o-mini",
                "output": [
                    {
                        "id": f"msg_{marker}",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": marker, "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
            }
        ).encode()
    )


def _provider(request: Request) -> Reply:
    if request.target == GUARDRAIL_PATH:
        return Reply(body=b'{"action":"NONE"}')
    if request.target == "/v1/responses":
        return _responses_reply()
    return _chat_reply()


def _deployment(model_name: str, provider: Wire, **parameters: JsonValue) -> dict[str, JsonValue]:
    return {
        "model_name": model_name,
        "litellm_params": {
            "model": "openai/gpt-4o-mini",
            "api_base": f"{provider.url}/v1",
            "api_key": "synthetic-baggage-key",
            **PRICED,
            **parameters,
        },
    }


def _write_config(
    directory: Path,
    model_list: list[JsonValue],
    general_settings: Mapping[str, JsonValue],
    litellm_settings: Mapping[str, JsonValue],
    guardrails: Sequence[JsonValue] = (),
) -> Path:
    base: Final = JSON_OBJECT.validate_python(yaml.safe_load(PROXY_CONFIG.read_text()))
    config: Final = {
        **base,
        "model_list": model_list,
        "general_settings": {**object_value(base["general_settings"]), **general_settings},
        "litellm_settings": {**object_value(base["litellm_settings"]), **litellm_settings},
        **({"guardrails": list(guardrails)} if guardrails else {}),
    }
    path: Final = directory / "baggage-tag-mappings.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _routed_deployment(marker: str, provider: Wire, name: str, tag: str) -> dict[str, JsonValue]:
    return {
        **_deployment(f"routed-{marker}", provider, tags=[tag]),
        "model_info": {"id": f"{name}-{marker}", "enable_tag_filtering": True},
    }


def _mapped_models(marker: str, provider: Wire, upstream_url: str) -> list[JsonValue]:
    routed: Final = (
        _routed_deployment(marker, provider, "tagged", f"usecase-a-{marker}"),
        _routed_deployment(marker, provider, "default", "default"),
    )
    realtime: Final[dict[str, JsonValue]] = {
        "model_name": f"realtime-{marker}",
        "litellm_params": {
            "model": "openai/gpt-realtime-2.1",
            "api_base": f"{upstream_url}/v1",
            "api_key": f"realtime-{marker}",
        },
    }
    guarded: Final = _deployment(f"guarded-{marker}", provider, tags=[f"guarded-{marker}:on"])
    pre_guarded: Final = _deployment(f"guarded-pre-{marker}", provider, tags=[f"guarded-{marker}:pre"])
    return [_deployment(f"chat-{marker}", provider), guarded, pre_guarded, *routed, realtime]


def _mapped_rules(marker: str, *, tenant_required: bool) -> list[JsonValue]:
    return [
        {"key": "entrypoint", "value": "a", "tag": f"usecase-a-{marker}"},
        {"key": "entrypoint", "tag_prefix": f"usecase-{marker}:"},
        {"key": "tenant", "tag_prefix": f"tenant-{marker}:", "required": tenant_required},
        {"key": "budgeted", "tag_prefix": f"budgeted-{marker}:", "require_budgeted_tag": True},
        {"key": "team+id", "tag_prefix": f"team-{marker}:"},
        {"key": "guard", "tag_prefix": f"guarded-{marker}:"},
    ]


def _deployment_tag_mode_guardrail(marker: str, provider: Wire) -> dict[str, JsonValue]:
    return {
        "guardrail_name": f"deployment-tag-mode-{marker}",
        "litellm_params": {
            "guardrail": "generic_guardrail_api",
            "mode": {
                "tags": {
                    f"guarded-{marker}:on": "post_call",
                    f"guarded-{marker}:pre": "pre_call",
                    f"policy-{marker}:sensitive": "post_call",
                }
            },
            "default_on": True,
            "api_base": f"{provider.url}{GUARDRAIL_PATH}",
            "api_key": "synthetic-guardrail-key",
        },
    }


@contextmanager
def _rig(
    tmp_path_factory: pytest.TempPathFactory,
    general_settings: Mapping[str, JsonValue],
    *,
    tenant_required: bool,
    workers: int,
    tag_mode_guardrail: bool = False,
) -> Generator[Rig]:
    marker: Final = uuid.uuid4().hex[:12]
    directory: Final = tmp_path_factory.mktemp("baggage-tag-mappings")
    with (
        gateway_from_environment() as base,
        wire_server(_provider) as provider,
        wire_server(lambda _request: Reply()) as callbacks,
    ):
        config: Final = _write_config(
            directory,
            _mapped_models(marker, provider, base.upstream_url),
            {"request_tag_mappings": _mapped_rules(marker, tenant_required=tenant_required), **general_settings},
            {"callbacks": ["generic_api"], "DEFAULT_FLUSH_INTERVAL_SECONDS": 1},
            [_deployment_tag_mode_guardrail(marker, provider)] if tag_mode_guardrail else (),
        )
        with owned_proxy(
            base, directory, {"GENERIC_LOGGER_ENDPOINT": callbacks.url}, config=config, workers=workers
        ) as proxy:
            yield Rig(proxy, provider, callbacks, marker)


@pytest.fixture(scope="module")
def mapped(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with _rig(tmp_path_factory, {}, tenant_required=True, workers=2) as rig:
        yield rig


@pytest.fixture(scope="module")
def optional(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with _rig(tmp_path_factory, {}, tenant_required=False, workers=1, tag_mode_guardrail=True) as rig:
        yield rig


@pytest.fixture(scope="module")
def rejecting(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with _rig(tmp_path_factory, {"reject_clientside_metadata_tags": True}, tenant_required=False, workers=1) as rig:
        yield rig


def _digest(key: str) -> str:
    return sha256(key.encode()).hexdigest()


def _caller_tags(raw: JsonValue) -> frozenset[str]:
    tags: Final = JSON_LIST.validate_json(raw) if isinstance(raw, str) else list_value(raw)
    return frozenset(tag for tag in tags if isinstance(tag, str) and not tag.startswith("User-Agent: "))


def _spend_tags(proxy: Gateway, key: str) -> frozenset[str]:
    rows: Final = eventually(
        lambda: proxy.rows('SELECT request_tags FROM "LiteLLM_SpendLogs" WHERE api_key=%s', (_digest(key),)),
        lambda values: len(values) == 1,
        seconds=70,
    )
    return _caller_tags(rows[0]["request_tags"])


def _batch_events(batch: Request) -> Iterator[dict[str, JsonValue]]:
    return (object_value(item) for item in JSON_LIST.validate_json(batch.body))


def _logged_event(callbacks: Wire, call_id: str) -> dict[str, JsonValue]:
    def delivered() -> tuple[dict[str, JsonValue], ...]:
        events: Final = itertools.chain.from_iterable(_batch_events(batch) for batch in callbacks.drain())
        return tuple(event for event in events if event.get("litellm_call_id") == call_id)

    events: Final = eventually(delivered, lambda values: len(values) == 1, seconds=30)
    return events[0]


def _inference_body(route: Route, model: str, text: str, caller_tag: str) -> dict[str, JsonValue]:
    match route:
        case "chat":
            return {"model": model, "messages": [{"role": "user", "content": text}], "metadata": {"tags": [caller_tag]}}
        case "chat_litellm_metadata":
            return {
                "model": model,
                "messages": [{"role": "user", "content": text}],
                "litellm_metadata": {"tags": [caller_tag]},
            }
        case "messages":
            return {
                "model": model,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": text}],
                "litellm_metadata": {"tags": [caller_tag]},
            }
        case "responses":
            return {"model": model, "input": text, "litellm_metadata": {"tags": [caller_tag]}}


ROUTE_PATHS: Final[Mapping[Route, str]] = {
    "chat": "/v1/chat/completions",
    "chat_litellm_metadata": "/v1/chat/completions",
    "messages": "/v1/messages",
    "responses": "/v1/responses",
}


def _guardrail_inputs(rig: Rig) -> tuple[JsonValue, ...]:
    return tuple(
        JSON_OBJECT.validate_json(request.body)["input_type"]
        for request in rig.provider.drain()
        if request.target == GUARDRAIL_PATH
    )


def _chat(
    rig: Rig,
    key: str,
    baggage: str | None,
    *,
    model: str | None = None,
    fields: Mapping[str, JsonValue] = NO_FIELDS,
    **headers: str,
) -> httpx.Response:
    return rig.proxy.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model or rig.chat_model,
            "messages": [{"role": "user", "content": f"baggage {uuid.uuid4().hex}"}],
            "max_tokens": 16,
            **fields,
        },
        key=key,
        headers={**({"baggage": baggage} if baggage is not None else {}), **headers},
    )


@pytest.mark.parametrize("route", ["chat", "chat_litellm_metadata", "messages", "responses"])
def test_mapped_baggage_tags_reach_spend_and_callbacks_but_not_requester_metadata(mapped: Rig, route: Route) -> None:
    tenant: Final = uuid.uuid4().hex
    caller_tag: Final = f"caller-{tenant}"
    mapped_tags: Final = frozenset({mapped.exact_tag, mapped.tenant_tag(tenant)})
    with mapped.proxy.scenario() as scenario:
        key: Final = scenario.key()
        mapped.provider.drain()
        response: Final = mapped.proxy.request(
            "POST",
            ROUTE_PATHS[route],
            _inference_body(route, mapped.chat_model, f"baggage {tenant}", caller_tag),
            key=key,
            headers={"baggage": f"entrypoint=a, tenant={tenant}"},
        )
        assert response.status_code == 200, response.text
        assert len(mapped.provider.drain()) == 1
        assert _spend_tags(mapped.proxy, key) == mapped_tags | {caller_tag}
        event: Final = _logged_event(mapped.callbacks, response.headers["x-litellm-call-id"])
        assert mapped_tags <= _caller_tags(event["request_tags"]), event["request_tags"]
        requester: Final = object_value(object_value(event["metadata"]).get("requester_metadata") or {})
        assert not mapped_tags & _caller_tags(requester.get("tags") or []), requester


def test_json_string_metadata_gains_exactly_the_mapped_tags(optional: Rig) -> None:
    tenant: Final = uuid.uuid4().hex
    mapped_tags: Final = frozenset({optional.exact_tag, optional.tenant_tag(tenant)})
    fields: Final[Mapping[str, JsonValue]] = {"metadata": json.dumps({"tags": [f"caller-{tenant}"]})}
    with optional.proxy.scenario() as scenario:
        plain_key: Final = scenario.key()
        key: Final = scenario.key()
        optional.provider.drain()
        plain: Final = _chat(optional, plain_key, None, fields=fields)
        response: Final = _chat(optional, key, f"entrypoint=a,tenant={tenant}", fields=fields)
        assert (plain.status_code, response.status_code) == (200, 200), (plain.text, response.text)
        assert len(optional.provider.drain()) == 2
        assert _spend_tags(optional.proxy, key) == _spend_tags(optional.proxy, plain_key) | mapped_tags
        event: Final = _logged_event(optional.callbacks, response.headers["x-litellm-call-id"])
        assert mapped_tags <= _caller_tags(event["request_tags"]), event["request_tags"]
        requester: Final = object_value(object_value(event["metadata"]).get("requester_metadata") or {})
        assert not mapped_tags & _caller_tags(requester.get("tags") or []), requester


def test_a2a_spend_row_carries_the_mapped_tags(mapped: Rig) -> None:
    marker: Final = "baggagea2a" + uuid.uuid4().hex[:12]

    def card(url: str) -> dict[str, JsonValue]:
        return {
            "protocolVersion": "0.3",
            "name": marker,
            "description": "Synthetic peer",
            "version": "1.0.0",
            "url": url + "/",
            "capabilities": {"streaming": True},
            "defaultInputModes": ["text"],
            "defaultOutputModes": ["text"],
            "skills": [],
        }

    def peer(request: Request) -> Reply:
        if request.method == "GET":
            return Reply(body=json.dumps(card(wire.url)).encode())
        rpc: Final = JSON_OBJECT.validate_json(request.body)
        return Reply(
            body=json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": rpc["id"],
                    "result": {
                        "kind": "message",
                        "role": "agent",
                        "messageId": f"{marker}-out",
                        "parts": [{"kind": "text", "text": "done"}],
                    },
                }
            ).encode()
        )

    tenant: Final = uuid.uuid4().hex
    with wire_server(peer) as wire, mapped.proxy.scenario() as scenario:
        created: Final = mapped.proxy.post(
            "/v1/agents",
            {"agent_name": marker, "agent_card_params": card(wire.url), "litellm_params": {"cost_per_query": 0.25}},
        )
        agent: Final = str(created["agent_id"])
        scenario.cleanups.callback(lambda: mapped.proxy.request("DELETE", f"/v1/agents/{agent}"))
        key: Final = scenario.key()
        sent: Final = mapped.proxy.request(
            "POST",
            f"/a2a/{agent}",
            {
                "jsonrpc": "2.0",
                "id": f"{marker}-send",
                "method": "message/send",
                "params": {
                    "message": {
                        "kind": "message",
                        "role": "user",
                        "messageId": f"{marker}-in",
                        "parts": [{"kind": "text", "text": "ping"}],
                    }
                },
            },
            key=key,
            headers={"baggage": f"entrypoint=a,tenant={tenant}"},
        )
        assert sent.status_code == 200, sent.text
        assert {mapped.exact_tag, mapped.tenant_tag(tenant)} <= _spend_tags(mapped.proxy, key)


@pytest.mark.parametrize(
    ("key_fields", "fields", "headers"),
    [
        ({}, NO_FIELDS, {}),
        ({}, {"metadata": {"tags": "run-123"}}, {"x-litellm-tags": "caller-tag"}),
        ({"metadata": {"tags": ["key-tag"]}}, {"metadata": {"tags": "run-123"}}, {}),
        ({}, {"metadata": {"tags": ""}}, {}),
    ],
    ids=["plain", "string_tags_with_header_tag", "string_tags_with_key_tag", "empty_string_tags"],
)
def test_spent_mapped_tag_budget_rejects_the_next_request_before_the_provider(
    mapped: Rig, key_fields: Mapping[str, JsonValue], fields: Mapping[str, JsonValue], headers: Mapping[str, str]
) -> None:
    entry: Final = uuid.uuid4().hex
    tag: Final = mapped.usecase_tag(entry)
    baggage: Final = f"entrypoint={entry},tenant={entry}"
    with mapped.proxy.scenario() as scenario:
        key: Final = scenario.key(**key_fields)
        mapped.proxy.post("/tag/new", {"name": tag, "max_budget": 0.0001})
        scenario.cleanups.callback(lambda: mapped.proxy.request("POST", "/tag/delete", {"name": tag}))
        first: Final = _chat(mapped, key, baggage)
        assert first.status_code == 200, first.text
        eventually(lambda: _chat(mapped, key, baggage).status_code, lambda status: status != 200, seconds=70)
        mapped.provider.drain()
        blocked: Final = _chat(mapped, key, baggage, fields=fields, **headers)
        assert blocked.status_code == 422, blocked.text
        assert "budget" in blocked.text.lower(), blocked.text
        assert mapped.provider.drain() == ()
        other: Final = _chat(mapped, key, f"entrypoint=other-{entry},tenant={entry}")
        assert other.status_code == 200, other.text


def test_required_mapping_rejects_a_request_without_its_baggage_value_before_the_provider(mapped: Rig) -> None:
    with mapped.proxy.scenario() as scenario:
        key: Final = scenario.key()
        mapped.provider.drain()
        rejected: Final = _chat(mapped, key, "entrypoint=a")
        assert rejected.status_code == 400, rejected.text
        assert mapped.provider.drain() == ()
        admitted: Final = _chat(mapped, key, f"entrypoint=a,tenant={uuid.uuid4().hex}")
        assert admitted.status_code == 200, admitted.text


def test_require_budgeted_tag_rejects_an_unbudgeted_mapped_tag_until_the_tag_has_a_budget(mapped: Rig) -> None:
    value: Final = uuid.uuid4().hex
    tag: Final = mapped.budgeted_tag(value)
    baggage: Final = f"budgeted={value},tenant={value}"
    with mapped.proxy.scenario() as scenario:
        key: Final = scenario.key()
        mapped.provider.drain()
        unbudgeted: Final = tuple(_chat(mapped, key, baggage, Connection="close") for _ in range(WORKER_PROBES))
        assert [response.status_code for response in unbudgeted] == [400] * WORKER_PROBES, [
            response.text for response in unbudgeted
        ]
        assert mapped.provider.drain() == ()
        mapped.proxy.post("/tag/new", {"name": tag, "max_budget": 100})
        scenario.cleanups.callback(lambda: mapped.proxy.request("POST", "/tag/delete", {"name": tag}))
        budgeted: Final = tuple(_chat(mapped, key, baggage, Connection="close") for _ in range(WORKER_PROBES))
        assert [response.status_code for response in budgeted] == [200] * WORKER_PROBES, [
            response.text for response in budgeted
        ]


def _routing_caller_tag(rig: Rig, caller_tag: CallerTag) -> tuple[Mapping[str, JsonValue], Mapping[str, str]]:
    json_metadata: Final[Mapping[str, JsonValue]] = {"metadata": json.dumps({"tags": [rig.exact_tag]})}
    match caller_tag:
        case "none":
            return NO_FIELDS, {}
        case "header":
            return NO_FIELDS, {"x-litellm-tags": rig.exact_tag}
        case "metadata_json_with_header":
            return json_metadata, {"x-litellm-tags": f"other-{rig.marker}"}
        case "litellm_metadata":
            return {"litellm_metadata": {"tags": [rig.exact_tag]}}, {}


@pytest.mark.parametrize(
    "caller_tag",
    ["none", "header", "metadata_json_with_header", "litellm_metadata"],
    ids=["baggage_only", "caller_tag", "metadata_json_tag_with_header", "litellm_metadata_tag"],
)
def test_mapped_tags_never_select_a_tag_routed_deployment(mapped: Rig, caller_tag: CallerTag) -> None:
    tenant: Final = uuid.uuid4().hex
    expected: Final = f"default-{mapped.marker}" if caller_tag == "none" else f"tagged-{mapped.marker}"
    fields, headers = _routing_caller_tag(mapped, caller_tag)
    with mapped.proxy.scenario() as scenario:
        key: Final = scenario.key()
        responses: Final = tuple(
            _chat(
                mapped,
                key,
                f"entrypoint=a,tenant={tenant}",
                model=f"routed-{mapped.marker}",
                fields=fields,
                **headers,
            )
            for _ in range(3)
        )
        assert [response.status_code for response in responses] == [200] * 3, [r.text for r in responses]
        assert [response.headers.get("x-litellm-model-id") for response in responses] == [expected] * 3


def test_mapped_tags_never_trigger_a_tag_scoped_policy_attachment(mapped: Rig) -> None:
    entry: Final = uuid.uuid4().hex
    tag: Final = mapped.usecase_tag(entry)
    baggage: Final = f"entrypoint={entry},tenant={entry}"
    proxy: Final = mapped.proxy
    with wire_server(lambda _request: Reply(body=GUARDRAIL_VERDICT)) as guardrail, proxy.scenario() as scenario:
        key: Final = scenario.key()
        guardrail_created: Final = proxy.post(
            "/guardrails",
            {
                "guardrail": {
                    "guardrail_name": f"guardrail-{entry}",
                    "litellm_params": {
                        "guardrail": "generic_guardrail_api",
                        "mode": "pre_call",
                        "default_on": False,
                        "api_base": f"{guardrail.url}/beta/litellm_basic_guardrail_api",
                        "api_key": "synthetic-guardrail-key",
                    },
                }
            },
        )
        scenario.cleanups.callback(lambda: proxy.request("DELETE", f"/guardrails/{guardrail_created['guardrail_id']}"))
        policy: Final = proxy.post(
            "/policies", {"policy_name": f"policy-{entry}", "guardrails_add": [f"guardrail-{entry}"]}
        )
        scenario.cleanups.callback(lambda: proxy.request("DELETE", f"/policies/{policy['policy_id']}"))
        attachment: Final = proxy.post("/policies/attachments", {"policy_name": f"policy-{entry}", "tags": [tag]})
        scenario.cleanups.callback(
            lambda: proxy.request("DELETE", f"/policies/attachments/{attachment['attachment_id']}")
        )
        caller_tagged: Final = _chat(mapped, key, baggage, **{"x-litellm-tags": tag})
        assert caller_tagged.status_code == 200, caller_tagged.text
        assert caller_tagged.headers.get("x-litellm-applied-policies") == f"policy-{entry}", caller_tagged.text
        assert len(guardrail.drain()) == 1
        baggage_only: Final = _chat(mapped, key, baggage)
        assert baggage_only.status_code == 200, baggage_only.text
        assert baggage_only.headers.get("x-litellm-applied-policies") is None, baggage_only.text
        assert guardrail.drain() == ()


def _handshake_status(url: str, headers: Mapping[str, str]) -> int:
    try:
        with connect(url, additional_headers=headers, open_timeout=30):
            return 101
    except InvalidStatus as refusal:
        return refusal.response.status_code


def test_required_mapping_refuses_websocket_inference_before_the_upstream(mapped: Rig) -> None:
    proxy_url: Final = str(mapped.proxy.client.base_url).rstrip("/")
    upstream_api_key: Final = f"Bearer realtime-{mapped.marker}"
    with (
        mapped.proxy.scenario() as scenario,
        httpx.Client(base_url=mapped.proxy.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        key: Final = scenario.key()
        upstream.get("/__observations").raise_for_status()
        status: Final = _handshake_status(
            f"ws{proxy_url.removeprefix('http')}/v1/realtime?model=realtime-{mapped.marker}",
            {"Authorization": f"Bearer {key}", "baggage": f"entrypoint=a,tenant={uuid.uuid4().hex}"},
        )
        assert status == 403, status
        observed: Final = list_value(JSON_OBJECT.validate_json(upstream.get("/__observations").content)["requests"])
        upgrades: Final = tuple(
            request
            for request in (object_value(item) for item in observed)
            if request["method"] == "WEBSOCKET" and request["authorization"] == upstream_api_key
        )
        assert upgrades == (), upgrades


@pytest.mark.parametrize(
    ("route", "body"),
    [
        ("/utils/token_counter", {"messages": [{"role": "user", "content": "count me"}]}),
        ("/v1/messages/count_tokens", {"messages": [{"role": "user", "content": "count me"}]}),
    ],
    ids=["token_counter", "messages_count_tokens"],
)
def test_required_mapping_skips_unbilled_routes(mapped: Rig, route: str, body: dict[str, JsonValue]) -> None:
    with mapped.proxy.scenario() as scenario:
        key: Final = scenario.key()
        response: Final = mapped.proxy.request("POST", route, {"model": mapped.chat_model, **body}, key=key)
        assert response.status_code == 200, response.text


def test_custom_passthrough_spend_row_carries_the_mapped_tags(mapped: Rig) -> None:
    tenant: Final = uuid.uuid4().hex
    path: Final = f"/baggage-passthrough-{tenant}"
    with mapped.proxy.scenario() as scenario:
        created: Final = mapped.proxy.post(
            "/config/pass_through_endpoint", {"path": path, "target": f"{mapped.provider.url}/echo", "auth": True}
        )
        endpoint_id: Final = str(object_value(list_value(created["endpoints"])[0])["id"])
        scenario.cleanups.callback(
            lambda: mapped.proxy.request("DELETE", "/config/pass_through_endpoint", params={"endpoint_id": endpoint_id})
        )
        key: Final = scenario.key(allowed_passthrough_routes=[path])
        response: Final = mapped.proxy.request(
            "POST",
            path,
            {"model": mapped.chat_model, "messages": [{"role": "user", "content": tenant}]},
            key=key,
            headers={"baggage": f"entrypoint=a,tenant={tenant}"},
        )
        assert response.status_code == 200, response.text
        assert {mapped.exact_tag, mapped.tenant_tag(tenant)} <= _spend_tags(mapped.proxy, key)


def _rejected_client_tag(client_tag: RejectedClientTag, tag: str) -> tuple[Mapping[str, JsonValue], Mapping[str, str]]:
    match client_tag:
        case "none":
            return NO_FIELDS, {}
        case "header":
            return NO_FIELDS, {"x-litellm-tags": tag}
        case "body":
            return {"metadata": {"tags": [tag]}}, {}
        case "metadata_json":
            return {"metadata": json.dumps({"tags": [tag]})}, {}
        case "litellm_metadata":
            return {"litellm_metadata": {"tags": [tag]}}, {}


@pytest.mark.parametrize(
    ("client_tag", "baggage_maps", "plain_statuses"),
    [
        ("none", True, (200,)),
        ("header", True, (400,)),
        ("body", True, (400,)),
        ("metadata_json", True, (200, 400)),
        ("metadata_json", False, (200, 400)),
        ("litellm_metadata", True, (200,)),
    ],
    ids=[
        "baggage_only",
        "header_tag",
        "body_tag",
        "metadata_json_tag",
        "metadata_json_tag_unmapped_baggage",
        "litellm_metadata_tag",
    ],
)
def test_reject_clientside_metadata_tags_with_baggage(
    rejecting: Rig, client_tag: RejectedClientTag, baggage_maps: bool, plain_statuses: tuple[int, ...]
) -> None:
    tenant: Final = uuid.uuid4().hex
    fields, headers = _rejected_client_tag(client_tag, f"caller-{tenant}")
    baggage: Final = f"entrypoint=a,tenant={tenant}" if baggage_maps else f"region={tenant}"
    mapped_tags: Final = frozenset({rejecting.exact_tag, rejecting.tenant_tag(tenant)} if baggage_maps else ())
    with rejecting.proxy.scenario() as scenario:
        plain_key: Final = scenario.key()
        key: Final = scenario.key()
        plain: Final = _chat(rejecting, plain_key, None, fields=fields, **headers)
        response: Final = _chat(rejecting, key, baggage, fields=fields, **headers)
        assert plain.status_code in plain_statuses, plain.text
        assert response.status_code == plain.status_code, (plain.text, response.text)
        if response.status_code != 200:
            return
        assert _spend_tags(rejecting.proxy, key) == _spend_tags(rejecting.proxy, plain_key) | mapped_tags


@pytest.mark.parametrize(
    ("baggage_keys", "maps_team"),
    [(("team+id",), True), (("team%2Bid",), True), (("team+id", "team%2Bid"), False)],
    ids=["plus", "percent_encoded", "both_spellings"],
)
def test_plus_sign_rule_key_matches_either_wire_spelling_but_not_both(
    optional: Rig, baggage_keys: tuple[str, ...], maps_team: bool
) -> None:
    value: Final = uuid.uuid4().hex
    baggage: Final = ",".join(("entrypoint=a", *(f"{baggage_key}={value}" for baggage_key in baggage_keys)))
    with optional.proxy.scenario() as scenario:
        key: Final = scenario.key()
        response: Final = _chat(optional, key, baggage)
        assert response.status_code == 200, response.text
        team_tags: Final = {optional.team_tag(value)} if maps_team else set()
        assert _spend_tags(optional.proxy, key) == {optional.exact_tag} | team_tags


@pytest.mark.parametrize("deployment_tagged", [True, False], ids=["tagged_deployment", "untagged_deployment"])
def test_serving_deployment_tag_selects_its_guardrail_mode_when_baggage_maps_the_same_tag(
    optional: Rig, deployment_tagged: bool
) -> None:
    model: Final = f"guarded-{optional.marker}" if deployment_tagged else optional.chat_model
    with optional.proxy.scenario() as scenario:
        key: Final = scenario.key()
        optional.provider.drain()
        response: Final = _chat(optional, key, "guard=on", model=model)
        assert response.status_code == 200, response.text
        guardrail_inputs: Final = _guardrail_inputs(optional)
        assert guardrail_inputs == (("response",) if deployment_tagged else ()), guardrail_inputs


def test_baggage_mapping_a_deployment_tag_keeps_the_team_tag_guardrail_mode_first(optional: Rig) -> None:
    model: Final = f"guarded-pre-{optional.marker}"
    with optional.proxy.scenario() as scenario:
        team: Final = scenario.team(metadata={"tags": [optional.policy_tag]})
        key: Final = scenario.key(team_id=team)
        optional.provider.drain()
        plain: Final = _chat(optional, key, None, model=model)
        plain_inputs: Final = _guardrail_inputs(optional)
        response: Final = _chat(optional, key, "guard=pre", model=model)
        assert (plain.status_code, response.status_code) == (200, 200), (plain.text, response.text)
        assert (plain_inputs, _guardrail_inputs(optional)) == (("response",), ("response",))


@pytest.mark.parametrize("metadata", ["run-123", {"tags": "run-123"}], ids=["non_object", "string_tags"])
@pytest.mark.parametrize(
    ("header_tag", "baggage"),
    [(True, "region=sg"), (True, "entrypoint=a"), (False, "region=sg"), (False, "entrypoint=a")],
    ids=["header_tag_unmapped_baggage", "header_tag_mapped_baggage", "unmapped_baggage", "mapped_baggage"],
)
def test_malformed_metadata_keeps_its_status_under_baggage(
    optional: Rig, header_tag: bool, baggage: str, metadata: JsonValue
) -> None:
    headers: Final = {"x-litellm-tags": f"caller-{uuid.uuid4().hex}"} if header_tag else {}
    fields: Final[Mapping[str, JsonValue]] = {"metadata": metadata}
    with optional.proxy.scenario() as scenario:
        key: Final = scenario.key()
        optional.provider.drain()
        plain: Final = _chat(optional, key, None, fields=fields, **headers)
        plain_calls: Final = len(optional.provider.drain())
        response: Final = _chat(optional, key, baggage, fields=fields, **headers)
        assert plain.status_code < 500, plain.text
        assert plain_calls == (1 if plain.status_code == 200 else 0)
        assert (response.status_code, len(optional.provider.drain())) == (plain.status_code, plain_calls), (
            plain.text,
            response.text,
        )


@pytest.mark.parametrize("outage_flag", [False, True], ids=["outage_flag_off", "outage_flag_on"])
def test_budgeted_mapping_on_a_failed_tag_read(gateway: Gateway, tmp_path: Path, outage_flag: bool) -> None:
    marker: Final = uuid.uuid4().hex[:12]
    value: Final = uuid.uuid4().hex
    tag: Final = f"budgeted-{marker}:{value}"
    with ExitStack() as stack:
        database_url: Final = stack.enter_context(scratch_database())
        relay, relayed_url = stack.enter_context(dropped_connection_relay(database_url, tag.encode()))
        provider: Final = stack.enter_context(wire_server(_provider))
        config: Final = _write_config(
            tmp_path,
            [_deployment(f"chat-{marker}", provider)],
            {
                "request_tag_mappings": [
                    {"key": "budgeted", "tag_prefix": f"budgeted-{marker}:", "require_budgeted_tag": True}
                ],
                "allow_requests_on_db_unavailable": outage_flag,
            },
            {},
        )
        proxy: Final = stack.enter_context(
            owned_proxy(
                gateway,
                tmp_path,
                {"DATABASE_URL": plaintext_database_url(relayed_url), "PRISMA_HEALTH_WATCHDOG_ENABLED": "false"},
                config=config,
                remove_environment=("DATABASE_URL_READ_REPLICA",),
            )
        )
        scenario: Final = stack.enter_context(proxy.scenario())
        key: Final = scenario.key(max_budget=50)
        proxy.post("/tag/new", {"name": tag, "max_budget": 100})
        rig: Final = Rig(proxy, provider, provider, marker)
        warmed: Final = _chat(rig, key, None)
        assert warmed.status_code == 200, warmed.text
        provider.drain()
        relay.arm()
        response: Final = _chat(rig, key, f"budgeted={value}")
        relay.disarm()
        assert relay.dropped.is_set(), response.text
        if not outage_flag:
            assert response.status_code == 503, response.text
            assert provider.drain() == ()
            return
        assert response.status_code == 200, response.text
        assert response.headers.get("x-litellm-key-max-budget") == "50.0", dict(response.headers)
        assert len(provider.drain()) == 1
