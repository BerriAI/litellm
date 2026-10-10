from __future__ import annotations

import uuid
from itertools import count
from typing import Final

import pytest
from integration._support.client import JSON_OBJECT, Gateway, eventually, object_value, string_value
from integration._support.responses_vendor import same_response
from integration._support.wire import Reply, Request, wire_server
from integration.providers._coralbricks import (
    ANSWER,
    API_KEY,
    COMPLETION_TOKENS,
    MODEL,
    MODELS,
    NO_CACHE,
    PROMPT_TOKENS,
    PROVIDER,
    USER_MESSAGES,
    anthropic_client,
    assert_billed,
    assert_failed,
    async_anthropic_client,
    async_openai_client,
    closed_port,
    deployment,
    expected_spend,
    failing_provider,
    group_rows,
    is_readiness_probe,
    not_yet_routable,
    number,
    only_call,
    openai_client,
    provider,
    provider_calls,
    rates_of,
    raw_deployment,
    ready,
    spend_row,
)
from pydantic import JsonValue

DEPLOYMENT_RETRIES: Final = 2


@pytest.mark.parametrize("model", MODELS)
def test_openai_sdk_chat_reaches_coralbricks_and_bills_the_cost_map_row(gateway: Gateway, model: str) -> None:
    identity: Final = f"req_{uuid.uuid4().hex}"
    with wire_server(provider(identity, model)) as wire, gateway.scenario() as scenario:
        alias: Final = deployment(scenario, wire, model)
        response: Final = openai_client(gateway).chat.completions.create(
            model=alias,
            messages=[{"role": "user", "content": "Say hello to the reef"}],
            extra_body={"cache": NO_CACHE},
        )
        assert response.id == identity
        assert response.choices[0].message.content == ANSWER
        sent: Final = only_call(wire, "/v1/chat/completions", model)
        assert sent["messages"] == USER_MESSAGES, sent
        assert_billed(spend_row(identity), alias, "acompletion", model)


async def test_async_openai_sdk_chat_reaches_coralbricks(gateway: Gateway) -> None:
    identity: Final = f"req_{uuid.uuid4().hex}"
    with wire_server(provider(identity)) as wire, gateway.scenario() as scenario:
        alias: Final = deployment(scenario, wire)
        response: Final = await async_openai_client(gateway).chat.completions.create(
            model=alias,
            messages=[{"role": "user", "content": "Say hello to the reef"}],
            extra_body={"cache": NO_CACHE},
        )
        assert response.id == identity
        assert response.choices[0].message.content == ANSWER
        assert only_call(wire, "/v1/chat/completions")["messages"] == USER_MESSAGES
        assert_billed(spend_row(identity), alias, "acompletion")


def test_openai_sdk_chat_stream_is_forwarded_and_billed_from_the_usage_frame(gateway: Gateway) -> None:
    identity: Final = f"req_{uuid.uuid4().hex}"
    with wire_server(provider(identity)) as wire, gateway.scenario() as scenario:
        alias: Final = deployment(scenario, wire)
        chunks: Final = list(
            openai_client(gateway).chat.completions.create(
                model=alias,
                messages=[{"role": "user", "content": "Say hello to the reef"}],
                stream=True,
                stream_options={"include_usage": True},
                extra_body={"cache": NO_CACHE},
            )
        )
        assert {chunk.id for chunk in chunks} == {identity}
        assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices) == ANSWER
        usage: Final = [chunk.usage for chunk in chunks if chunk.usage is not None]
        assert len(usage) == 1 and usage[0].prompt_tokens == PROMPT_TOKENS, usage
        sent: Final = only_call(wire, "/v1/chat/completions")
        assert sent["stream"] is True, sent
        assert_billed(spend_row(identity), alias, "acompletion")


async def test_async_openai_sdk_chat_stream_is_forwarded_and_billed(gateway: Gateway) -> None:
    identity: Final = f"req_{uuid.uuid4().hex}"
    with wire_server(provider(identity)) as wire, gateway.scenario() as scenario:
        alias: Final = deployment(scenario, wire)
        stream: Final = await async_openai_client(gateway).chat.completions.create(
            model=alias,
            messages=[{"role": "user", "content": "Say hello to the reef"}],
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"cache": NO_CACHE},
        )
        chunks: Final = [chunk async for chunk in stream]
        assert {chunk.id for chunk in chunks} == {identity}
        assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices) == ANSWER
        assert only_call(wire, "/v1/chat/completions")["stream"] is True
        assert_billed(spend_row(identity), alias, "acompletion")


def test_raw_chat_request_carries_the_call_id_and_lands_once(gateway: Gateway) -> None:
    identity: Final = f"req_{uuid.uuid4().hex}"
    with wire_server(provider(identity)) as wire, gateway.scenario() as scenario:
        alias: Final = deployment(scenario, wire)
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", {"model": alias, "messages": USER_MESSAGES, "cache": NO_CACHE}
        )
        assert response.status_code == 200, response.text
        assert "x-litellm-call-id" in response.headers, response.headers
        body: Final = JSON_OBJECT.validate_json(response.content)
        assert body["id"] == identity, response.text
        assert only_call(wire, "/v1/chat/completions")["messages"] == USER_MESSAGES
        assert_billed(spend_row(identity), alias, "acompletion")


def test_openai_sdk_responses_go_natively_to_coralbricks_responses(gateway: Gateway) -> None:
    identity: Final = f"resp_{uuid.uuid4().hex}"
    with wire_server(provider(identity)) as wire, gateway.scenario() as scenario:
        alias: Final = deployment(scenario, wire)
        response: Final = openai_client(gateway).responses.create(
            model=alias, input="Say hello to the reef", extra_body={"cache": NO_CACHE}
        )
        assert response.output_text == ANSWER
        assert same_response(response.id, identity), response.id
        sent: Final = only_call(wire, "/v1/responses")
        assert sent["input"] == "Say hello to the reef", sent
        assert_billed(spend_row(response.id), alias, "aresponses")


async def test_async_openai_sdk_responses_go_natively_to_coralbricks_responses(gateway: Gateway) -> None:
    identity: Final = f"resp_{uuid.uuid4().hex}"
    with wire_server(provider(identity)) as wire, gateway.scenario() as scenario:
        alias: Final = deployment(scenario, wire)
        response: Final = await async_openai_client(gateway).responses.create(
            model=alias, input="Say hello to the reef", extra_body={"cache": NO_CACHE}
        )
        assert response.output_text == ANSWER
        assert same_response(response.id, identity), response.id
        assert only_call(wire, "/v1/responses")["input"] == "Say hello to the reef"
        assert_billed(spend_row(response.id), alias, "aresponses")


def test_openai_sdk_responses_stream_is_forwarded_natively_and_billed_once(gateway: Gateway) -> None:
    identity: Final = f"resp_{uuid.uuid4().hex}"
    with wire_server(provider(identity)) as wire, gateway.scenario() as scenario:
        alias: Final = deployment(scenario, wire)
        events: Final = list(
            openai_client(gateway).responses.create(
                model=alias, input="Say hello to the reef", stream=True, extra_body={"cache": NO_CACHE}
            )
        )
        assert [event.type for event in events] == [
            "response.created",
            "response.output_text.delta",
            "response.completed",
        ], [event.type for event in events]
        completed: Final = events[-1]
        assert completed.type == "response.completed"
        assert completed.response.output_text == ANSWER
        sent: Final = only_call(wire, "/v1/responses")
        assert sent["stream"] is True and sent["input"] == "Say hello to the reef", sent
        rows: Final = group_rows(alias, 1)
        assert same_response(completed.response.id, string_value(rows[0]["request_id"])), rows
        assert_billed(rows[0], alias, "aresponses")


async def test_async_openai_sdk_responses_stream_is_forwarded_natively_and_billed_once(gateway: Gateway) -> None:
    identity: Final = f"resp_{uuid.uuid4().hex}"
    with wire_server(provider(identity)) as wire, gateway.scenario() as scenario:
        alias: Final = deployment(scenario, wire)
        stream: Final = await async_openai_client(gateway).responses.create(
            model=alias, input="Say hello to the reef", stream=True, extra_body={"cache": NO_CACHE}
        )
        events: Final = [event async for event in stream]
        assert [event.type for event in events] == [
            "response.created",
            "response.output_text.delta",
            "response.completed",
        ], [event.type for event in events]
        completed: Final = events[-1]
        assert completed.type == "response.completed"
        assert completed.response.output_text == ANSWER
        assert only_call(wire, "/v1/responses")["stream"] is True
        rows: Final = group_rows(alias, 1)
        assert same_response(completed.response.id, string_value(rows[0]["request_id"])), rows
        assert_billed(rows[0], alias, "aresponses")


def test_raw_responses_request_lands_once(gateway: Gateway) -> None:
    identity: Final = f"resp_{uuid.uuid4().hex}"
    with wire_server(provider(identity)) as wire, gateway.scenario() as scenario:
        alias: Final = deployment(scenario, wire)
        response: Final = gateway.request(
            "POST", "/v1/responses", {"model": alias, "input": "Say hello to the reef", "cache": NO_CACHE}
        )
        assert response.status_code == 200, response.text
        body: Final = JSON_OBJECT.validate_json(response.content)
        assert same_response(string_value(body["id"]), identity), response.text
        assert only_call(wire, "/v1/responses")["input"] == "Say hello to the reef"
        assert_billed(spend_row(string_value(body["id"])), alias, "aresponses")


def test_anthropic_sdk_messages_go_natively_to_coralbricks_messages(gateway: Gateway) -> None:
    identity: Final = f"msg_{uuid.uuid4().hex}"
    with wire_server(provider(identity)) as wire, gateway.scenario() as scenario:
        alias: Final = deployment(scenario, wire)
        response: Final = anthropic_client(gateway).messages.create(
            model=alias,
            max_tokens=64,
            messages=[{"role": "user", "content": "Say hello to the reef"}],
            extra_body={"cache": NO_CACHE},
        )
        assert response.id == identity
        assert response.content[0].type == "text" and response.content[0].text == ANSWER
        assert only_call(wire, "/v1/messages")["messages"] == USER_MESSAGES
        assert_billed(spend_row(identity), alias, "anthropic_messages")


async def test_async_anthropic_sdk_messages_go_natively_to_coralbricks_messages(gateway: Gateway) -> None:
    identity: Final = f"msg_{uuid.uuid4().hex}"
    with wire_server(provider(identity)) as wire, gateway.scenario() as scenario:
        alias: Final = deployment(scenario, wire)
        response: Final = await async_anthropic_client(gateway).messages.create(
            model=alias,
            max_tokens=64,
            messages=[{"role": "user", "content": "Say hello to the reef"}],
            extra_body={"cache": NO_CACHE},
        )
        assert response.id == identity
        assert response.content[0].type == "text" and response.content[0].text == ANSWER
        assert only_call(wire, "/v1/messages")["messages"] == USER_MESSAGES
        assert_billed(spend_row(identity), alias, "anthropic_messages")


def test_anthropic_sdk_messages_stream_is_forwarded_natively_and_billed_once(gateway: Gateway) -> None:
    identity: Final = f"msg_{uuid.uuid4().hex}"
    with wire_server(provider(identity)) as wire, gateway.scenario() as scenario:
        alias: Final = deployment(scenario, wire)
        with anthropic_client(gateway).messages.stream(
            model=alias,
            max_tokens=64,
            messages=[{"role": "user", "content": "Say hello to the reef"}],
            extra_body={"cache": NO_CACHE},
        ) as stream:
            text: Final = "".join(stream.text_stream)
            final: Final = stream.get_final_message()
        assert text == ANSWER
        assert final.id == identity and final.usage.output_tokens == COMPLETION_TOKENS, final
        assert only_call(wire, "/v1/messages")["stream"] is True
        assert_billed(spend_row(identity), alias, "anthropic_messages")


async def test_async_anthropic_sdk_messages_stream_is_forwarded_natively_and_billed_once(gateway: Gateway) -> None:
    identity: Final = f"msg_{uuid.uuid4().hex}"
    with wire_server(provider(identity)) as wire, gateway.scenario() as scenario:
        alias: Final = deployment(scenario, wire)
        async with async_anthropic_client(gateway).messages.stream(
            model=alias,
            max_tokens=64,
            messages=[{"role": "user", "content": "Say hello to the reef"}],
            extra_body={"cache": NO_CACHE},
        ) as stream:
            text: Final = "".join([piece async for piece in stream.text_stream])
            final: Final = await stream.get_final_message()
        assert text == ANSWER
        assert final.id == identity, final
        assert only_call(wire, "/v1/messages")["stream"] is True
        assert_billed(spend_row(identity), alias, "anthropic_messages")


def test_raw_messages_request_lands_once(gateway: Gateway) -> None:
    identity: Final = f"msg_{uuid.uuid4().hex}"
    with wire_server(provider(identity)) as wire, gateway.scenario() as scenario:
        alias: Final = deployment(scenario, wire)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {"model": alias, "max_tokens": 64, "messages": USER_MESSAGES, "cache": NO_CACHE},
        )
        assert response.status_code == 200, response.text
        assert JSON_OBJECT.validate_json(response.content)["id"] == identity, response.text
        assert only_call(wire, "/v1/messages")["messages"] == USER_MESSAGES
        assert_billed(spend_row(identity), alias, "anthropic_messages")


def request_body(target: str, alias: str) -> dict[str, JsonValue]:
    match target:
        case "/v1/responses":
            return {"model": alias, "input": "Say hello to the reef", "cache": NO_CACHE}
        case "/v1/messages":
            return {"model": alias, "max_tokens": 64, "messages": USER_MESSAGES, "cache": NO_CACHE}
        case _:
            return {"model": alias, "messages": USER_MESSAGES, "cache": NO_CACHE}


@pytest.mark.parametrize(
    ("target", "status"),
    (("/v1/chat/completions", 401), ("/v1/chat/completions", 404), ("/v1/responses", 401), ("/v1/messages", 401)),
)
def test_provider_client_error_reaches_the_caller_after_one_attempt(gateway: Gateway, target: str, status: int) -> None:
    message: Final = f"scripted coralbricks {status} {uuid.uuid4().hex}"
    with wire_server(failing_provider(status, message)) as wire, gateway.scenario() as scenario:
        alias: Final = scenario.model(model=f"{PROVIDER}/{MODEL}", api_base=f"{wire.url}/v1", api_key=API_KEY)
        probe: Final = eventually(
            lambda: gateway.request("POST", "/v1/chat/completions", {"model": alias, "messages": USER_MESSAGES}),
            lambda response: not not_yet_routable(response),
            seconds=60,
        )
        assert probe.status_code == status, probe.text
        wire.drain()
        response: Final = gateway.request("POST", target, request_body(target, alias))
        assert response.status_code == status, response.text
        assert message in response.text, response.text
        attempts: Final = [(request.method, request.target) for request in wire.drain()]
        assert attempts == [("POST", target)], attempts
        assert_failed(spend_row(response.headers["x-litellm-call-id"]), alias)
        assert gateway.request("GET", "/health/liveliness").status_code == 200


@pytest.mark.parametrize("status", (429, 500))
def test_provider_server_error_is_retried_then_reaches_the_caller(gateway: Gateway, status: int) -> None:
    message: Final = f"scripted coralbricks {status} {uuid.uuid4().hex}"
    with wire_server(failing_provider(status, message)) as wire, gateway.scenario() as scenario:
        alias: Final = scenario.model(
            model=f"{PROVIDER}/{MODEL}", api_base=f"{wire.url}/v1", api_key=API_KEY, num_retries=DEPLOYMENT_RETRIES
        )
        eventually(
            lambda: gateway.request("POST", "/v1/chat/completions", {"model": alias, "messages": USER_MESSAGES}),
            lambda response: not not_yet_routable(response),
            seconds=60,
        )
        wire.drain()
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", {"model": alias, "messages": USER_MESSAGES, "cache": NO_CACHE}
        )
        assert response.status_code == status, response.text
        assert message in response.text, response.text
        attempts: Final = [(request.method, request.target) for request in wire.drain()]
        assert attempts == [("POST", "/v1/chat/completions")] * (1 + DEPLOYMENT_RETRIES), attempts
        assert_failed(spend_row(response.headers["x-litellm-call-id"]), alias)


def test_unreachable_api_base_reports_the_connection_error_and_leaves_the_proxy_serving(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        control: Final = scenario.model()
        alias: Final = scenario.model(
            model=f"{PROVIDER}/{MODEL}", api_base=f"http://127.0.0.1:{closed_port()}/v1", api_key=API_KEY
        )
        response: Final = eventually(
            lambda: gateway.request(
                "POST", "/v1/chat/completions", {"model": alias, "messages": USER_MESSAGES, "cache": NO_CACHE}
            ),
            lambda candidate: not not_yet_routable(candidate),
            seconds=60,
        )
        assert response.status_code == 500, response.text
        assert "Connection error" in response.text, response.text
        assert_failed(spend_row(response.headers["x-litellm-call-id"]), alias)
        assert gateway.request("GET", "/health/liveliness").status_code == 200
        ready(gateway, control)
        assert gateway.chat(control, text="still serving")["object"] == "chat.completion"


@pytest.mark.parametrize("api_key", ("missing", None, ""), ids=("missing", "null", "empty"))
def test_deployment_without_a_key_and_no_env_key_fails_before_the_provider(
    gateway: Gateway, api_key: str | None
) -> None:
    with wire_server(provider("unused")) as wire, gateway.scenario() as scenario:
        litellm_params: Final[dict[str, JsonValue]] = {
            "model": f"{PROVIDER}/{MODEL}",
            "api_base": f"{wire.url}/v1",
            **({} if api_key == "missing" else {"api_key": api_key}),
        }
        alias: Final = raw_deployment(scenario, f"integration-{uuid.uuid4().hex}", litellm_params)
        response: Final = eventually(
            lambda: gateway.request(
                "POST", "/v1/chat/completions", {"model": alias, "messages": USER_MESSAGES, "cache": NO_CACHE}
            ),
            lambda candidate: not not_yet_routable(candidate),
            seconds=60,
        )
        assert response.status_code == 500, response.text
        assert "AuthenticationError" in response.text and "CORALBRICKS_API_KEY" in response.text, response.text
        assert_failed(spend_row(response.headers["x-litellm-call-id"]), alias)
        assert wire.drain() == (), "the provider was called without a key"
        assert gateway.request("GET", "/health/liveliness").status_code == 200


@pytest.mark.parametrize(
    "model",
    (7, ["coralbricks/deepseek-v4.1-flash-fast"], f"{PROVIDER}/" + "g" * 5120),
    ids=("int", "list", "5kb"),
)
def test_hostile_model_field_is_rejected_without_crashing(gateway: Gateway, model: JsonValue) -> None:
    for _ in range(2):
        response: Final = gateway.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": USER_MESSAGES, "cache": NO_CACHE}
        )
        assert response.status_code in (400, 422), response.text
    assert gateway.request("GET", "/health/liveliness").status_code == 200


def test_identical_uncached_chats_are_each_forwarded_and_billed(gateway: Gateway) -> None:
    prefix: Final = f"req_{uuid.uuid4().hex}"
    calls: Final = count(1)

    def respond(request: Request) -> Reply:
        if is_readiness_probe(request):
            return provider(f"{prefix}-probe")(request)
        return provider(f"{prefix}-{next(calls)}")(request)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        alias: Final = deployment(scenario, wire)
        wire.drain()
        first: Final = gateway.post(
            "/v1/chat/completions", {"model": alias, "messages": USER_MESSAGES, "cache": NO_CACHE}
        )
        second: Final = gateway.post(
            "/v1/chat/completions", {"model": alias, "messages": USER_MESSAGES, "cache": NO_CACHE}
        )
        assert first["id"] != second["id"], (first["id"], second["id"])
        assert [(request.method, request.target) for request in provider_calls(wire)] == [
            ("POST", "/v1/chat/completions")
        ] * 2
        rows: Final = group_rows(alias, 2)
        assert {string_value(row["request_id"]) for row in rows} == {string_value(first["id"]), string_value(second["id"])}
        for row in rows:
            assert_billed(row, alias, "acompletion")


def test_model_info_reports_the_coralbricks_cost_map_row(gateway: Gateway) -> None:
    rates: Final = rates_of(MODEL)
    with wire_server(provider("unused")) as wire, gateway.scenario() as scenario:
        alias: Final = scenario.model(model=f"{PROVIDER}/{MODEL}", api_base=f"{wire.url}/v1", api_key=API_KEY)
        entries: Final = gateway.get("/model/info")["data"]
        assert isinstance(entries, list)
        mine: Final = [object_value(entry) for entry in entries if object_value(entry)["model_name"] == alias]
        assert len(mine) == 1, [object_value(entry)["model_name"] for entry in entries]
        litellm_params: Final = object_value(mine[0]["litellm_params"])
        info: Final = object_value(mine[0]["model_info"])
        assert litellm_params["model"] == f"{PROVIDER}/{MODEL}", litellm_params
        assert info["litellm_provider"] == PROVIDER, info
        assert number(info["input_cost_per_token"]) == pytest.approx(rates.input), info
        assert number(info["output_cost_per_token"]) == pytest.approx(rates.output), info
        assert number(info["cache_creation_input_token_cost"]) == pytest.approx(rates.cache_write), info
        assert info["max_input_tokens"] == 1048576 and info["mode"] == "chat", info
        assert info["supports_function_calling"] is True and info["supports_prompt_caching"] is True, info


def test_models_listing_and_model_group_info_carry_the_coralbricks_deployment(gateway: Gateway) -> None:
    rates: Final = rates_of(MODEL)
    with wire_server(provider("unused")) as wire, gateway.scenario() as scenario:
        alias: Final = scenario.model(model=f"{PROVIDER}/{MODEL}", api_base=f"{wire.url}/v1", api_key=API_KEY)
        listed: Final = gateway.get("/v1/models")["data"]
        assert isinstance(listed, list)
        assert alias in {object_value(entry)["id"] for entry in listed}
        groups: Final = gateway.get("/model_group/info")["data"]
        assert isinstance(groups, list)
        mine: Final = [object_value(group) for group in groups if object_value(group)["model_group"] == alias]
        assert len(mine) == 1, [object_value(group)["model_group"] for group in groups]
        assert mine[0]["providers"] == [PROVIDER], mine[0]
        assert number(mine[0]["input_cost_per_token"]) == pytest.approx(rates.input), mine[0]
        assert number(mine[0]["output_cost_per_token"]) == pytest.approx(rates.output), mine[0]


def test_health_probe_reaches_coralbricks_with_the_deployment_key(gateway: Gateway) -> None:
    identity: Final = f"req_{uuid.uuid4().hex}"
    with wire_server(provider(identity)) as wire, gateway.scenario() as scenario:
        alias: Final = deployment(scenario, wire)
        wire.drain()
        health: Final = gateway.get("/health", {"model": alias})
        assert health["healthy_count"] == 1 and health["unhealthy_count"] == 0, health
        healthy: Final = health["healthy_endpoints"]
        assert isinstance(healthy, list) and object_value(healthy[0])["model"] == f"{PROVIDER}/{MODEL}", health
        probes: Final = provider_calls(wire)
        assert [(request.method, request.target) for request in probes] == [("POST", "/v1/chat/completions")], probes
        assert probes[0].headers.get("authorization") == f"Bearer {API_KEY}", probes[0].headers
        assert JSON_OBJECT.validate_json(probes[0].body)["model"] == MODEL


def test_add_model_form_fields_name_the_coralbricks_credentials(gateway: Gateway) -> None:
    response: Final = gateway.request("GET", "/public/providers/fields")
    assert response.status_code == 200, response.text
    providers: Final = response.json()
    assert isinstance(providers, list), response.text
    mine: Final = [object_value(entry) for entry in providers if object_value(entry)["litellm_provider"] == PROVIDER]
    assert len(mine) == 1, [object_value(entry)["litellm_provider"] for entry in providers]
    assert mine[0]["provider_display_name"] == "CoralBricks", mine[0]
    fields: Final = mine[0]["credential_fields"]
    assert isinstance(fields, list), mine[0]
    by_key: Final = {string_value(object_value(field)["key"]): object_value(field) for field in fields}
    assert set(by_key) == {"api_key", "api_base"}, sorted(by_key)
    assert by_key["api_key"]["required"] is True and by_key["api_base"]["required"] is False, by_key
    assert string_value(by_key["api_base"]["placeholder"]).endswith("/v1"), by_key["api_base"]


def test_key_scoped_to_the_coralbricks_deployment_is_charged_its_spend(gateway: Gateway) -> None:
    identity: Final = f"req_{uuid.uuid4().hex}"
    with wire_server(provider(identity)) as wire, gateway.scenario() as scenario:
        alias: Final = deployment(scenario, wire)
        token: Final = scenario.key(models=[alias], max_budget=1.0)
        wire.drain()
        response: Final = gateway.post(
            "/v1/chat/completions", {"model": alias, "messages": USER_MESSAGES, "cache": NO_CACHE}, key=token
        )
        assert response["id"] == identity, response
        only_call(wire, "/v1/chat/completions")
        assert_billed(spend_row(identity), alias, "acompletion")
        info: Final = eventually(
            lambda: object_value(gateway.get("/key/info", {"key": token})["info"]),
            lambda found: number(found["spend"]) > 0,
            seconds=70,
        )
        assert number(info["spend"]) == pytest.approx(expected_spend(MODEL)), info


def test_wildcard_coralbricks_deployment_routes_the_requested_model(gateway: Gateway) -> None:
    identity: Final = f"req_{uuid.uuid4().hex}"
    with wire_server(provider(identity)) as wire, gateway.scenario() as scenario:
        prefix: Final = f"integration-{uuid.uuid4().hex}"
        raw_deployment(
            scenario,
            f"{prefix}/*",
            {"model": f"{PROVIDER}/*", "api_base": f"{wire.url}/v1", "api_key": API_KEY},
        )
        requested: Final = f"{prefix}/{MODEL}"
        response: Final = eventually(
            lambda: gateway.request(
                "POST", "/v1/chat/completions", {"model": requested, "messages": USER_MESSAGES, "cache": NO_CACHE}
            ),
            lambda candidate: not not_yet_routable(candidate),
            seconds=60,
        )
        assert response.status_code == 200, response.text
        assert JSON_OBJECT.validate_json(response.content)["id"] == identity, response.text
        only_call(wire, "/v1/chat/completions")
        row: Final = spend_row(identity)
        assert row["model"] == f"{PROVIDER}/{MODEL}" and row["custom_llm_provider"] == PROVIDER, row
        assert number(row["spend"]) == pytest.approx(expected_spend(MODEL)), row


def test_fallback_from_a_refused_primary_lands_on_coralbricks(gateway: Gateway) -> None:
    identity: Final = f"req_{uuid.uuid4().hex}"
    with wire_server(provider(identity)) as wire, gateway.scenario() as scenario:
        alias: Final = deployment(scenario, wire)
        primary: Final = scenario.model(
            model=f"{PROVIDER}/{MODEL}", api_base=f"http://127.0.0.1:{closed_port()}/v1", api_key=API_KEY
        )
        wire.drain()
        response: Final = eventually(
            lambda: gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": primary, "messages": USER_MESSAGES, "cache": NO_CACHE, "fallbacks": [alias]},
            ),
            lambda candidate: not not_yet_routable(candidate),
            seconds=60,
        )
        assert response.status_code == 200, response.text
        assert response.headers.get("x-litellm-attempted-fallbacks") == "1", response.headers
        assert JSON_OBJECT.validate_json(response.content)["id"] == identity, response.text
        only_call(wire, "/v1/chat/completions")
        assert_billed(spend_row(identity), alias, "acompletion")
