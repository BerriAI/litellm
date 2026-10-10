import base64
import json
import math
import signal
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final

import anthropic
import httpx
import jwt
import openai
import psutil
import pytest
import yaml
from cryptography.hazmat.primitives.asymmetric import rsa
from integration._support.anthropic_thinking import JSON_OBJECT
from integration._support.client import Gateway, Scenario, eventually, object_value
from integration._support.database import write_rows
from integration._support.process import group_members, owned_proxy, owned_proxy_process
from integration._support.wire import Reply, Request, wire_server
from integration.authorization._hidden_alias_budget import (
    BUDGET,
    BUDGET_EXCEEDED,
    CHAT_REPLY,
    GATEWAY_BURST,
    PEER_BURST,
    PROXY_BUDGET_USER,
    RESPONSES_REPLY,
    AliasRig,
    alias_rig,
    anthropic_client,
    assert_free_row,
    async_anthropic_client,
    async_openai_client,
    base_url,
    chat_statuses,
    clear_provider_script,
    error_type,
    exhausted_key,
    fresh_chat,
    fresh_message,
    fresh_post,
    fresh_response,
    hidden,
    install_aliases,
    landed,
    landed_all_once,
    landed_once,
    openai_client,
    remove_aliases,
    script_provider,
    settle_candidate,
    settle_chat,
    spend_marker,
    upstream_hits,
    upstream_requests,
)
from pydantic import JsonValue

pytestmark: Final = pytest.mark.timeout(240)

# $0 comes from litellm's own cost-map row (model_prices_and_context_window.json, added 2025-10-14 in 900a57f5b1f):
# Lemonade is a local inference server, so there is no vendor price for the row to drift from. A row that gains a
# price is stale, not a regression: pick another chat row the map prices at $0 and serves over the OpenAI wire
UNPRICED_FREE_PROVIDER_MODEL: Final = "lemonade/Gemma-3-4b-it-GGUF"


@pytest.fixture(scope="module")
def rig() -> Iterator[AliasRig]:
    with alias_rig() as built:
        yield built


def test_exhausted_key_reaches_hidden_free_alias_through_openai_chat(rig: AliasRig) -> None:
    marker: Final = "chat-sync-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        key: Final = exhausted_key(rig, scenario)
        with openai_client(rig.gateway, key) as client:
            completion: Final = client.chat.completions.create(
                model=rig.hidden_free,
                messages=[{"role": "user", "content": marker}],
                extra_headers=spend_marker(marker),
            )
        assert completion.choices[0].message.content == CHAT_REPLY, completion
        assert upstream_hits(upstream_requests(rig.gateway.upstream_url), marker) == 1
        row: Final = landed_once(key, marker)
        assert row["request_id"] == completion.id, row
        assert_free_row(row, rig.hidden_free)


async def test_exhausted_key_reaches_hidden_free_alias_through_streamed_openai_chat(rig: AliasRig) -> None:
    marker: Final = "chat-stream-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        key: Final = exhausted_key(rig, scenario)
        async with async_openai_client(rig.gateway, key) as client:
            stream: Final = await client.chat.completions.create(
                model=rig.hidden_free,
                messages=[{"role": "user", "content": marker}],
                stream=True,
                stream_options={"include_usage": True},
                extra_headers=spend_marker(marker),
            )
            chunks: Final = tuple([chunk async for chunk in stream])
        assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices) == CHAT_REPLY
        assert upstream_hits(upstream_requests(rig.gateway.upstream_url), marker) == 1
        row: Final = landed_once(key, marker)
        assert row["request_id"] == chunks[0].id, row
        assert_free_row(row, rig.hidden_free)


def test_exhausted_key_reaches_hidden_free_alias_through_anthropic_messages(rig: AliasRig) -> None:
    marker: Final = "messages-sync-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        key: Final = exhausted_key(rig, scenario)
        with anthropic_client(rig.gateway, key) as client:
            message: Final = client.messages.create(
                model=rig.hidden_responses,
                max_tokens=16,
                messages=[{"role": "user", "content": marker}],
                extra_headers=spend_marker(marker),
            )
        assert [block.text for block in message.content if block.type == "text"] == [RESPONSES_REPLY], message
        assert upstream_hits(upstream_requests(rig.gateway.upstream_url), marker) == 1
        row: Final = landed_once(key, marker)
        assert row["request_id"] == message.id, row
        assert_free_row(row, rig.hidden_responses)


async def test_exhausted_key_reaches_hidden_free_alias_through_streamed_anthropic_messages(rig: AliasRig) -> None:
    marker: Final = "messages-stream-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        key: Final = exhausted_key(rig, scenario)
        async with (
            async_anthropic_client(rig.gateway, key) as client,
            client.messages.stream(
                model=rig.hidden_responses_stream,
                max_tokens=16,
                messages=[{"role": "user", "content": marker}],
                extra_headers=spend_marker(marker),
            ) as stream,
        ):
            text: Final = "".join([piece async for piece in stream.text_stream])
            final: Final = await stream.get_final_message()
        assert text == RESPONSES_REPLY, final
        assert upstream_hits(upstream_requests(rig.gateway.upstream_url), marker) == 1
        row: Final = landed_once(key, marker)
        assert row["request_id"] == final.id, row
        assert_free_row(row, rig.hidden_responses_stream)


def test_exhausted_key_reaches_hidden_free_alias_through_openai_responses(rig: AliasRig) -> None:
    marker: Final = "responses-sync-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        key: Final = exhausted_key(rig, scenario)
        with openai_client(rig.gateway, key) as client:
            response: Final = client.responses.create(
                model=rig.hidden_responses, input=marker, extra_headers=spend_marker(marker)
            )
        assert response.output_text == RESPONSES_REPLY, response
        assert upstream_hits(upstream_requests(rig.gateway.upstream_url), marker) == 1
        row: Final = landed_once(key, marker)
        assert row["request_id"] == response.id, row
        assert_free_row(row, rig.hidden_responses)


async def test_exhausted_key_reaches_hidden_free_alias_through_streamed_openai_responses(rig: AliasRig) -> None:
    marker: Final = "responses-stream-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        key: Final = exhausted_key(rig, scenario)
        async with async_openai_client(rig.gateway, key) as client:
            stream: Final = await client.responses.create(
                model=rig.hidden_responses_stream, input=marker, stream=True, extra_headers=spend_marker(marker)
            )
            events: Final = tuple([event async for event in stream])
        assert events[-1].type == "response.completed", events
        assert "".join(event.delta for event in events if event.type == "response.output_text.delta") == RESPONSES_REPLY
        assert upstream_hits(upstream_requests(rig.gateway.upstream_url), marker) == 1
        assert_free_row(landed_once(key, marker), rig.hidden_responses_stream)


def _assert_raw_chat_served(rig: AliasRig, candidate: Gateway, key: str) -> None:
    marker: Final = "raw-" + uuid.uuid4().hex
    response: Final = fresh_chat(candidate, rig.hidden_free, key, marker)
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == CHAT_REPLY, response.text
    assert upstream_hits(upstream_requests(rig.gateway.upstream_url), marker) == 1
    row: Final = landed_once(key, marker)
    assert row["request_id"] == response.json()["id"], row
    assert_free_row(row, rig.hidden_free)


def test_exhausted_key_reaches_hidden_free_alias_over_raw_http_on_both_replicas(rig: AliasRig) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = exhausted_key(rig, scenario)
        _assert_raw_chat_served(rig, rig.gateway, key)
        _assert_raw_chat_served(rig, rig.peer, key)


def _duplicate_model_post(rig: AliasRig, key: str, first: str, last: str, marker: str) -> httpx.Response:
    messages: Final = json.dumps([{"role": "user", "content": marker}])
    return httpx.post(
        f"{base_url(rig.gateway)}/v1/chat/completions",
        content=f'{{"model": {json.dumps(first)}, "model": {json.dumps(last)}, "messages": {messages}}}'.encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", **spend_marker(marker)},
        timeout=60,
        trust_env=False,
    )


def test_duplicate_model_field_is_judged_by_its_last_value(rig: AliasRig) -> None:
    free_marker: Final = "duplicate-free-" + uuid.uuid4().hex
    paid_marker: Final = "duplicate-paid-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        key: Final = exhausted_key(rig, scenario)
        served: Final = _duplicate_model_post(rig, key, rig.hidden_paid, rig.hidden_free, free_marker)
        assert served.status_code == 200, served.text
        refused: Final = _duplicate_model_post(rig, key, rig.hidden_free, rig.hidden_paid, paid_marker)
        assert refused.status_code == BUDGET_EXCEEDED, refused.text
        observed: Final = upstream_requests(rig.gateway.upstream_url)
        assert upstream_hits(observed, free_marker) == 1
        assert upstream_hits(observed, paid_marker) == 0
        assert_free_row(landed_once(key, free_marker), rig.hidden_free)


def test_provider_failure_behind_hidden_free_alias_reaches_the_caller(rig: AliasRig) -> None:
    failed_marker: Final = "provider-failure-" + uuid.uuid4().hex
    recovered_marker: Final = "provider-recovered-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        key: Final = exhausted_key(rig, scenario)
        scenario.cleanups.callback(clear_provider_script, rig)
        script_provider(rig, 1)
        failed: Final = fresh_chat(rig.gateway, rig.hidden_failing, key, failed_marker)
        assert failed.status_code == 500, failed.text
        assert "Controlled provider failure" in failed.text, failed.text
        assert "budget" not in failed.text.lower(), failed.text
        clear_provider_script(rig)
        recovered: Final = fresh_chat(rig.gateway, rig.hidden_failing, key, recovered_marker)
        assert recovered.status_code == 200, recovered.text
        observed: Final = upstream_requests(rig.gateway.upstream_url)
        assert upstream_hits(observed, failed_marker) == 1
        assert upstream_hits(observed, recovered_marker) == 1
        assert_free_row(landed_once(key, recovered_marker), rig.hidden_failing)


def test_exhausted_user_budget_still_reaches_hidden_free_alias(rig: AliasRig) -> None:
    with rig.gateway.scenario() as scenario:
        user: Final = scenario.user(max_budget=BUDGET)
        key: Final = scenario.key(user_id=user, max_budget=5.0)
        first: Final = fresh_chat(rig.gateway, rig.paid, key, "user-exhaust-" + uuid.uuid4().hex)
        assert first.status_code == 200, first.text
        settle_chat(rig, rig.paid, key, BUDGET_EXCEEDED, seconds=90)
        assert chat_statuses(rig.gateway, rig.hidden_free, key, 8) == {200}
        assert chat_statuses(rig.peer, rig.hidden_free, key, 8) == {200}
        settle_chat(rig, rig.hidden_paid, key, BUDGET_EXCEEDED, seconds=90)
        refused: Final = fresh_chat(rig.gateway, rig.hidden_paid, key, "user-refused-" + uuid.uuid4().hex)
        assert f"User={user}" in refused.text, refused.text


def test_exhausted_team_budget_still_reaches_hidden_free_alias(rig: AliasRig) -> None:
    with rig.gateway.scenario() as scenario:
        team: Final = scenario.team(max_budget=BUDGET)
        key: Final = scenario.key(team_id=team, max_budget=5.0)
        first: Final = fresh_chat(rig.gateway, rig.paid, key, "team-exhaust-" + uuid.uuid4().hex)
        assert first.status_code == 200, first.text
        settle_chat(rig, rig.paid, key, BUDGET_EXCEEDED, seconds=90)
        assert chat_statuses(rig.gateway, rig.hidden_free, key, 8) == {200}
        assert chat_statuses(rig.peer, rig.hidden_free, key, 8) == {200}
        settle_chat(rig, rig.hidden_paid, key, BUDGET_EXCEEDED, seconds=90)
        refused: Final = fresh_chat(rig.gateway, rig.hidden_paid, key, "team-refused-" + uuid.uuid4().hex)
        assert f"Team={team}" in refused.text, refused.text


def test_exhausted_tag_budget_still_reaches_hidden_free_alias(rig: AliasRig) -> None:
    tag: Final = "hidden-alias-tag-" + uuid.uuid4().hex
    tagged: Final[Mapping[str, JsonValue]] = {"metadata": {"tags": [tag]}}
    with rig.gateway.scenario() as scenario:
        rig.gateway.post("/tag/new", {"name": tag, "max_budget": BUDGET})
        scenario.cleanups.callback(rig.gateway.post, "/tag/delete", {"name": tag})
        key: Final = scenario.key(max_budget=5.0)
        first: Final = fresh_chat(rig.gateway, rig.paid, key, "tag-exhaust-" + uuid.uuid4().hex, tagged)
        assert first.status_code == 200, first.text
        settle_chat(rig, rig.paid, key, BUDGET_EXCEEDED, seconds=90, extra=tagged)
        assert chat_statuses(rig.gateway, rig.hidden_free, key, 8, tagged) == {200}
        assert chat_statuses(rig.peer, rig.hidden_free, key, 8, tagged) == {200}
        settle_chat(rig, rig.hidden_paid, key, BUDGET_EXCEEDED, seconds=90, extra=tagged)
        refused: Final = fresh_chat(rig.gateway, rig.hidden_paid, key, "tag-refused-" + uuid.uuid4().hex, tagged)
        assert f"Tag={tag}" in refused.text, refused.text
        untagged: Final = fresh_chat(rig.gateway, rig.hidden_paid, key, "tag-untagged-" + uuid.uuid4().hex)
        assert untagged.status_code == 200, untagged.text


def test_hidden_alias_repointed_between_paid_and_free_groups_follows_the_target(rig: AliasRig) -> None:
    alias: Final = "hidden-repoint-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        install_aliases(rig.gateway, {alias: hidden(rig.paid)})
        scenario.cleanups.callback(remove_aliases, rig.gateway, frozenset({alias}))
        settle_chat(rig, alias, rig.gateway.key, 200)
        key: Final = exhausted_key(rig, scenario)
        settle_chat(rig, alias, key, BUDGET_EXCEEDED)
        install_aliases(rig.gateway, {alias: hidden(rig.free)})
        settle_chat(rig, alias, key, 200)
        install_aliases(rig.gateway, {alias: hidden(rig.paid)})
        settle_chat(rig, alias, key, BUDGET_EXCEEDED)


def test_visible_alias_repointed_to_a_paid_group_loses_the_bypass(rig: AliasRig) -> None:
    alias: Final = "visible-repoint-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        install_aliases(rig.gateway, {alias: rig.free})
        scenario.cleanups.callback(remove_aliases, rig.gateway, frozenset({alias}))
        key: Final = exhausted_key(rig, scenario)
        settle_chat(rig, alias, key, 200)
        install_aliases(rig.gateway, {alias: rig.paid})
        settle_chat(rig, alias, key, BUDGET_EXCEEDED)
        install_aliases(rig.gateway, {alias: rig.free})
        settle_chat(rig, alias, key, 200)


def test_failed_free_primary_falls_back_to_hidden_free_alias_for_exhausted_key(rig: AliasRig) -> None:
    free_marker: Final = "fallback-free-" + uuid.uuid4().hex
    paid_marker: Final = "fallback-paid-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        key: Final = exhausted_key(rig, scenario)
        scenario.cleanups.callback(clear_provider_script, rig)
        script_provider(rig, 2)
        served: Final = fresh_chat(rig.gateway, rig.failing_free, key, free_marker, {"fallbacks": [rig.hidden_free]})
        assert served.status_code == 200, served.text
        assert served.headers["x-litellm-model-group"] == rig.hidden_free, dict(served.headers)
        refused: Final = fresh_chat(rig.gateway, rig.failing_free, key, paid_marker, {"fallbacks": [rig.hidden_paid]})
        assert refused.status_code == 500, refused.text
        assert "Controlled provider failure" in refused.text, refused.text
        observed: Final = upstream_requests(rig.gateway.upstream_url)
        assert upstream_hits(observed, free_marker) == 2
        assert upstream_hits(observed, paid_marker) == 1
        assert_free_row(landed_once(key, free_marker), rig.hidden_free)


def test_exhausted_key_is_served_a_cached_reply_through_hidden_free_alias(rig: AliasRig) -> None:
    marker: Final = "cache-twin-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        key: Final = exhausted_key(rig, scenario)
        first: Final = fresh_chat(rig.gateway, rig.hidden_free, key, marker)
        assert first.status_code == 200, first.text
        second: Final = fresh_chat(rig.gateway, rig.hidden_free, key, marker)
        assert second.status_code == 200, second.text
        assert second.json()["id"] == first.json()["id"], second.text
        assert "x-litellm-cache-key" in second.headers, dict(second.headers)
        assert upstream_hits(upstream_requests(rig.gateway.upstream_url), marker) == 1
        rows: Final = landed(key, marker)
        assert all(float(str(row["spend"])) == 0.0 for row in rows), rows


def _base_config() -> Mapping[str, JsonValue]:
    return JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))


def _own_config(directory: Path, name: str, section: str, value: JsonValue) -> Path:
    path: Final = directory / name
    path.write_text(yaml.safe_dump({**_base_config(), section: value}))
    return path


def _delete_proxy_budget_row(rig: AliasRig) -> None:
    deleted: Final = rig.gateway.request("POST", "/user/delete", {"user_ids": [PROXY_BUDGET_USER]})
    assert deleted.status_code == 200, deleted.text


@pytest.mark.timeout(480)
def test_exhausted_proxy_budget_still_reaches_hidden_free_alias(rig: AliasRig, tmp_path: Path) -> None:
    settings: Final = object_value(_base_config()["litellm_settings"])
    config: Final = _own_config(
        tmp_path,
        "proxy-budget.yaml",
        "litellm_settings",
        {**settings, "max_budget": BUDGET, "budget_duration": "30d"},
    )
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key()
        scenario.cleanups.callback(_delete_proxy_budget_row, rig)
        with owned_proxy(rig.gateway, tmp_path, {}, config=config, workers=2) as candidate:
            settle_candidate(candidate, rig.hidden_free, key, 200, seconds=120)
            first: Final = fresh_chat(candidate, rig.paid, key, "proxy-exhaust-" + uuid.uuid4().hex)
            assert first.status_code == 200, first.text
            settle_candidate(candidate, rig.paid, key, BUDGET_EXCEEDED, seconds=120)
            refused: Final = fresh_chat(candidate, rig.paid, key, "proxy-refused-" + uuid.uuid4().hex)
            assert error_type(refused) == "budget_exceeded", refused.text
            assert "Key=" not in refused.text, refused.text
            assert chat_statuses(candidate, rig.hidden_free, key, 8) == {200}
            settle_candidate(candidate, rig.hidden_paid, key, BUDGET_EXCEEDED, seconds=120)
        unbudgeted: Final = fresh_chat(rig.gateway, rig.paid, key, "proxy-unbudgeted-" + uuid.uuid4().hex)
        assert unbudgeted.status_code == 200, unbudgeted.text


_TAG_ADDER: Final = """from litellm.integrations.custom_guardrail import CustomGuardrail


class TagAdder(CustomGuardrail):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        metadata = data.setdefault("metadata", {})
        metadata["tags"] = [*(metadata.get("tags") or []), "__TAG__"]
        return data
"""


@pytest.mark.timeout(480)
def test_guardrail_added_tag_over_budget_still_reaches_hidden_free_alias(rig: AliasRig, tmp_path: Path) -> None:
    tag: Final = "hidden-alias-guardrail-tag-" + uuid.uuid4().hex
    module: Final = "tag_adder_" + uuid.uuid4().hex
    served_marker: Final = "guardrail-served-" + uuid.uuid4().hex
    (tmp_path / f"{module}.py").write_text(_TAG_ADDER.replace("__TAG__", tag))
    config: Final = _own_config(
        tmp_path,
        "guardrail-tag.yaml",
        "guardrails",
        [
            {
                "guardrail_name": "tag-adder-" + uuid.uuid4().hex,
                "litellm_params": {"guardrail": f"{module}.TagAdder", "mode": "pre_call", "default_on": True},
            }
        ],
    )
    tagged: Final[Mapping[str, JsonValue]] = {"metadata": {"tags": [tag]}}
    with rig.gateway.scenario() as scenario:
        rig.gateway.post("/tag/new", {"name": tag, "max_budget": BUDGET})
        scenario.cleanups.callback(rig.gateway.post, "/tag/delete", {"name": tag})
        key: Final = scenario.key(max_budget=5.0)
        first: Final = fresh_chat(rig.gateway, rig.paid, key, "guardrail-exhaust-" + uuid.uuid4().hex, tagged)
        assert first.status_code == 200, first.text
        settle_chat(rig, rig.paid, key, BUDGET_EXCEEDED, seconds=90, extra=tagged)
        untagged: Final = fresh_chat(rig.gateway, rig.hidden_paid, key, "guardrail-untagged-" + uuid.uuid4().hex)
        assert untagged.status_code == 200, untagged.text
        with owned_proxy(rig.gateway, tmp_path, {}, config=config, workers=2) as candidate:
            settle_candidate(candidate, rig.hidden_paid, key, BUDGET_EXCEEDED, seconds=120)
            refused: Final = fresh_chat(candidate, rig.hidden_paid, key, "guardrail-refused-" + uuid.uuid4().hex)
            assert f"Tag={tag}" in refused.text, refused.text
            assert chat_statuses(candidate, rig.hidden_free, key, 8) == {200}
            served: Final = fresh_chat(candidate, rig.hidden_free, key, served_marker)
            assert served.status_code == 200, served.text
            row: Final = landed_once(key, served_marker)
            assert row["request_id"] == served.json()["id"], row
            assert_free_row(row, rig.hidden_free)


def _assert_free_alias_served(rig: AliasRig, candidate: Gateway, alias: str, key: str, prefix: str) -> None:
    marker: Final = f"{prefix}-" + uuid.uuid4().hex
    response: Final = fresh_chat(candidate, alias, key, marker)
    assert response.status_code == 200, response.text
    assert_free_row(landed_once(key, marker), alias)


def test_exhausted_key_reaches_visible_free_alias(rig: AliasRig) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = exhausted_key(rig, scenario)
        _assert_free_alias_served(rig, rig.gateway, rig.visible_free, key, "visible")
        _assert_free_alias_served(rig, rig.peer, rig.visible_free, key, "visible")


def _assert_hidden_paid_refused(rig: AliasRig, candidate: Gateway, key: str) -> None:
    marker: Final = "hidden-paid-" + uuid.uuid4().hex
    response: Final = fresh_chat(candidate, rig.hidden_paid, key, marker)
    assert response.status_code == BUDGET_EXCEEDED, response.text
    assert error_type(response) == "budget_exceeded", response.text
    assert upstream_hits(upstream_requests(rig.gateway.upstream_url), marker) == 0


def test_exhausted_key_is_refused_on_hidden_paid_alias(rig: AliasRig) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = exhausted_key(rig, scenario)
        _assert_hidden_paid_refused(rig, rig.gateway, key)
        _assert_hidden_paid_refused(rig, rig.peer, key)


def test_exhausted_key_reaches_free_group_by_its_own_name(rig: AliasRig) -> None:
    marker: Final = "plain-free-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        key: Final = exhausted_key(rig, scenario)
        response: Final = fresh_chat(rig.gateway, rig.free, key, marker)
        assert response.status_code == 200, response.text
        assert_free_row(landed_once(key, marker), rig.free)


def test_key_with_headroom_is_billed_through_hidden_paid_alias(rig: AliasRig) -> None:
    paid_marker: Final = "headroom-paid-" + uuid.uuid4().hex
    free_marker: Final = "headroom-free-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(max_budget=5.0)
        paid: Final = fresh_chat(rig.gateway, rig.hidden_paid, key, paid_marker)
        assert paid.status_code == 200, paid.text
        free: Final = fresh_chat(rig.gateway, rig.hidden_free, key, free_marker)
        assert free.status_code == 200, free.text
        billed: Final = landed_once(key, paid_marker)
        assert math.isclose(float(str(billed["spend"])), 20 * 0.001 + 20 * 0.002), billed
        assert billed["model_group"] == rig.hidden_paid, billed
        assert_free_row(landed_once(key, free_marker), rig.hidden_free)


def test_key_restricted_to_the_free_group_reaches_its_hidden_alias(rig: AliasRig) -> None:
    marker: Final = "restricted-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(models=[rig.free], max_budget=5.0)
        response: Final = fresh_chat(rig.gateway, rig.hidden_free, key, marker)
        assert response.status_code == 200, response.text
        refused: Final = fresh_chat(rig.gateway, rig.hidden_paid, key, "restricted-paid-" + uuid.uuid4().hex)
        assert refused.status_code == 403, refused.text
        assert error_type(refused) == "key_model_access_denied", refused.text
        assert_free_row(landed_once(key, marker), rig.hidden_free)


@pytest.mark.parametrize("flag", ["false", "null"])
def test_alias_with_a_non_hidden_flag_keeps_the_bypass(rig: AliasRig, flag: str) -> None:
    alias: Final = {"false": rig.shown_free, "null": rig.null_hidden_free}[flag]
    with rig.gateway.scenario() as scenario:
        key: Final = exhausted_key(rig, scenario)
        _assert_free_alias_served(rig, rig.gateway, alias, key, f"flag-{flag}")
        _assert_free_alias_served(rig, rig.peer, alias, key, f"flag-{flag}")


def test_hidden_alias_to_a_group_priced_by_the_cost_map_stays_budgeted(rig: AliasRig) -> None:
    marker: Final = "unpriced-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        key: Final = exhausted_key(rig, scenario)
        response: Final = fresh_chat(rig.gateway, rig.hidden_unpriced, key, marker)
        assert response.status_code == BUDGET_EXCEEDED, response.text
        assert upstream_hits(upstream_requests(rig.gateway.upstream_url), marker) == 0


def _reported_per_token_price(rig: AliasRig, group: str) -> tuple[float, float]:
    entries: Final = rig.gateway.get("/model/info")["data"]
    assert isinstance(entries, list), entries
    infos: Final = tuple(
        object_value(object_value(entry)["model_info"])
        for entry in entries
        if object_value(entry)["model_name"] == group
    )
    assert len(infos) == 1, infos
    return float(str(infos[0]["input_cost_per_token"])), float(str(infos[0]["output_cost_per_token"]))


def _serving_deployment(candidate: Gateway, model: str) -> str | None:
    response: Final = fresh_chat(candidate, model, candidate.key, "route-" + uuid.uuid4().hex)
    return response.headers.get("x-litellm-model-id") if response.status_code == 200 else None


def _routes_alike(candidate: Gateway, name: str, target: str) -> bool:
    served: Final = _serving_deployment(candidate, name)
    return served is not None and served == _serving_deployment(candidate, target)


def _await_alias(candidate: Gateway, name: str, target: str) -> None:
    """Wait until ``candidate`` serves ``name`` from ``target``'s deployment.

    ``/router/settings`` answers from the stored config before a replica's next reload applies it, and a
    shadowing alias's name answers 200 either way, so the serving deployment is the only signal on the wire.
    """
    eventually(lambda: _routes_alike(candidate, name, target), bool, seconds=90)


def _shadowing_alias(
    rig: AliasRig, scenario: Scenario, target: str, candidates: tuple[Gateway, ...], **shadow: JsonValue
) -> str:
    """A hidden alias whose name is also a real group, live on every candidate so the name routes to ``target``."""
    name: Final = scenario.model(**shadow)
    install_aliases(rig.gateway, {name: hidden(target)})
    scenario.cleanups.callback(remove_aliases, rig.gateway, frozenset({name}))
    for candidate in candidates:
        _await_alias(candidate, name, target)
    return name


def _unpriced_free_target(rig: AliasRig, scenario: Scenario, provider_model: str = UNPRICED_FREE_PROVIDER_MODEL) -> str:
    target: Final = scenario.model(model=provider_model)
    assert _reported_per_token_price(rig, target) == (0.0, 0.0), (
        f"{provider_model} must price at $0 in the cost map with no explicit price on the deployment, "
        "or the shadowed alias is refused by the price gate and never reaches the one under test"
    )
    return target


def _shadow_over_unpriced_target(rig: AliasRig, scenario: Scenario) -> tuple[str, str]:
    target: Final = _unpriced_free_target(rig, scenario)
    shadow: Final = _shadowing_alias(
        rig, scenario, target, (rig.gateway, rig.peer), input_cost_per_token=0, output_cost_per_token=0
    )
    return target, shadow


def _refusal(response: httpx.Response) -> tuple[int, str]:
    return response.status_code, error_type(response)


def _assert_refused_like_target(rig: AliasRig, candidate: Gateway, key: str, shadow: str, target: str) -> None:
    for route, send in (("chat", fresh_chat), ("responses", fresh_response), ("messages", fresh_message)):
        shadow_marker: Final = f"shadow-{route}-" + uuid.uuid4().hex
        target_marker: Final = f"target-{route}-" + uuid.uuid4().hex
        via_shadow: Final = send(candidate, shadow, key, shadow_marker)
        by_name: Final = send(candidate, target, key, target_marker)
        assert _refusal(by_name) == (BUDGET_EXCEEDED, "budget_exceeded"), by_name.text
        assert via_shadow.status_code == by_name.status_code, via_shadow.text
        assert error_type(via_shadow) == error_type(by_name), via_shadow.text
        observed: Final = upstream_requests(rig.gateway.upstream_url)
        assert upstream_hits(observed, shadow_marker) == 0
        assert upstream_hits(observed, target_marker) == 0


def test_hidden_alias_shadowing_an_explicitly_free_group_is_refused_like_its_unpriced_target(rig: AliasRig) -> None:
    with rig.gateway.scenario() as scenario:
        target, shadow = _shadow_over_unpriced_target(rig, scenario)
        key: Final = exhausted_key(rig, scenario)
        settle_chat(rig, target, key, BUDGET_EXCEEDED)
        _assert_refused_like_target(rig, rig.gateway, key, shadow, target)
        _assert_refused_like_target(rig, rig.peer, key, shadow, target)


def _assert_refused_on_every_route(rig: AliasRig, candidate: Gateway, key: str, model: str) -> None:
    for route, send in (("chat", fresh_chat), ("responses", fresh_response), ("messages", fresh_message)):
        marker: Final = f"chain-{route}-" + uuid.uuid4().hex
        refused: Final = send(candidate, model, key, marker)
        assert refused.status_code == BUDGET_EXCEEDED, refused.text
        assert error_type(refused) == "budget_exceeded", refused.text
        assert upstream_hits(upstream_requests(rig.gateway.upstream_url), marker) == 0


def test_hidden_alias_shadow_chain_is_judged_by_the_deployment_it_is_served_from(rig: AliasRig) -> None:
    """A shadowing alias routes one hop to its target's own deployment, even when that target is an alias too."""
    with rig.gateway.scenario() as scenario:
        middle, shadow = _shadow_over_unpriced_target(rig, scenario)
        served_from: Final = _serving_deployment(rig.gateway, middle)
        install_aliases(rig.gateway, {middle: hidden(rig.free)})
        scenario.cleanups.callback(remove_aliases, rig.gateway, frozenset({middle}))
        for candidate in (rig.gateway, rig.peer):
            _await_alias(candidate, middle, rig.free)
            assert _serving_deployment(candidate, shadow) == served_from
        key: Final = exhausted_key(rig, scenario)
        for candidate in (rig.gateway, rig.peer):
            marker: Final = "chain-target-" + uuid.uuid4().hex
            by_name: Final = fresh_chat(candidate, middle, key, marker)
            assert by_name.status_code == 200, by_name.text
            assert upstream_hits(upstream_requests(rig.gateway.upstream_url), marker) == 1
            _assert_refused_on_every_route(rig, candidate, key, shadow)


def _await_route(candidate: Gateway, name: str, route: str) -> None:
    eventually(lambda: _serving_deployment(candidate, name), lambda served: served == route, seconds=90)


def _deployment_named(rig: AliasRig, scenario: Scenario, name: str, **parameters: JsonValue) -> str:
    created: Final = rig.gateway.post(
        "/model/new",
        {
            "model_name": name,
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_key": "integration-provider-key",
                "api_base": f"{rig.gateway.upstream_url}/v1",
                **parameters,
            },
            "model_info": {},
        },
    )
    identity: Final = str(object_value(created["model_info"])["id"])
    scenario.cleanups.callback(scenario.delete_model, identity)
    return identity


def _priced_wildcard_route(rig: AliasRig, scenario: Scenario, prefix: str) -> str:
    """A wildcard deployment serving every ``<prefix>/...`` name at an explicit positive per-token price."""
    return _deployment_named(
        rig, scenario, f"{prefix}/*", model="openai/*", input_cost_per_token=0.00001, output_cost_per_token=0.00002
    )


@pytest.mark.parametrize("entry_on_pattern", [False, True], ids=["entry_off_pattern", "entry_on_pattern"])
def test_hidden_alias_chain_served_by_a_priced_wildcard_route_is_refused(rig: AliasRig, entry_on_pattern: bool) -> None:
    """A chain the router serves from a priced wildcard route is budgeted at that route's price.

    The chain's middle name is only an alias key to the free group, so the group's price reads $0
    through it while the request is served by the wildcard route matching the middle name.
    """
    prefix: Final = "wild" + uuid.uuid4().hex[:8]
    middle: Final = f"{prefix}/gpt-4o-mini"
    entry: Final = (f"{prefix}/" if entry_on_pattern else "chain-entry-") + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        route: Final = _priced_wildcard_route(rig, scenario, prefix)
        install_aliases(rig.gateway, {entry: hidden(middle), middle: hidden(rig.free)})
        scenario.cleanups.callback(remove_aliases, rig.gateway, frozenset({entry, middle}))
        for candidate in (rig.gateway, rig.peer):
            _await_route(candidate, entry, route)
            _await_alias(candidate, middle, rig.free)
        key: Final = exhausted_key(rig, scenario)
        settle_chat(rig, rig.paid, key, BUDGET_EXCEEDED)
        for candidate in (rig.gateway, rig.peer):
            _assert_refused_on_every_route(rig, candidate, key, entry)


def _ptu_shadow_config(rig: AliasRig, directory: Path, name: str) -> Path:
    base: Final = _base_config()["model_list"]
    assert isinstance(base, list), base
    deployment: Final[JsonValue] = {
        "model_name": name,
        "litellm_params": {
            "model": "openai/gpt-4o-mini",
            "api_key": "integration-provider-key",
            "api_base": f"{rig.gateway.upstream_url}/v1",
            "input_cost_per_token": 0,
            "output_cost_per_token": 0,
        },
        "model_info": {"ptu_count": 100, "cost_per_ptu_per_hour": 2.0},
    }
    return _own_config(directory, "ptu-shadow.yaml", "model_list", [*base, deployment])


@pytest.mark.timeout(480)
def test_hidden_alias_shadowing_a_ptu_group_is_served_through_its_free_target(rig: AliasRig, tmp_path: Path) -> None:
    shadow: Final = "ptu-shadow-" + uuid.uuid4().hex
    marker: Final = "ptu-shadow-served-" + uuid.uuid4().hex
    config: Final = _ptu_shadow_config(rig, tmp_path, shadow)
    with rig.gateway.scenario() as scenario:
        install_aliases(rig.gateway, {shadow: hidden(rig.free)})
        scenario.cleanups.callback(remove_aliases, rig.gateway, frozenset({shadow}))
        key: Final = exhausted_key(rig, scenario)
        with owned_proxy(rig.gateway, tmp_path, {}, config=config) as candidate:
            _await_alias(candidate, shadow, rig.free)
            settle_candidate(candidate, rig.paid, key, BUDGET_EXCEEDED, seconds=120)
            served: Final = fresh_chat(candidate, shadow, key, marker)
            assert served.status_code == 200, served.text
            assert served.json()["choices"][0]["message"]["content"] == CHAT_REPLY, served.text
            assert upstream_hits(upstream_requests(rig.gateway.upstream_url), marker) == 1
            assert_free_row(landed_once(key, marker), shadow)


def test_hidden_alias_shadowing_an_explicitly_free_group_to_a_paid_target_stays_budgeted(rig: AliasRig) -> None:
    marker: Final = "paid-shadow-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        shadow: Final = _shadowing_alias(
            rig, scenario, rig.paid, (rig.gateway, rig.peer), input_cost_per_token=0, output_cost_per_token=0
        )
        key: Final = exhausted_key(rig, scenario)
        refused: Final = fresh_chat(rig.gateway, shadow, key, marker)
        assert _refusal(refused) == (BUDGET_EXCEEDED, "budget_exceeded"), refused.text
        assert upstream_hits(upstream_requests(rig.gateway.upstream_url), marker) == 0


def test_hidden_alias_to_a_missing_group_is_refused_and_the_proxy_stays_healthy(rig: AliasRig) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = exhausted_key(rig, scenario)
        refused: Final = fresh_chat(rig.gateway, rig.hidden_missing, key, "missing-" + uuid.uuid4().hex)
        assert refused.status_code == BUDGET_EXCEEDED, refused.text
        unroutable: Final = fresh_chat(rig.gateway, rig.hidden_missing, rig.gateway.key, "missing-" + uuid.uuid4().hex)
        assert unroutable.status_code == 400, unroutable.text
        assert "no healthy deployments" in unroutable.text, unroutable.text
        for candidate in (rig.gateway, rig.peer):
            assert candidate.request("GET", "/health/liveliness").status_code == 200
            assert candidate.request("GET", "/model/info").status_code == 200
            assert candidate.request("GET", "/v1/models").status_code == 200
        served: Final = fresh_chat(rig.gateway, rig.hidden_free, key, "missing-control-" + uuid.uuid4().hex)
        assert served.status_code == 200, served.text


@pytest.mark.parametrize(
    ("shape", "status"),
    [("int", BUDGET_EXCEEDED), ("list", 400), ("empty", BUDGET_EXCEEDED), ("oversized", BUDGET_EXCEEDED)],
)
def test_malformed_model_value_never_takes_the_bypass(rig: AliasRig, shape: str, status: int) -> None:
    marker: Final = f"malformed-{shape}-" + uuid.uuid4().hex
    models: Final[Mapping[str, JsonValue]] = {
        "int": 5,
        "list": [rig.hidden_free],
        "empty": "",
        "oversized": rig.hidden_free + "x" * 5120,
    }
    with rig.gateway.scenario() as scenario:
        key: Final = exhausted_key(rig, scenario)
        response: Final = rig.gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": models[shape], "messages": [{"role": "user", "content": marker}]},
            key=key,
        )
        assert response.status_code == status, response.text
        assert upstream_hits(upstream_requests(rig.gateway.upstream_url), marker) == 0
        served: Final = fresh_chat(rig.gateway, rig.hidden_free, key, "malformed-control-" + uuid.uuid4().hex)
        assert served.status_code == 200, served.text


def test_unauthenticated_request_to_hidden_alias_is_rejected(rig: AliasRig) -> None:
    marker: Final = "unauthenticated-" + uuid.uuid4().hex
    response: Final = httpx.post(
        f"{base_url(rig.gateway)}/v1/chat/completions",
        json={"model": rig.hidden_free, "messages": [{"role": "user", "content": marker}]},
        timeout=60,
        trust_env=False,
    )
    assert response.status_code == 401, response.text
    assert upstream_hits(upstream_requests(rig.gateway.upstream_url), marker) == 0


def _assert_hidden_aliases_unlisted(rig: AliasRig, candidate: Gateway) -> None:
    models: Final = candidate.get("/v1/models")["data"]
    groups: Final = candidate.get("/model_group/info")["data"]
    assert isinstance(models, list) and isinstance(groups, list)
    listed: Final = frozenset(str(object_value(entry)["id"]) for entry in models)
    described: Final = frozenset(str(object_value(entry)["model_group"]) for entry in groups)
    assert rig.visible_free in listed and rig.visible_free in described
    assert rig.shown_free in listed and rig.shown_free in described
    for name in (rig.hidden_free, rig.hidden_paid, rig.hidden_responses, rig.hidden_missing):
        assert name not in listed and name not in described, name


def test_hidden_alias_stays_out_of_model_listings(rig: AliasRig) -> None:
    _assert_hidden_aliases_unlisted(rig, rig.gateway)
    _assert_hidden_aliases_unlisted(rig, rig.peer)


def _deployment_id(rig: AliasRig, name: str) -> str:
    entries: Final = rig.gateway.get("/model/info")["data"]
    assert isinstance(entries, list), entries
    identities: Final = tuple(
        str(object_value(object_value(entry)["model_info"])["id"])
        for entry in entries
        if object_value(entry)["model_name"] == name
    )
    assert len(identities) == 1, identities
    return identities[0]


def _assert_budget_refusal(status_code: int, body: str) -> None:
    assert status_code == BUDGET_EXCEEDED, body
    assert "budget_exceeded" in body, body


def _assert_no_served_row(key: str, marker: str) -> None:
    assert all(row["status"] != "success" for row in landed(key, marker)), landed(key, marker)


def test_exhausted_key_is_refused_on_shadowing_alias_through_streamed_openai_chat(rig: AliasRig) -> None:
    marker: Final = "shadow-chat-stream-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        _, shadow = _shadow_over_unpriced_target(rig, scenario)
        key: Final = exhausted_key(rig, scenario)
        with openai_client(rig.gateway, key) as client, pytest.raises(openai.APIStatusError) as refused:
            client.chat.completions.create(
                model=shadow,
                messages=[{"role": "user", "content": marker}],
                stream=True,
                extra_headers=spend_marker(marker),
            )
        _assert_budget_refusal(refused.value.status_code, str(refused.value.body))
        assert upstream_hits(upstream_requests(rig.gateway.upstream_url), marker) == 0


async def test_exhausted_key_is_refused_on_shadowing_alias_through_streamed_openai_responses(rig: AliasRig) -> None:
    marker: Final = "shadow-responses-stream-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        _, shadow = _shadow_over_unpriced_target(rig, scenario)
        key: Final = exhausted_key(rig, scenario)
        async with async_openai_client(rig.gateway, key) as client:
            with pytest.raises(openai.APIStatusError) as refused:
                await client.responses.create(
                    model=shadow, input=marker, stream=True, extra_headers=spend_marker(marker)
                )
        _assert_budget_refusal(refused.value.status_code, str(refused.value.body))
        assert upstream_hits(upstream_requests(rig.gateway.upstream_url), marker) == 0


async def test_exhausted_key_is_refused_on_shadowing_alias_through_streamed_anthropic_messages(rig: AliasRig) -> None:
    marker: Final = "shadow-messages-stream-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        _, shadow = _shadow_over_unpriced_target(rig, scenario)
        key: Final = exhausted_key(rig, scenario)
        async with async_anthropic_client(rig.gateway, key) as client:
            with pytest.raises(anthropic.APIStatusError) as refused:
                async with client.messages.stream(
                    model=shadow,
                    max_tokens=16,
                    messages=[{"role": "user", "content": marker}],
                    extra_headers=spend_marker(marker),
                ) as stream:
                    await stream.get_final_message()
        _assert_budget_refusal(refused.value.status_code, str(refused.value.body))
        assert upstream_hits(upstream_requests(rig.gateway.upstream_url), marker) == 0


def _await_served_within(candidate: Gateway, name: str, deployments: frozenset[str]) -> None:
    eventually(lambda: _serving_deployment(candidate, name), lambda served: served in deployments, seconds=90)


def _assert_served_by(
    rig: AliasRig, candidate: Gateway, name: str, key: str, deployments: frozenset[str], prefix: str
) -> str:
    marker: Final = f"{prefix}-" + uuid.uuid4().hex
    served: Final = fresh_chat(candidate, name, key, marker)
    assert served.status_code == 200, served.text
    assert served.headers["x-litellm-model-id"] in deployments, dict(served.headers)
    assert upstream_hits(upstream_requests(rig.gateway.upstream_url), marker) == 1
    response_id: Final = str(JSON_OBJECT.validate_json(served.content)["id"])
    row: Final = landed_once(key, marker)
    assert row["request_id"] == response_id, row
    assert_free_row(row, name)
    return response_id


def test_key_with_headroom_is_served_by_the_target_through_a_shadowing_alias(rig: AliasRig) -> None:
    with rig.gateway.scenario() as scenario:
        target, shadow = _shadow_over_unpriced_target(rig, scenario)
        key: Final = scenario.key(max_budget=BUDGET)
        for candidate in (rig.gateway, rig.peer):
            _assert_served_by(rig, candidate, shadow, key, frozenset({_deployment_id(rig, target)}), "headroom-shadow")


def test_exhausted_key_reaches_hidden_alias_to_a_name_served_by_a_free_wildcard_route(rig: AliasRig) -> None:
    prefix: Final = "freewild" + uuid.uuid4().hex[:8]
    entry: Final = "free-wildcard-entry-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        route: Final = _deployment_named(
            rig, scenario, f"{prefix}/*", model="openai/*", input_cost_per_token=0, output_cost_per_token=0
        )
        install_aliases(rig.gateway, {entry: hidden(f"{prefix}/gpt-4o-mini")})
        scenario.cleanups.callback(remove_aliases, rig.gateway, frozenset({entry}))
        for candidate in (rig.gateway, rig.peer):
            _await_route(candidate, entry, route)
        key: Final = exhausted_key(rig, scenario)
        for candidate in (rig.gateway, rig.peer):
            _assert_served_by(rig, candidate, entry, key, frozenset({route}), "free-wildcard")


def test_exhausted_key_reaches_a_group_with_one_explicitly_free_deployment(rig: AliasRig) -> None:
    group: Final = "mixed-free-" + uuid.uuid4().hex
    entry: Final = "mixed-free-entry-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        members: Final = frozenset(
            {
                _deployment_named(rig, scenario, group, input_cost_per_token=0, output_cost_per_token=0),
                _deployment_named(rig, scenario, group, model=UNPRICED_FREE_PROVIDER_MODEL),
            }
        )
        install_aliases(rig.gateway, {entry: hidden(group)})
        scenario.cleanups.callback(remove_aliases, rig.gateway, frozenset({entry}))
        for candidate in (rig.gateway, rig.peer):
            _await_served_within(candidate, entry, members)
        key: Final = exhausted_key(rig, scenario)
        for candidate in (rig.gateway, rig.peer):
            for name in (group, entry):
                _assert_served_by(rig, candidate, name, key, members, "mixed")


def _script_target(rig: AliasRig, statuses: tuple[int, ...]) -> None:
    provider_model: Final = UNPRICED_FREE_PROVIDER_MODEL.split("/", 1)[1]
    scripted: Final = httpx.post(
        f"{rig.gateway.upstream_url}/__scripts/{provider_model}",
        json={"statuses": list(statuses)},
        timeout=15,
        trust_env=False,
    )
    assert scripted.status_code == 200, scripted.text


def _clear_target_script(rig: AliasRig) -> None:
    provider_model: Final = UNPRICED_FREE_PROVIDER_MODEL.split("/", 1)[1]
    cleared: Final = httpx.delete(f"{rig.gateway.upstream_url}/__scripts/{provider_model}", timeout=15, trust_env=False)
    assert cleared.status_code in (200, 404), cleared.text


def _assert_provider_failure(response: httpx.Response) -> None:
    assert response.status_code == 500, response.text
    assert "Controlled provider failure" in response.text, response.text
    assert "budget" not in response.text.lower(), response.text


def test_provider_failure_behind_a_shadowing_alias_reaches_the_caller(rig: AliasRig) -> None:
    failed_marker: Final = "shadow-provider-failure-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        target, shadow = _shadow_over_unpriced_target(rig, scenario)
        healthy: Final = scenario.key(max_budget=BUDGET)
        exhausted: Final = exhausted_key(rig, scenario)
        settle_chat(rig, target, exhausted, BUDGET_EXCEEDED)
        _assert_refused_like_target(rig, rig.gateway, exhausted, shadow, target)
        scenario.cleanups.callback(_clear_target_script, rig)
        _script_target(rig, (500,))
        _assert_provider_failure(fresh_chat(rig.gateway, shadow, healthy, failed_marker))
        assert upstream_hits(upstream_requests(rig.gateway.upstream_url), failed_marker) == 1
        _clear_target_script(rig)
        _assert_served_by(
            rig, rig.gateway, shadow, healthy, frozenset({_deployment_id(rig, target)}), "shadow-provider-recovered"
        )
        _assert_no_served_row(healthy, failed_marker)
        _assert_refused_like_target(rig, rig.gateway, exhausted, shadow, target)


def test_unauthenticated_request_to_a_shadowing_alias_is_rejected(rig: AliasRig) -> None:
    marker: Final = "shadow-unauthenticated-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        _, shadow = _shadow_over_unpriced_target(rig, scenario)
        response: Final = httpx.post(
            f"{base_url(rig.gateway)}/v1/chat/completions",
            json={"model": shadow, "messages": [{"role": "user", "content": marker}]},
            timeout=60,
            trust_env=False,
        )
        assert response.status_code == 401, response.text
        assert upstream_hits(upstream_requests(rig.gateway.upstream_url), marker) == 0


# $0 comes from litellm's own cost-map row (model_prices_and_context_window.json, added 2025-12-08 in #17692). The cell
# needs an embedding row priced $0 only by the map and served over the OpenAI wire; a row that gains a price is stale,
# not a regression: pick another such row
UNPRICED_FREE_EMBEDDING_MODEL: Final = "fireworks_ai/accounts/fireworks/models/qwen3-embedding-0p6b"


def _embed(candidate: Gateway, model: str, key: str, marker: str) -> httpx.Response:
    return fresh_post(candidate, "/v1/embeddings", {"model": model, "input": marker}, key, marker)


def _assert_embeddings_judged_like_target(
    candidate: Gateway, shadow: str, target: str, healthy: str, exhausted: str
) -> None:
    marker: Final = "embed-shadow-" + uuid.uuid4().hex
    served: Final = _embed(candidate, shadow, healthy, marker)
    assert served.status_code == 200, served.text
    vectors: Final = JSON_OBJECT.validate_json(served.content)["data"]
    assert isinstance(vectors, list) and len(vectors) == 1, served.text
    assert_free_row(landed_once(healthy, marker), shadow)
    via_shadow: Final = _embed(candidate, shadow, exhausted, "embed-refused-shadow-" + uuid.uuid4().hex)
    by_name: Final = _embed(candidate, target, exhausted, "embed-refused-target-" + uuid.uuid4().hex)
    assert _refusal(by_name) == (BUDGET_EXCEEDED, "budget_exceeded"), by_name.text
    assert _refusal(via_shadow) == _refusal(by_name), via_shadow.text


def test_exhausted_key_is_refused_on_a_shadowing_alias_through_embeddings(rig: AliasRig) -> None:
    with rig.gateway.scenario() as scenario:
        target: Final = _unpriced_free_target(rig, scenario, UNPRICED_FREE_EMBEDDING_MODEL)
        shadow: Final = _shadowing_alias(
            rig, scenario, target, (rig.gateway, rig.peer), input_cost_per_token=0, output_cost_per_token=0
        )
        healthy: Final = scenario.key(max_budget=BUDGET)
        exhausted: Final = exhausted_key(rig, scenario)
        for candidate in (rig.gateway, rig.peer):
            _assert_embeddings_judged_like_target(candidate, shadow, target, healthy, exhausted)


def test_failed_primary_does_not_fall_back_to_a_shadowing_alias_for_exhausted_key(rig: AliasRig) -> None:
    marker: Final = "fallback-shadow-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        _, shadow = _shadow_over_unpriced_target(rig, scenario)
        key: Final = exhausted_key(rig, scenario)
        scenario.cleanups.callback(clear_provider_script, rig)
        script_provider(rig, 1)
        _assert_provider_failure(fresh_chat(rig.gateway, rig.failing_free, key, marker, {"fallbacks": [shadow]}))
        assert upstream_hits(upstream_requests(rig.gateway.upstream_url), marker) == 1
        _assert_no_served_row(key, marker)


def test_removing_a_shadowing_alias_restores_the_explicitly_free_deployment(rig: AliasRig) -> None:
    with rig.gateway.scenario() as scenario:
        target, shadow = _shadow_over_unpriced_target(rig, scenario)
        real: Final = _deployment_id(rig, shadow)
        key: Final = exhausted_key(rig, scenario)
        settle_chat(rig, target, key, BUDGET_EXCEEDED)
        for candidate in (rig.gateway, rig.peer):
            _assert_refused_like_target(rig, candidate, key, shadow, target)
        remove_aliases(rig.gateway, frozenset({shadow}))
        for candidate in (rig.gateway, rig.peer):
            _await_route(candidate, shadow, real)
            _assert_served_by(rig, candidate, shadow, key, frozenset({real}), "restored")


ZERO_MODEL_BUDGET: Final[JsonValue] = {"budget_limit": 0, "time_period": "1d"}
MODEL_BUDGET_SQL: Final = 'UPDATE "LiteLLM_VerificationToken" SET model_max_budget = %s::jsonb WHERE token = %s'


def _key_with_zero_model_budgets(scenario: Scenario, names: tuple[str, ...]) -> str:
    key: Final = scenario.key(max_budget=BUDGET)
    budgets: Final = json.dumps({name: ZERO_MODEL_BUDGET for name in names})
    write_rows(MODEL_BUDGET_SQL, (budgets, sha256(key.encode()).hexdigest()))
    return key


def _assert_model_budget_refused_alike(rig: AliasRig, candidate: Gateway, key: str, shadow: str, target: str) -> None:
    shadow_marker: Final = "model-budget-shadow-" + uuid.uuid4().hex
    target_marker: Final = "model-budget-target-" + uuid.uuid4().hex
    via_shadow: Final = fresh_chat(candidate, shadow, key, shadow_marker)
    by_name: Final = fresh_chat(candidate, target, key, target_marker)
    assert by_name.status_code == BUDGET_EXCEEDED, by_name.text
    assert _refusal(via_shadow) == _refusal(by_name), via_shadow.text
    observed: Final = upstream_requests(rig.gateway.upstream_url)
    assert upstream_hits(observed, shadow_marker) == 0
    assert upstream_hits(observed, target_marker) == 0


def test_per_model_budget_on_a_shadowing_alias_is_judged_like_its_target(rig: AliasRig) -> None:
    with rig.gateway.scenario() as scenario:
        target, shadow = _shadow_over_unpriced_target(rig, scenario)
        key: Final = _key_with_zero_model_budgets(scenario, (shadow, target))
        for candidate in (rig.gateway, rig.peer):
            _assert_model_budget_refused_alike(rig, candidate, key, shadow, target)


def test_exhausted_key_is_not_served_a_cached_reply_through_a_shadowing_alias(rig: AliasRig) -> None:
    marker: Final = "shadow-cache-twin-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        _, shadow = _shadow_over_unpriced_target(rig, scenario)
        healthy: Final = scenario.key(max_budget=BUDGET)
        exhausted: Final = exhausted_key(rig, scenario)
        first: Final = fresh_chat(rig.gateway, shadow, healthy, marker)
        assert first.status_code == 200, first.text
        warmed: Final = fresh_chat(rig.gateway, shadow, healthy, marker)
        assert warmed.status_code == 200, warmed.text
        assert warmed.json()["id"] == first.json()["id"], warmed.text
        assert "x-litellm-cache-key" in warmed.headers, dict(warmed.headers)
        refused: Final = fresh_chat(rig.gateway, shadow, exhausted, marker)
        _assert_budget_refusal(refused.status_code, refused.text)
        assert "x-litellm-cache-key" not in refused.headers, dict(refused.headers)
        assert upstream_hits(upstream_requests(rig.gateway.upstream_url), marker) == 1
        _assert_no_served_row(exhausted, marker)


BUDGET_VERDICTS: Final = frozenset({200, BUDGET_EXCEEDED})


def _only_budget_verdicts(seen: frozenset[int]) -> bool:
    assert seen <= BUDGET_VERDICTS, seen
    return seen == frozenset({200})


def test_shadowing_alias_removed_under_traffic_keeps_every_answer_a_budget_verdict(rig: AliasRig) -> None:
    with rig.gateway.scenario() as scenario:
        target, shadow = _shadow_over_unpriced_target(rig, scenario)
        real: Final = _deployment_id(rig, shadow)
        key: Final = exhausted_key(rig, scenario)
        settle_chat(rig, shadow, key, BUDGET_EXCEEDED)
        remove_aliases(rig.gateway, frozenset({shadow}))
        eventually(lambda: chat_statuses(rig.gateway, shadow, key, GATEWAY_BURST), _only_budget_verdicts, seconds=90)
        eventually(lambda: chat_statuses(rig.peer, shadow, key, PEER_BURST), _only_budget_verdicts, seconds=90)
        for candidate in (rig.gateway, rig.peer):
            _assert_served_by(rig, candidate, shadow, key, frozenset({real}), "after-removal")
        assert _serving_deployment(rig.gateway, target) == _deployment_id(rig, target)


JWT_KEY_ID: Final = "integration-shadow-jwt-key"


def _jwt_config(directory: Path) -> Path:
    general: Final = object_value(_base_config()["general_settings"])
    settings: Final[JsonValue] = {
        **general,
        "enable_jwt_auth": True,
        "litellm_jwtauth": {"team_id_jwt_field": "team_id"},
    }
    return _own_config(directory, "shadow-jwt.yaml", "general_settings", settings)


def _jwk_integer(value: int) -> str:
    return base64.urlsafe_b64encode(value.to_bytes((value.bit_length() + 7) // 8, "big")).rstrip(b"=").decode()


def _jwks(private_key: rsa.RSAPrivateKey) -> bytes:
    numbers: Final = private_key.public_key().public_numbers()
    key: Final = {"kty": "RSA", "use": "sig", "alg": "RS256", "kid": JWT_KEY_ID}
    return json.dumps({"keys": [{**key, "n": _jwk_integer(numbers.n), "e": _jwk_integer(numbers.e)}]}).encode()


@pytest.mark.timeout(480)
def test_jwt_caller_over_team_budget_is_refused_on_a_shadowing_alias(rig: AliasRig, tmp_path: Path) -> None:
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwks_body: Final = _jwks(private_key)

    def respond(request: Request) -> Reply:
        assert request.method == "GET", request
        return Reply(body=jwks_body)

    with wire_server(respond) as jwks, rig.gateway.scenario() as scenario:
        target, shadow = _shadow_over_unpriced_target(rig, scenario)
        team: Final = scenario.team(max_budget=BUDGET)
        issued: Final = int(time.time())
        token: Final = jwt.encode(
            {"sub": "integration-shadow-jwt-caller", "team_id": team, "iat": issued, "exp": issued + 900},
            private_key,
            algorithm="RS256",
            headers={"kid": JWT_KEY_ID},
        )
        with owned_proxy(
            rig.gateway, tmp_path, {"JWT_PUBLIC_KEY_URL": jwks.url}, config=_jwt_config(tmp_path)
        ) as candidate:
            _await_alias(candidate, shadow, target)
            first: Final = fresh_chat(candidate, rig.paid, token, "jwt-exhaust-" + uuid.uuid4().hex)
            assert first.status_code == 200, first.text
            settle_candidate(candidate, rig.paid, token, BUDGET_EXCEEDED, seconds=120)
            settle_candidate(candidate, target, token, BUDGET_EXCEEDED, seconds=120)
            _assert_refused_like_target(rig, candidate, token, shadow, target)
            free_marker: Final = "jwt-free-" + uuid.uuid4().hex
            served: Final = fresh_chat(candidate, rig.hidden_free, token, free_marker)
            assert served.status_code == 200, served.text
            assert upstream_hits(upstream_requests(rig.gateway.upstream_url), free_marker) == 1


CUSTOM_AUTH_DOUBLE: Final = """
import os

from fastapi import Request

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth


async def user_api_key_auth(request: Request, api_key: str) -> UserAPIKeyAuth:
    if api_key == os.environ["LITELLM_MASTER_KEY"]:
        return UserAPIKeyAuth(api_key=api_key, user_role=LitellmUserRoles.PROXY_ADMIN)
    if api_key != os.environ["INTEGRATION_CUSTOM_AUTH_KEY"]:
        raise Exception("unknown caller")
    budgets = {
        name: {"budget_limit": 0, "time_period": "1d"}
        for name in os.environ["INTEGRATION_ZERO_BUDGET_MODELS"].split(",")
    }
    return UserAPIKeyAuth(api_key=api_key, user_id="integration-custom-caller", model_max_budget=budgets)
"""


def _custom_auth_config(directory: Path) -> Path:
    (directory / "custom_auth_double.py").write_text(CUSTOM_AUTH_DOUBLE)
    base: Final = _base_config()
    general: Final = object_value(base["general_settings"])
    litellm_settings: Final = object_value(base["litellm_settings"])
    path: Final = directory / "shadow-custom-auth.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                **base,
                "general_settings": {**general, "custom_auth": "custom_auth_double.user_api_key_auth"},
                "litellm_settings": {**litellm_settings, "enable_post_custom_auth_checks": True},
            }
        )
    )
    return path


@pytest.mark.timeout(480)
def test_custom_auth_caller_with_zero_model_budgets_is_refused_on_a_shadowing_alias(
    rig: AliasRig, tmp_path: Path
) -> None:
    caller: Final = "sk-custom-" + uuid.uuid4().hex
    with rig.gateway.scenario() as scenario:
        target, shadow = _shadow_over_unpriced_target(rig, scenario)
        overrides: Final = {
            "INTEGRATION_CUSTOM_AUTH_KEY": caller,
            "INTEGRATION_ZERO_BUDGET_MODELS": ",".join((shadow, target, rig.free)),
        }
        with owned_proxy(rig.gateway, tmp_path, overrides, config=_custom_auth_config(tmp_path)) as candidate:
            _await_alias(candidate, shadow, target)
            settle_candidate(candidate, target, caller, BUDGET_EXCEEDED, seconds=120)
            _assert_refused_like_target(rig, candidate, caller, shadow, target)
            free_marker: Final = "custom-free-" + uuid.uuid4().hex
            served: Final = fresh_chat(candidate, rig.free, caller, free_marker)
            assert served.status_code == 200, served.text
            assert upstream_hits(upstream_requests(rig.gateway.upstream_url), free_marker) == 1
            unknown: Final = fresh_chat(candidate, shadow, "sk-unknown-" + uuid.uuid4().hex, "custom-unknown")
            assert unknown.status_code == 401, unknown.text


BURST_SERVED: Final = 40
BURST_FAILURES: Final = 8
BURST_PROBES: Final = 8
BURST_KINDS: Final = ("chat", "chat-stream", "responses", "responses-stream", "messages", "messages-stream")


@dataclass(frozen=True, slots=True)
class _BurstCall:
    kind: str
    marker: str


def _burst_calls() -> tuple[_BurstCall, ...]:
    served: Final = tuple(
        _BurstCall(BURST_KINDS[index % len(BURST_KINDS)], "burst-" + uuid.uuid4().hex) for index in range(BURST_SERVED)
    )
    probes: Final = tuple(_BurstCall("probe", "probe-" + uuid.uuid4().hex) for _ in range(BURST_PROBES))
    liveliness: Final = tuple(_BurstCall("liveliness", "live-" + uuid.uuid4().hex) for _ in range(BURST_PROBES))
    return served + probes + liveliness


def _send_burst_call(rig: AliasRig, shadow: str, healthy: str, exhausted: str, call: _BurstCall) -> httpx.Response:
    stream: Final[Mapping[str, JsonValue]] = {"stream": True}
    match call.kind:
        case "chat":
            return fresh_chat(rig.gateway, shadow, healthy, call.marker)
        case "chat-stream":
            return fresh_chat(rig.gateway, shadow, healthy, call.marker, stream)
        case "responses":
            return fresh_response(rig.gateway, shadow, healthy, call.marker)
        case "responses-stream":
            return fresh_response(rig.gateway, shadow, healthy, call.marker, stream)
        case "messages":
            return fresh_message(rig.gateway, shadow, healthy, call.marker)
        case "messages-stream":
            return fresh_message(rig.gateway, shadow, healthy, call.marker, stream)
        case "probe":
            return fresh_chat(rig.gateway, shadow, exhausted, call.marker)
        case "liveliness":
            return httpx.get(f"{base_url(rig.gateway)}/health/liveliness", timeout=60, trust_env=False)
        case _:
            raise AssertionError(call.kind)


def _served_calls(responses: Mapping[_BurstCall, httpx.Response]) -> Mapping[_BurstCall, httpx.Response]:
    return {call: response for call, response in responses.items() if call.kind in BURST_KINDS}


def _calls_of_kind(responses: Mapping[_BurstCall, httpx.Response], kind: str) -> tuple[_BurstCall, ...]:
    return tuple(call for call in responses if call.kind == kind)


def test_shadowing_alias_burst_with_provider_outage_lands_every_served_id_once(rig: AliasRig) -> None:
    with rig.gateway.scenario() as scenario:
        _, shadow = _shadow_over_unpriced_target(rig, scenario)
        healthy: Final = scenario.key(max_budget=BUDGET)
        exhausted: Final = exhausted_key(rig, scenario)
        scenario.cleanups.callback(_clear_target_script, rig)
        _script_target(rig, (500,) * BURST_FAILURES + (200,) * BURST_SERVED)
        calls: Final = _burst_calls()

        def send(call: _BurstCall) -> httpx.Response:
            return _send_burst_call(rig, shadow, healthy, exhausted, call)

        with ThreadPoolExecutor(max_workers=len(calls)) as pool:
            responses: Final = dict(zip(calls, pool.map(send, calls), strict=True))
        served: Final = _served_calls(responses)
        failed: Final = {call: response for call, response in served.items() if response.status_code != 200}
        assert len(failed) == BURST_FAILURES, {call.kind: response.status_code for call, response in failed.items()}
        for response in failed.values():
            _assert_provider_failure(response)
        for call in _calls_of_kind(responses, "probe"):
            _assert_budget_refusal(responses[call].status_code, responses[call].text)
        for call in _calls_of_kind(responses, "liveliness"):
            assert responses[call].status_code == 200, responses[call].text
        served_markers: Final = frozenset(call.marker for call in served if call not in failed)
        assert len(served_markers) == BURST_SERVED - BURST_FAILURES
        observed: Final = upstream_requests(rig.gateway.upstream_url)
        assert all(upstream_hits(observed, call.marker) == 1 for call in served)
        assert all(upstream_hits(observed, call.marker) == 0 for call in _calls_of_kind(responses, "probe"))
        for row in landed_all_once(healthy, served_markers):
            assert_free_row(row, shadow)
        for call in failed:
            _assert_no_served_row(healthy, call.marker)
        for call in _calls_of_kind(responses, "probe"):
            _assert_no_served_row(exhausted, call.marker)


def _is_worker(process: psutil.Process, root: int) -> bool:
    try:
        return (
            process.ppid() == root
            and process.status() != psutil.STATUS_ZOMBIE
            and "spawn_main" in " ".join(process.cmdline())
        )
    except psutil.Error:
        return False


def _workers(root: int) -> tuple[psutil.Process, ...]:
    return tuple(process for process in group_members(root) if _is_worker(process, root))


def _replaced(victim: psutil.Process) -> Callable[[tuple[psutil.Process, ...]], bool]:
    return lambda workers: len(workers) == 2 and victim.pid not in {worker.pid for worker in workers}


@pytest.mark.timeout(600)
def test_killing_one_worker_keeps_the_shadowing_alias_refused(rig: AliasRig, tmp_path: Path) -> None:
    config: Final = _own_config(
        tmp_path, "shadow-two-workers.yaml", "general_settings", _base_config()["general_settings"]
    )
    with rig.gateway.scenario() as scenario:
        target, shadow = _shadow_over_unpriced_target(rig, scenario)
        healthy: Final = scenario.key(max_budget=BUDGET)
        exhausted: Final = exhausted_key(rig, scenario)
        with owned_proxy_process(rig.gateway, tmp_path, {}, config=config, workers=2) as owned:
            candidate: Final = owned.gateway
            root: Final = owned.process.pid
            before: Final = eventually(lambda: _workers(root), lambda workers: len(workers) == 2, seconds=60)
            _await_alias(candidate, shadow, target)
            settle_candidate(candidate, target, exhausted, BUDGET_EXCEEDED, seconds=120)
            settle_candidate(candidate, shadow, exhausted, BUDGET_EXCEEDED, seconds=120)
            victim: Final = before[0]
            victim.send_signal(signal.SIGKILL)
            eventually(lambda: _workers(root), _replaced(victim), seconds=120)
            settle_candidate(candidate, shadow, exhausted, BUDGET_EXCEEDED, seconds=120)
            assert chat_statuses(candidate, shadow, exhausted, GATEWAY_BURST) == {BUDGET_EXCEEDED}
            assert chat_statuses(candidate, shadow, healthy, GATEWAY_BURST) == {200}
            assert chat_statuses(candidate, target, exhausted, GATEWAY_BURST) == {BUDGET_EXCEEDED}
