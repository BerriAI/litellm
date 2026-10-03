import json
import math
import uuid
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.anthropic_thinking import JSON_OBJECT
from integration._support.client import Gateway, object_value
from integration._support.process import owned_proxy
from integration.authorization._hidden_alias_budget import (
    BUDGET,
    BUDGET_EXCEEDED,
    CHAT_REPLY,
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
    hidden,
    install_aliases,
    landed,
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
