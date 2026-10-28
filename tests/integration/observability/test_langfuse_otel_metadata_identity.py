import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final

import pytest
from _langfuse_otel_metadata_support import (
    EXPORTED_FIELDS,
    IDENTITIES,
    assert_identity,
    chat_identity_fields,
    decoded_response_id,
    exported_llm_span,
    langfuse_rig,
    scenario_identity,
    span_response_id,
    watched_spans,
)
from integration._support.client import Gateway, eventually


def test_langfuse_otel_failure_span_carries_request_metadata(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "lf-fail-" + uuid.uuid4().hex
    with (
        langfuse_rig(gateway, tmp_path, {}, marker) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = scenario_identity(scenario, rig.provider)
        response: Final = rig.candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": identity["model"],
                "messages": [{"role": "user", "content": f"{marker} scripted -fail- upstream"}],
                "user": identity["end_user"],
                "cache": {"no-cache": True},
            },
            key=key,
        )
        assert response.status_code >= 500, response.text

        def failed_calls(spans: Sequence[Mapping[str, object]]) -> tuple[Mapping[str, object], ...]:
            return tuple(
                span
                for span in spans
                if span.get("langfuse.trace.metadata.team_id") == identity["team_id"] and "error.type" in span
            )

        failed: Final = failed_calls(
            eventually(watched_spans(rig.collector), lambda spans: bool(failed_calls(spans)), seconds=25)
        )
        for attrs in failed:
            assert_identity(attrs, chat_identity_fields(identity))


def test_langfuse_otel_oversized_caller_metadata_is_not_copied_into_the_langfuse_keys(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = "lf-bigmeta-" + uuid.uuid4().hex
    spend_logs_metadata: Final = {
        "blob": "x" * 5000,
        "count": 7,
        "steps": ["a", "b"],
        "nested": {"inner": {"leaf": True}},
    }
    caller_note: Final = "y" * 50000
    with (
        langfuse_rig(gateway, tmp_path, {}, marker) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = scenario_identity(scenario, rig.provider, spend_logs_metadata=spend_logs_metadata)
        response: Final = rig.candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": identity["model"],
                "messages": [{"role": "user", "content": marker}],
                "user": identity["end_user"],
                "metadata": {"caller_note": caller_note},
                "cache": {"no-cache": True},
            },
            key=key,
        )
        assert response.status_code == 200, response.text
        attrs: Final = exported_llm_span(rig.collector, response.json()["id"])
        assert_identity(attrs, chat_identity_fields(identity))
        assert len(str(attrs["langfuse.observation.metadata"])) < 1000, len(str(attrs["langfuse.observation.metadata"]))
        langfuse_metadata: Final = json.dumps(
            {
                name: value
                for name, value in attrs.items()
                if name == "langfuse.observation.metadata" or name.startswith("langfuse.trace.metadata.")
            }
        )
        assert caller_note not in langfuse_metadata, len(langfuse_metadata)
        assert spend_logs_metadata["blob"] not in langfuse_metadata, len(langfuse_metadata)


def test_langfuse_otel_unauthenticated_request_leaks_no_identity(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "lf-unauth-" + uuid.uuid4().hex
    with (
        langfuse_rig(gateway, tmp_path, {}, marker) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = scenario_identity(scenario, rig.provider)
        denied: Final = rig.candidate.client.post(
            "/v1/chat/completions",
            json={
                "model": identity["model"],
                "messages": [{"role": "user", "content": f"{marker}-denied"}],
            },
            headers={},
        )
        assert denied.status_code == 401, denied.text
        watch: Final = watched_spans(rig.collector)
        first: Final = rig.candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": identity["model"],
                "messages": [{"role": "user", "content": marker}],
                "cache": {"no-cache": True},
            },
            key=key,
        )
        assert first.status_code == 200, first.text
        first_id: Final = first.json()["id"]
        eventually(watch, lambda spans: sum(span_response_id(s) == first_id for s in spans) == 1, seconds=25)
        second: Final = rig.candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": identity["model"],
                "messages": [{"role": "user", "content": marker + "-settle"}],
                "cache": {"no-cache": True},
            },
            key=key,
        )
        assert second.status_code == 200, second.text
        second_id: Final = second.json()["id"]
        settled: Final = eventually(
            watch, lambda spans: sum(span_response_id(s) == second_id for s in spans) == 1, seconds=25
        )
        carriers: Final = tuple(
            span_response_id(span)
            for span in settled
            if "langfuse.observation.metadata" in span
            or any(name.startswith("langfuse.trace.metadata.user_api_key_") for name in span)
        )
        assert sorted(carriers, key=str) == sorted((first_id, second_id)), carriers


def test_langfuse_otel_v2_mapper_emits_request_metadata_and_identity(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "lf-v2-" + uuid.uuid4().hex
    with (
        langfuse_rig(
            gateway,
            tmp_path,
            {},
            marker,
        ) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = scenario_identity(scenario, rig.provider)
        response: Final = rig.candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": identity["model"],
                "messages": [{"role": "user", "content": marker}],
                "user": identity["end_user"],
                "cache": {"no-cache": True},
            },
            key=key,
        )
        assert response.status_code == 200, response.text
        watch: Final = watched_spans(rig.collector)
        observed: Final = eventually(
            watch,
            lambda spans: sum(s.get("gen_ai.response.id") == response.json()["id"] for s in spans) == 1,
            seconds=25,
        )
        attrs: Final = next(s for s in observed if s.get("gen_ai.response.id") == response.json()["id"])
        assert attrs.get("langfuse.trace.metadata.team_id") == identity["team_id"], attrs
        assert attrs.get("langfuse.trace.metadata.team_alias") == identity["team_alias"], attrs
        assert "langfuse.observation.metadata" in attrs, sorted(attrs)
        observation: Final = json.loads(str(attrs["langfuse.observation.metadata"]))
        expected: Final = chat_identity_fields(identity)
        assert {key: observation.get(key) for key in expected} == dict(expected), observation
        assert set(observation) <= EXPORTED_FIELDS, observation
        flattened: Final = {
            attribute: attrs.get(attribute)
            for attribute in (f"langfuse.trace.metadata.{field}" for field in IDENTITIES)
            if attribute in attrs
        }
        assert flattened == {f"langfuse.trace.metadata.{field}": value for field, value in expected.items()}, attrs


def test_langfuse_otel_v2_mapper_redacts_user_api_key_fields(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "lf-v2-redact-" + uuid.uuid4().hex
    with (
        langfuse_rig(
            gateway,
            tmp_path,
            {"redact_user_api_key_info": True},
            marker,
        ) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key: Final = scenario.key(
            key_alias="lf-v2-redact-" + uuid.uuid4().hex,
            metadata={"spend_logs_metadata": {"ticket": "LIT-8283"}},
        )
        model: Final = scenario.model(api_base=rig.provider.url + "/v1")
        response: Final = rig.candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": marker}],
                "cache": {"no-cache": True},
            },
            key=key,
        )
        assert response.status_code == 200, response.text
        watch: Final = watched_spans(rig.collector)
        observed: Final = eventually(
            watch,
            lambda spans: sum(s.get("gen_ai.response.id") == response.json()["id"] for s in spans) == 1,
            seconds=25,
        )
        attrs: Final = next(s for s in observed if s.get("gen_ai.response.id") == response.json()["id"])
        assert "langfuse.observation.metadata" not in attrs, attrs["langfuse.observation.metadata"]
        assert not [name for name in attrs if name.startswith("langfuse.trace.metadata.user_api_key_")], sorted(attrs)


def test_langfuse_otel_v2_mapper_teamless_key_emits_no_team_identity(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "lf-v2-noteam-" + uuid.uuid4().hex
    with (
        langfuse_rig(
            gateway,
            tmp_path,
            {},
            marker,
        ) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = scenario_identity(scenario, rig.provider, team=False)
        response: Final = rig.candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": identity["model"],
                "messages": [{"role": "user", "content": marker}],
                "cache": {"no-cache": True},
            },
            key=key,
        )
        assert response.status_code == 200, response.text
        watch: Final = watched_spans(rig.collector)
        observed: Final = eventually(
            watch,
            lambda spans: sum(s.get("gen_ai.response.id") == response.json()["id"] for s in spans) == 1,
            seconds=25,
        )
        attrs: Final = next(s for s in observed if s.get("gen_ai.response.id") == response.json()["id"])
        assert "langfuse.observation.metadata" in attrs, sorted(attrs)
        observation: Final = json.loads(str(attrs["langfuse.observation.metadata"]))
        for field in ("user_api_key_team_id", "user_api_key_team_alias", "user_api_key_end_user_id"):
            assert field not in observation, observation
        assert set(observation) <= EXPORTED_FIELDS, observation
        assert {
            attribute: attrs.get(attribute)
            for attribute in (f"langfuse.trace.metadata.{field}" for field in IDENTITIES)
            if attribute in attrs
        } == {"langfuse.trace.metadata.user_api_key_alias": identity["key_alias"]}, attrs


@pytest.mark.parametrize("route", ["/v1/chat/completions", "/v1/messages", "/v1/responses"])
def test_otel_v2_langfuse_gets_no_request_headers_or_caller_metadata(
    route: str, gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = "lf-v2-probe-" + uuid.uuid4().hex
    probe: Final = "lf-probe-" + uuid.uuid4().hex
    with (
        langfuse_rig(gateway, tmp_path, {}, marker) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key, identity = scenario_identity(scenario, rig.provider)
        model: Final = (
            scenario.model(model="anthropic/claude-sonnet-4-6", api_base=rig.provider.url)
            if route == "/v1/messages"
            else identity["model"]
        )
        body: Final = {
            "/v1/chat/completions": {
                "model": model,
                "messages": [{"role": "user", "content": marker}],
                "metadata": {"caller_note": probe},
                "cache": {"no-cache": True},
            },
            "/v1/messages": {
                "model": model,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": marker}],
                "litellm_metadata": {"caller_note": probe},
            },
            "/v1/responses": {"model": model, "input": marker, "litellm_metadata": {"caller_note": probe}},
        }[route]
        watch: Final = watched_spans(rig.collector)
        response: Final = rig.candidate.request("POST", route, body, key=key, headers={"x-lit8283-probe": probe})
        assert response.status_code == 200, response.text
        settle: Final = rig.candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": identity["model"],
                "messages": [{"role": "user", "content": marker}],
                "cache": {"no-cache": True},
            },
            key=key,
        )
        assert settle.status_code == 200, settle.text

        def llm_span(spans: Sequence[Mapping[str, object]], response_id: str) -> Mapping[str, object] | None:
            return next(
                (
                    span
                    for span in spans
                    if decoded_response_id(span.get("gen_ai.response.id")) == decoded_response_id(response_id)
                ),
                None,
            )

        exported: Final = eventually(
            watch,
            lambda spans: (
                llm_span(spans, response.json()["id"]) is not None and llm_span(spans, settle.json()["id"]) is not None
            ),
            seconds=25,
        )
        attrs: Final = llm_span(exported, response.json()["id"])
        assert attrs is not None
        observation: Final = json.loads(str(attrs["langfuse.observation.metadata"]))
        assert observation["user_api_key_alias"] == identity["key_alias"], observation
        assert set(observation) <= EXPORTED_FIELDS, observation
        assert not [span for span in exported if probe in json.dumps(span, default=str)], exported


def _keyword_router(router: str, keyword: str) -> Callable[[str], Sequence[Mapping[str, object]]]:
    def models(provider_url: str) -> Sequence[Mapping[str, object]]:
        tier: Final = f"{router}-tier"
        return (
            {
                "model_name": tier,
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "integration-provider-key",
                    "api_base": provider_url + "/v1",
                },
            },
            {
                "model_name": router,
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": {
                        "classifier_type": "heuristic",
                        "keyword_tier_rules": [{"keywords": [keyword], "tier": "REASONING"}],
                        "tiers": {"SIMPLE": tier, "MEDIUM": tier, "COMPLEX": tier, "REASONING": tier},
                    },
                },
            },
        )

    return models


def test_langfuse_otel_metadata_omits_prompt_quoting_routing_fields(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "lf-route-" + uuid.uuid4().hex
    router: Final = "lf-router-" + uuid.uuid4().hex[:8]
    keyword: Final = "acquire-globex-" + uuid.uuid4().hex[:8]
    with (
        langfuse_rig(
            gateway,
            tmp_path,
            {},
            marker,
            overrides={"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "no_content"},
            model_list=_keyword_router(router, keyword),
        ) as rig,
        rig.candidate.scenario() as scenario,
    ):
        key: Final = scenario.key(key_alias="lf-route-" + uuid.uuid4().hex)
        response: Final = rig.candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": router,
                "messages": [{"role": "user", "content": f"{marker} please {keyword} now"}],
                "cache": {"no-cache": True},
            },
            key=key,
        )
        assert response.status_code == 200, response.text

        attrs: Final = exported_llm_span(rig.collector, response.json()["id"])
        observation: Final = json.loads(str(attrs["langfuse.observation.metadata"]))
        assert set(observation) <= EXPORTED_FIELDS, observation
        langfuse_metadata: Final = {
            name: value
            for name, value in attrs.items()
            if name == "langfuse.observation.metadata" or name.startswith("langfuse.trace.metadata.")
        }
        assert keyword not in json.dumps(langfuse_metadata, default=str), langfuse_metadata
        assert keyword not in json.dumps(attrs, default=str), attrs
