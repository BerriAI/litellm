import uuid
from contextlib import ExitStack
from hashlib import sha256
from typing import Final

import httpx
import pytest
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, rule, run_state_machine_as_test

from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.generation import LIFECYCLE_SETTINGS, bounded_http_requests


@pytest.mark.covers("quota_management.response_cache.generated_sequences_preserve_content_and_accounting")
@pytest.mark.timeout(180)
def test_generated_cache_sequences_preserve_content_usage_and_zero_hit_cost(gateway: Gateway) -> None:
    class CacheRequests(RuleBasedStateMachine):
        def __init__(self) -> None:
            super().__init__()
            self.resources = ExitStack()
            try:
                self.scenario = self.resources.enter_context(gateway.scenario())
                self.upstream = self.resources.enter_context(
                    httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False)
                )
                self.model = self.scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
                self.key = self.scenario.key(models=[self.model])
                self.prefix = uuid.uuid4().hex
                self.seen: frozenset[int] = frozenset()
                self.requests = 0
                self.paid = 0
                self.failed = False
                self.identities: dict[int, str] = {}
            except BaseException:
                with budget.cleanup():
                    self.resources.close()
                raise

        @rule(marker=st.integers(min_value=0, max_value=2))
        def request(self, marker: int) -> None:
            try:
                self.perform_request(marker)
            except BaseException:
                self.failed = True
                raise

        def perform_request(self, marker: int) -> None:
            self.upstream.get("/__observations").raise_for_status()
            response: Final = gateway.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": self.model,
                    "messages": [{"role": "user", "content": f"{self.prefix}-{marker}"}],
                },
                key=self.key,
            )
            assert response.status_code == 200, response.text
            self.requests += 1
            body: Final = response.json()
            assert (
                body["choices"][0]["message"]["content"]
                == "Hello! This is a mock response from the fake OpenAI endpoint."
            )
            assert body["usage"]["total_tokens"] == 40
            observed: Final = self.upstream.get("/__observations").json()["requests"]
            expected_calls: Final = 0 if marker in self.seen else 1
            assert len(observed) == expected_calls, observed
            if marker not in self.seen:
                assert float(response.headers["x-litellm-response-cost"]) == pytest.approx(0.06)
            if marker in self.identities:
                assert body["id"] == self.identities[marker]
            else:
                assert body["id"] not in self.identities.values()
                self.identities = {**self.identities, marker: body["id"]}
            self.paid += expected_calls
            self.seen = self.seen.union((marker,))

        def teardown(self) -> None:
            try:
                if self.requests and not self.failed:
                    rows: Final = eventually(
                        lambda: read_rows(
                            "SELECT request_id, spend, cache_hit, prompt_tokens, "
                            'completion_tokens FROM "LiteLLM_SpendLogs" WHERE api_key=%s',
                            (sha256(self.key.encode()).hexdigest(),),
                        ),
                        lambda values: len(values) == self.requests,
                        seconds=70,
                    )
                    assert len({row["request_id"] for row in rows}) == self.requests
                    assert sum(float(row["spend"]) for row in rows) == pytest.approx(self.paid * 0.06)
                    assert sum(row["cache_hit"] == "True" for row in rows) == self.requests - self.paid
                    for row in rows:
                        assert row["prompt_tokens"] == 20 and row["completion_tokens"] == 20
                        if row["cache_hit"] == "True":
                            assert float(row["spend"]) == 0 and "_cache_hit" in row["request_id"]
                            assert any(
                                row["request_id"].startswith(identity + "_cache_hit")
                                for identity in self.identities.values()
                            )
                        else:
                            assert row["request_id"] in self.identities.values()
                            assert float(row["spend"]) == pytest.approx(0.06)
            finally:
                with budget.cleanup():
                    self.resources.close()

    with bounded_http_requests((gateway,), limit=2000) as budget:
        run_state_machine_as_test(CacheRequests, settings=LIFECYCLE_SETTINGS)


@pytest.mark.covers("quota_management.response_cache.repeated_hits_preserve_identity_and_single_charge")
def test_repeated_hits_keep_response_identity_and_create_distinct_zero_cost_rows(gateway: Gateway) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        key: Final = scenario.key(models=[model])
        prompt: Final = f"repeated cache {uuid.uuid4().hex}"
        upstream.get("/__observations").raise_for_status()
        results: Final = tuple(
            gateway.post(
                "/v1/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "metadata": {"integration_marker": f"{prompt}-{index}"},
                },
                key=key,
            )
            for index in range(3)
        )
        assert len(upstream.get("/__observations").json()["requests"]) == 1
        assert len({result["id"] for result in results}) == 1
        for result in results:
            assert (
                result["choices"][0]["message"]["content"]
                == "Hello! This is a mock response from the fake OpenAI endpoint."
            )
            assert result["usage"]["total_tokens"] == 40
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT request_id, spend, cache_hit FROM "LiteLLM_SpendLogs" WHERE api_key=%s',
                (sha256(key.encode()).hexdigest(),),
            ),
            lambda values: len(values) == 3,
            seconds=70,
        )
        assert len({row["request_id"] for row in rows}) == 3
        assert sorted(float(row["spend"]) for row in rows) == [0, 0, 0.06]
        for row in rows:
            if row["cache_hit"] == "True":
                assert float(row["spend"]) == 0
                assert row["request_id"].startswith(results[0]["id"] + "_cache_hit")
            else:
                assert row["request_id"] == results[0]["id"] and float(row["spend"]) == pytest.approx(0.06)


@pytest.mark.covers("quota_management.budget.key.boundary_blocks_before_provider_and_reset_restores")
def test_key_budget_at_boundary_blocks_provider_then_explicit_reset_restores(gateway: Gateway) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        key: Final = scenario.key(models=[model], max_budget=0.06)
        control: Final = scenario.key(models=[model])
        first: Final = gateway.chat(model, key=key, text=f"budget {uuid.uuid4().hex}")
        assert first["usage"]["total_tokens"] == 40
        digest: Final = sha256(key.encode()).hexdigest()
        spent: Final = eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_VerificationToken" WHERE token=%s', (digest,)),
            lambda values: len(values) == 1 and float(values[0]["spend"]) >= 0.06,
            seconds=70,
        )
        assert float(spent[0]["spend"]) == pytest.approx(0.06)
        upstream.get("/__observations").raise_for_status()
        denied: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": f"over budget {uuid.uuid4().hex}"}]},
            key=key,
        )
        assert denied.status_code == 429 and denied.json()["error"]["type"] == "budget_exceeded", denied.text
        assert upstream.get("/__observations").json()["requests"] == []
        assert gateway.chat(model, key=control, text=f"control {uuid.uuid4().hex}")["usage"]["total_tokens"] == 40
        gateway.post("/key/update", {"key": key, "spend": 0})
        assert read_rows('SELECT spend, max_budget FROM "LiteLLM_VerificationToken" WHERE token=%s', (digest,)) == [
            {"spend": 0.0, "max_budget": 0.06}
        ]
        assert gateway.chat(model, key=key, text=f"reset {uuid.uuid4().hex}")["usage"]["total_tokens"] == 40
        eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_VerificationToken" WHERE token=%s', (digest,)),
            lambda values: len(values) == 1 and float(values[0]["spend"]) >= 0.06,
            seconds=70,
        )
        upstream.get("/__observations").raise_for_status()
        denied_again: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": f"boundary again {uuid.uuid4().hex}"}]},
            key=key,
        )
        assert denied_again.status_code == 429 and denied_again.json()["error"]["type"] == "budget_exceeded", (
            denied_again.text
        )
        assert upstream.get("/__observations").json()["requests"] == []


@pytest.mark.covers("quota_management.response_cache.system_messages_partition_cache_identity")
def test_different_system_messages_do_not_share_a_cached_response(gateway: Gateway) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        model: Final = scenario.model()
        prompt: Final = uuid.uuid4().hex
        identities: dict[str, str] = {}
        for system, expected_calls in (("first policy", 1), ("second policy", 1), ("first policy", 0)):
            upstream.get("/__observations").raise_for_status()
            response: Final = gateway.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                },
            )
            assert response.status_code == 200 and response.json()["usage"]["total_tokens"] == 40, response.text
            calls: Final = upstream.get("/__observations").json()["requests"]
            assert len(calls) == expected_calls
            if system in identities:
                assert response.json()["id"] == identities[system]
            else:
                assert response.json()["id"] not in identities.values()
                identities = {**identities, system: response.json()["id"]}
            if calls:
                assert calls[0]["body"]["messages"] == [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ]
