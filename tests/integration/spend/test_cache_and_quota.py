import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
from typing import Final

import httpx
import pytest
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, rule, run_state_machine_as_test
from integration._support.client import Gateway, eventually, string_value
from integration._support.database import read_rows, scratch_database
from integration._support.database_relay import database_relay
from integration._support.generation import LIFECYCLE_SETTINGS, bounded_http_requests
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server


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
        assert denied.status_code == 422 and denied.json()["error"]["type"] == "budget_exceeded", denied.text
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
        assert denied_again.status_code == 422 and denied_again.json()["error"]["type"] == "budget_exceeded", (
            denied_again.text
        )
        assert upstream.get("/__observations").json()["requests"] == []


RESET_SWEEP_QUERY: Final = b'"LiteLLM_VerificationToken"."budget_reset_at" < $'


@pytest.mark.covers("quota_management.budget.key.scheduled_reset_survives_transient_db_outage")
@pytest.mark.timeout(300)
def test_scheduled_budget_reset_reconnects_after_db_transport_failure_and_unblocks_key(
    gateway: Gateway, tmp_path: Path
) -> None:
    with (
        scratch_database() as scratch_url,
        database_relay(scratch_url, RESET_SWEEP_QUERY) as (relay, relayed_url),
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
        owned_proxy(
            gateway,
            tmp_path,
            {
                "DATABASE_URL": relayed_url,
                "PROXY_BUDGET_RESCHEDULER_MIN_TIME": "30",
                "PROXY_BUDGET_RESCHEDULER_MAX_TIME": "30",
                "PRISMA_HEALTH_WATCHDOG_ENABLED": "false",
            },
        ) as candidate,
    ):
        model: Final = f"integration-{uuid.uuid4().hex}"
        candidate.post(
            "/model/new",
            {
                "model_name": model,
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "integration-provider-key",
                    "api_base": f"{gateway.upstream_url}/v1",
                    "input_cost_per_token": 0.001,
                    "output_cost_per_token": 0.002,
                },
                "model_info": {},
            },
        )
        key: Final = string_value(
            candidate.post("/key/generate", {"models": [model], "max_budget": 0.06, "budget_duration": "5s"})["key"]
        )
        digest: Final = sha256(key.encode()).hexdigest()
        row_query: Final = (
            'SELECT spend, budget_reset_at::text AS budget_reset_at FROM "LiteLLM_VerificationToken" WHERE token=%s'
        )
        assert candidate.chat(model, key=key, text=f"spend it {uuid.uuid4().hex}")["usage"]["total_tokens"] == 40
        exhausted: Final = eventually(
            lambda: read_rows(row_query, (digest,), database_url=scratch_url),
            lambda rows: len(rows) == 1 and float(rows[0]["spend"]) >= 0.06,
            seconds=70,
        )
        assert float(exhausted[0]["spend"]) == pytest.approx(0.06)
        upstream.get("/__observations").raise_for_status()
        denied: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": f"over budget {uuid.uuid4().hex}"}]},
            key=key,
        )
        assert denied.status_code == 422 and denied.json()["error"]["type"] == "budget_exceeded", denied.text
        assert upstream.get("/__observations").json()["requests"] == []
        relay.arm()
        assert relay.tripped.wait(90), "Scheduled reset sweep never reached the database"
        eventually(lambda: relay.refused, lambda count: count >= 1, seconds=30)
        reset: Final = eventually(
            lambda: read_rows(row_query, (digest,), database_url=scratch_url),
            lambda rows: len(rows) == 1 and float(rows[0]["spend"]) == 0,
            seconds=80,
            return_last_on_timeout=True,
        )
        assert len(reset) == 1 and reset[0]["spend"] == 0.0, (exhausted, reset)
        assert str(reset[0]["budget_reset_at"]) > str(exhausted[0]["budget_reset_at"]), (exhausted, reset)
        prompt: Final = f"after reset {uuid.uuid4().hex}"
        recovered: Final = candidate.request(
            "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": prompt}]}, key=key
        )
        assert recovered.status_code == 200, recovered.text
        assert recovered.json()["usage"]["total_tokens"] == 40, recovered.text
        reached: Final = upstream.get("/__observations").json()["requests"]
        assert len(reached) == 1 and reached[0]["body"]["messages"] == [{"role": "user", "content": prompt}], reached


@pytest.mark.covers("quota_management.budget.key.count_tokens_reserves_nothing_so_completion_within_budget_succeeds")
def test_repeated_count_tokens_on_budgeted_key_does_not_reserve_budget_or_block_later_completion(
    gateway: Gateway,
) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        key: Final = scenario.key(models=[model], max_budget=0.1)
        digest: Final = sha256(key.encode()).hexdigest()
        upstream.get("/__observations").raise_for_status()
        counts: Final = tuple(
            gateway.request(
                "POST",
                "/v1/messages/count_tokens",
                {"model": model, "messages": [{"role": "user", "content": "hello!!!"}]},
                key=key,
                headers={"anthropic-version": "2023-06-01"},
            )
            for _ in range(3)
        )
        for count in counts:
            assert count.status_code == 200, count.text
            assert count.json() == counts[0].json(), count.text
        input_tokens: Final = counts[0].json()["input_tokens"]
        assert isinstance(input_tokens, int) and input_tokens > 0, counts[0].text
        assert upstream.get("/__observations").json()["requests"] == []
        completion: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": f"after counting {uuid.uuid4().hex}"}]},
            key=key,
        )
        assert completion.status_code == 200, completion.text
        assert completion.json()["usage"]["total_tokens"] == 40, completion.text
        assert [request["path"] for request in upstream.get("/__observations").json()["requests"]] == [
            "/v1/chat/completions"
        ]
        spent: Final = eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_VerificationToken" WHERE token=%s', (digest,)),
            lambda values: len(values) == 1 and float(values[0]["spend"]) > 0,
            seconds=70,
        )
        assert float(spent[0]["spend"]) == pytest.approx(20 * 0.001 + 20 * 0.002)
        rows: Final = eventually(
            lambda: read_rows('SELECT call_type, spend FROM "LiteLLM_SpendLogs" WHERE api_key=%s', (digest,)),
            lambda values: len(values) >= 1,
            seconds=70,
        )
        assert [(row["call_type"], float(row["spend"])) for row in rows] == [("acompletion", pytest.approx(0.06))]


@pytest.mark.covers(
    "quota_management.budget.key.in_flight_count_tokens_reserves_nothing_so_completion_reaches_provider"
)
def test_in_flight_count_tokens_does_not_reserve_key_budget_away_from_a_completion(gateway: Gateway) -> None:
    counting_reached_provider: Final = threading.Event()
    completion_answered: Final = threading.Event()

    def respond(request: Request) -> Reply:
        counting_reached_provider.set()
        assert completion_answered.wait(timeout=30), "completion never ran while count tokens was in flight"
        return Reply(body=b'{"totalTokens": 12, "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 12}]}')

    with (
        wire_server(respond) as wire,
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
        ThreadPoolExecutor(max_workers=1) as background,
    ):
        counted: Final = scenario.model(
            model="gemini/gemini-3.8-flash",
            api_base=wire.url,
            api_key="synthetic-gemini-key",
            input_cost_per_token=0.001,
            output_cost_per_token=0.002,
        )
        completed: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        key: Final = scenario.key(models=[counted, completed], max_budget=0.06)
        contents: Final = [{"role": "user", "parts": [{"text": "hello"}]}]
        counting: Final = background.submit(
            gateway.request, "POST", f"/v1beta/models/{counted}:countTokens", {"contents": contents}, key=key
        )
        assert counting_reached_provider.wait(timeout=30), "count tokens request never reached the provider"
        upstream.get("/__observations").raise_for_status()
        prompt: Final = f"after count tokens {uuid.uuid4().hex}"
        completion: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": completed, "messages": [{"role": "user", "content": prompt}]},
            key=key,
        )
        completion_answered.set()
        count: Final = counting.result(timeout=30)
        assert completion.status_code == 200 and completion.json()["usage"]["total_tokens"] == 40, completion.text
        assert [call["body"]["messages"] for call in upstream.get("/__observations").json()["requests"]] == [
            [{"role": "user", "content": prompt}]
        ]
        assert count.status_code == 200, count.text
        assert count.json() == {"totalTokens": 12, "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 12}]}, (
            count.text
        )
        provider_calls: Final = wire.drain()
        assert [(call.method, call.target) for call in provider_calls] == [
            ("POST", "/v1beta/models/gemini-3.8-flash:countTokens")
        ]
        assert provider_calls[0].headers["x-goog-api-key"] == "synthetic-gemini-key"
        assert json.loads(provider_calls[0].body) == {"contents": contents}


@pytest.mark.covers("quota_management.response_cache.system_messages_partition_cache_identity")
def test_different_system_messages_do_not_share_a_cached_response(gateway: Gateway) -> None:
    with (
        gateway.scenario() as scenario,
        httpx.Client(base_url=gateway.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        model: Final = scenario.model()
        prompt: Final = uuid.uuid4().hex

        def completion_id(system: str, expected_calls: int) -> str:
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
            assert [call["body"]["messages"] for call in calls] == [
                [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
            ] * expected_calls, calls
            return response.json()["id"]

        first_policy_id: Final = completion_id("first policy", 1)
        second_policy_id: Final = completion_id("second policy", 1)
        assert first_policy_id != second_policy_id
        assert completion_id("first policy", 0) == first_policy_id
