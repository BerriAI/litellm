"""Live e2e: Langfuse OTEL logs_spend for registry cells in logging.yaml P0.

Registry cells:
- logging.langfuse.success.logs_spend  (exercised_on chat_completions, messages, embeddings)
- logging.langfuse.failure.logs_spend  (exercised_on chat_completions, messages)
- logging.langfuse.stream.logs_spend   (exercised_on chat_completions, messages)

Integration under test is ``langfuse_otel`` (OTLP to Langfuse), not the classic
``langfuse`` SDK callback. StandardLoggingPayload.response_cost is the spend
source of truth. Generations are named ``litellm_request``; correlate by unique
prompt marker and user_api_key_alias in metadata.

Dynamic credentials by product surface:
- team: POST /team/{id}/callback with callback_name=langfuse_otel
- user/key: key metadata.logging with callback_name=langfuse_otel
- org: organization + team under it + team callback (no org-level callback API)

Extra success paths assert tool calls and applied guardrails land on the trace.
"""

from __future__ import annotations

import json

import pytest

from e2e_config import unique_marker
from e2e_http import StreamingResponse, require_successful_call
from lifecycle import ResourceManager
from logging_client import (
    INVALID_UPSTREAM_API_KEY,
    WEATHER_TOOL,
    LangfuseCreds,
    LoggingClient,
    completion_response_id,
    costs_agree,
    observation_has_guardrail,
    observation_mentions_tool,
    observation_spend,
)
from models import LiteLLMParamsBody

pytestmark = pytest.mark.e2e

DRIVER_MODEL = "e2e-langfuse-haiku"
FAIL_BACKEND = "openai/gpt-4o-mini"


def _json_blob(value: object) -> str:
    return json.dumps(value, default=str)


def _assert_logs_spend(
    client: LoggingClient,
    *,
    key: str,
    outcome: StreamingResponse,
    obs_cost: float | None,
    scope: str,
    require_positive: bool = True,
) -> None:
    """logs_spend: Langfuse cost matches StandardLogging response_cost and proxy spend.

    Non-stream responses expose response_cost on x-litellm-response-cost. Streaming
    sends headers before final cost is known, so stream paths rely on /spend/logs.
    """
    if not require_positive:
        assert obs_cost is not None, (
            f"{scope}: failure path must still track spend (0 is fine); cost={obs_cost!r}"
        )
        return

    assert obs_cost is not None and obs_cost > 0, (
        f"{scope}: Langfuse must log positive spend; calculatedTotalCost={obs_cost!r}"
    )
    # Stream responses send headers before final cost is known, so the cost header
    # is often absent; non-stream must always expose x-litellm-response-cost.
    if not outcome.is_streaming:
        assert outcome.response_cost is not None and outcome.response_cost > 0, (
            f"{scope}: proxy must return positive x-litellm-response-cost; "
            f"got {outcome.response_cost!r}"
        )
        assert costs_agree(outcome.response_cost, obs_cost), (
            f"{scope}: Langfuse cost {obs_cost!r} disagrees with "
            f"x-litellm-response-cost {outcome.response_cost!r}"
        )
    elif outcome.response_cost is not None and outcome.response_cost > 0:
        assert costs_agree(outcome.response_cost, obs_cost), (
            f"{scope}: Langfuse cost {obs_cost!r} disagrees with "
            f"x-litellm-response-cost {outcome.response_cost!r}"
        )
    spend_row = client.poll_proxy_spend_for_key(
        key,
        response_id=completion_response_id(outcome.body),
        require_positive_spend=True,
    )
    assert spend_row is not None and spend_row.spend is not None and spend_row.spend > 0, (
        f"{scope}: proxy /spend/logs never produced a positive spend row for key"
    )
    assert costs_agree(spend_row.spend, obs_cost), (
        f"{scope}: Langfuse cost {obs_cost!r} disagrees with proxy spend "
        f"{spend_row.spend!r} (request_id={spend_row.request_id!r})"
    )


class TestLangfuseTeamLogging:
    """Team-scoped callback via POST /team/{id}/callback."""

    def _team_key(
        self,
        client: LoggingClient,
        resources: ResourceManager,
        creds: LangfuseCreds,
        *,
        models: list[str],
        organization_id: str | None = None,
    ) -> tuple[str, str, str]:
        marker = unique_marker()
        key_alias = f"e2e-lf-team-key-{marker}"
        team_id = client.create_team(
            f"e2e-lf-team-{marker}",
            models=models,
            organization_id=organization_id,
        )
        resources.defer(lambda: client.delete_team(team_id))
        client.add_team_langfuse_callback(team_id, creds)
        key = client.key_with_alias(key_alias, models=models, team_id=team_id)
        resources.defer(lambda: client.delete_key(key))
        return team_id, key, key_alias

    @pytest.mark.covers("logging.langfuse.success.logs_spend", exercised_on=["chat_completions"])
    def test_success_logs_spend(
        self,
        client: LoggingClient,
        resources: ResourceManager,
        langfuse_creds: LangfuseCreds,
    ) -> None:
        _, key, key_alias = self._team_key(
            client, resources, langfuse_creds, models=[DRIVER_MODEL]
        )
        prompt_marker = unique_marker()
        outcome = client.chat_raw(
            key, DRIVER_MODEL, f"reply with one word only {prompt_marker}"
        )
        require_successful_call(outcome)

        obs = client.poll_langfuse_observation(
            langfuse_creds,
            key_alias=key_alias,
            prompt_marker=prompt_marker,
            require_positive_cost=True,
        )
        assert obs is not None, (
            f"team scope: Langfuse never received generation for key_alias={key_alias!r}"
        )
        _assert_logs_spend(
            client,
            key=key,
            outcome=outcome,
            obs_cost=observation_spend(obs),
            scope="team-success",
        )

    @pytest.mark.covers("logging.langfuse.failure.logs_spend", exercised_on=["chat_completions"])
    def test_failure_logs_spend(
        self,
        client: LoggingClient,
        resources: ResourceManager,
        langfuse_creds: LangfuseCreds,
    ) -> None:
        """Provider-auth failure still ships a Langfuse observation with spend tracked.

        Uses a throwaway deployment whose upstream OpenAI key is
        INVALID_UPSTREAM_API_KEY (not a LiteLLM virtual key).
        """
        prompt_marker = unique_marker()
        model_name = f"e2e-lf-fail-{prompt_marker}"
        model_id = client.create_model(
            model_name,
            LiteLLMParamsBody(model=FAIL_BACKEND, api_key=INVALID_UPSTREAM_API_KEY),
        )
        resources.defer(lambda: client.delete_model(model_id))

        _, key, key_alias = self._team_key(
            client, resources, langfuse_creds, models=[model_name]
        )
        outcome = client.chat_raw(key, model_name, f"this must fail {prompt_marker}")
        assert not outcome.ok, (
            f"expected upstream provider failure for {INVALID_UPSTREAM_API_KEY!r}, "
            f"got {outcome.status_code}: {outcome.body[:200]}"
        )

        obs = client.poll_langfuse_observation(
            langfuse_creds,
            key_alias=key_alias,
            prompt_marker=prompt_marker,
            require_positive_cost=False,
        )
        assert obs is not None, (
            f"team failure path: Langfuse never received generation for key_alias={key_alias!r}"
        )
        _assert_logs_spend(
            client,
            key=key,
            outcome=outcome,
            obs_cost=observation_spend(obs),
            scope="team-failure",
            require_positive=False,
        )

    @pytest.mark.covers("logging.langfuse.stream.logs_spend", exercised_on=["chat_completions"])
    def test_stream_logs_spend(
        self,
        client: LoggingClient,
        resources: ResourceManager,
        langfuse_creds: LangfuseCreds,
    ) -> None:
        _, key, key_alias = self._team_key(
            client, resources, langfuse_creds, models=[DRIVER_MODEL]
        )
        prompt_marker = unique_marker()
        outcome = client.chat_raw(
            key, DRIVER_MODEL, f"reply with one word only {prompt_marker}", stream=True
        )
        require_successful_call(outcome)
        assert outcome.is_streaming
        assert outcome.chunks > 0

        obs = client.poll_langfuse_observation(
            langfuse_creds,
            key_alias=key_alias,
            prompt_marker=prompt_marker,
            require_positive_cost=True,
        )
        assert obs is not None
        # Streamed body is elided; correlate cost via header + key spend row.
        _assert_logs_spend(
            client,
            key=key,
            outcome=outcome,
            obs_cost=observation_spend(obs),
            scope="team-stream",
        )

    @pytest.mark.covers("logging.langfuse.success.logs_spend", exercised_on=["chat_completions"])
    def test_tool_calls_logged_with_cost(
        self,
        client: LoggingClient,
        resources: ResourceManager,
        langfuse_creds: LangfuseCreds,
    ) -> None:
        _, key, key_alias = self._team_key(
            client, resources, langfuse_creds, models=[DRIVER_MODEL]
        )
        prompt_marker = unique_marker()
        outcome = client.chat_raw(
            key,
            DRIVER_MODEL,
            f"Use get_weather for Paris. marker={prompt_marker}",
            tools=[WEATHER_TOOL],
            tool_choice="required",
            max_tokens=128,
        )
        require_successful_call(outcome)
        assert "get_weather" in outcome.body or "tool_calls" in outcome.body, (
            f"gateway response must include a tool call; body={outcome.body[:300]}"
        )

        obs = client.poll_langfuse_observation(
            langfuse_creds,
            key_alias=key_alias,
            prompt_marker=prompt_marker,
            require_positive_cost=True,
        )
        assert obs is not None
        assert observation_mentions_tool(obs, "get_weather"), (
            f"Langfuse generation must record the tool; name={obs.name!r} "
            f"input={str(obs.input)[:200]} output={str(obs.output)[:200]}"
        )
        _assert_logs_spend(
            client,
            key=key,
            outcome=outcome,
            obs_cost=observation_spend(obs),
            scope="team-tools",
        )

    @pytest.mark.covers("logging.langfuse.success.logs_spend", exercised_on=["chat_completions"])
    def test_tool_permission_guardrail_logged(
        self,
        client: LoggingClient,
        resources: ResourceManager,
        langfuse_creds: LangfuseCreds,
    ) -> None:
        """tool_permission post_call guardrail must appear on the Langfuse trace
        (StandardLogging guardrail_information -> Langfuse guardrail span)."""
        marker = unique_marker()
        guardrail_name = f"e2e-lf-tool-perm-{marker}"
        guardrail_id = client.create_tool_permission_guardrail(
            guardrail_name, allowed_tool="get_weather"
        )
        resources.defer(lambda: client.delete_guardrail(guardrail_id))

        _, key, key_alias = self._team_key(
            client, resources, langfuse_creds, models=[DRIVER_MODEL]
        )
        prompt_marker = unique_marker()
        outcome = client.chat_raw(
            key,
            DRIVER_MODEL,
            f"Use get_weather for Berlin. marker={prompt_marker}",
            tools=[WEATHER_TOOL],
            tool_choice="required",
            guardrails=[guardrail_name],
            max_tokens=128,
        )
        require_successful_call(outcome)

        observations = client.poll_langfuse_trace_observations(
            langfuse_creds, key_alias=key_alias, prompt_marker=prompt_marker
        )
        assert observations, (
            f"team+guardrail: no Langfuse observations for key_alias={key_alias!r}"
        )
        gen = next(
            (
                o
                for o in observations
                if prompt_marker in _json_blob(o.input)
                or key_alias in _json_blob(o.metadata)
                or o.name in (f"litellm:{key_alias}", "litellm_request")
            ),
            observations[0],
        )
        _assert_logs_spend(
            client,
            key=key,
            outcome=outcome,
            obs_cost=observation_spend(gen),
            scope="team-guardrail",
        )
        assert any(
            observation_has_guardrail(o, guardrail_name=guardrail_name)
            or (o.name is not None and "guardrail" in o.name.lower())
            for o in observations
        ), (
            f"Langfuse trace must include applied guardrail {guardrail_name!r}; "
            f"observation names={[o.name for o in observations]}"
        )


class TestLangfuseUserKeyLogging:
    """User-owned key with metadata.logging (key-level dynamic Langfuse credentials).

    Product surface: key metadata.logging on /key/generate, not a separate
    /user/.../callback route. The key is bound to a real /user/new user_id.
    """

    def _user_key(
        self,
        client: LoggingClient,
        resources: ResourceManager,
        creds: LangfuseCreds,
        *,
        models: list[str],
    ) -> tuple[str, str, str]:
        marker = unique_marker()
        key_alias = f"e2e-lf-user-key-{marker}"
        user_id = client.create_user(
            user_email=f"e2e-lf-user-{marker}@example.com",
            user_id=f"e2e-lf-user-{marker}",
        )
        resources.defer(lambda: client.delete_user(user_id))
        key = client.key_with_alias(
            key_alias,
            models=models,
            user_id=user_id,
            metadata=creds.key_logging_metadata(),
        )
        resources.defer(lambda: client.delete_key(key))
        return user_id, key, key_alias

    @pytest.mark.covers("logging.langfuse.success.logs_spend", exercised_on=["chat_completions"])
    def test_success_logs_spend(
        self,
        client: LoggingClient,
        resources: ResourceManager,
        langfuse_creds: LangfuseCreds,
    ) -> None:
        user_id, key, key_alias = self._user_key(
            client, resources, langfuse_creds, models=[DRIVER_MODEL]
        )
        prompt_marker = unique_marker()
        outcome = client.chat_raw(
            key, DRIVER_MODEL, f"reply with one word only {prompt_marker}"
        )
        require_successful_call(outcome)

        obs = client.poll_langfuse_observation(
            langfuse_creds,
            key_alias=key_alias,
            prompt_marker=prompt_marker,
            require_positive_cost=True,
        )
        assert obs is not None, (
            f"user/key scope: Langfuse never received generation for key_alias={key_alias!r}"
        )
        meta_blob = _json_blob(obs.metadata)
        assert user_id in meta_blob or key_alias in (obs.name or ""), (
            f"user/key scope should attribute the user or key; metadata={meta_blob[:300]}"
        )
        _assert_logs_spend(
            client,
            key=key,
            outcome=outcome,
            obs_cost=observation_spend(obs),
            scope="user-key",
        )

    @pytest.mark.covers("logging.langfuse.success.logs_spend", exercised_on=["chat_completions"])
    def test_tool_calls_logged_with_cost(
        self,
        client: LoggingClient,
        resources: ResourceManager,
        langfuse_creds: LangfuseCreds,
    ) -> None:
        _, key, key_alias = self._user_key(
            client, resources, langfuse_creds, models=[DRIVER_MODEL]
        )
        prompt_marker = unique_marker()
        outcome = client.chat_raw(
            key,
            DRIVER_MODEL,
            f"Use get_weather for Tokyo. marker={prompt_marker}",
            tools=[WEATHER_TOOL],
            tool_choice="required",
            max_tokens=128,
        )
        require_successful_call(outcome)

        obs = client.poll_langfuse_observation(
            langfuse_creds,
            key_alias=key_alias,
            prompt_marker=prompt_marker,
            require_positive_cost=True,
        )
        assert obs is not None
        assert observation_mentions_tool(obs, "get_weather"), (
            f"user/key tool path: tool missing from Langfuse; output={str(obs.output)[:200]}"
        )
        _assert_logs_spend(
            client,
            key=key,
            outcome=outcome,
            obs_cost=observation_spend(obs),
            scope="user-key-tools",
        )


class TestLangfuseOrgScopedLogging:
    """Org-scoped run: organization + team under it + team Langfuse callback.

    There is no /organization/.../callback today; logging attaches at the team
    (or key) under the org. This class proves org-linked team keys still deliver
    accurate Langfuse spend and team attribution (StandardLogging metadata
    user_api_key_team_id / user_api_key_org_id).
    """

    def _org_team_key(
        self,
        client: LoggingClient,
        resources: ResourceManager,
        creds: LangfuseCreds,
        *,
        models: list[str],
    ) -> tuple[str, str, str, str]:
        marker = unique_marker()
        key_alias = f"e2e-lf-org-key-{marker}"
        org_id = client.create_org(f"e2e-lf-org-{marker}", models=models)
        resources.defer(lambda: client.delete_org(org_id))
        team_id = client.create_team(
            f"e2e-lf-org-team-{marker}",
            models=models,
            organization_id=org_id,
        )
        resources.defer(lambda: client.delete_team(team_id))
        client.add_team_langfuse_callback(team_id, creds)
        key = client.key_with_alias(
            key_alias,
            models=models,
            team_id=team_id,
            organization_id=org_id,
        )
        resources.defer(lambda: client.delete_key(key))
        return org_id, team_id, key, key_alias

    @pytest.mark.covers("logging.langfuse.success.logs_spend", exercised_on=["chat_completions"])
    def test_success_logs_spend_with_team_attribution(
        self,
        client: LoggingClient,
        resources: ResourceManager,
        langfuse_creds: LangfuseCreds,
    ) -> None:
        org_id, team_id, key, key_alias = self._org_team_key(
            client, resources, langfuse_creds, models=[DRIVER_MODEL]
        )
        prompt_marker = unique_marker()
        outcome = client.chat_raw(
            key, DRIVER_MODEL, f"reply with one word only {prompt_marker}"
        )
        require_successful_call(outcome)

        obs = client.poll_langfuse_observation(
            langfuse_creds,
            key_alias=key_alias,
            prompt_marker=prompt_marker,
            require_positive_cost=True,
        )
        assert obs is not None, (
            f"org scope: Langfuse never received generation for key_alias={key_alias!r}"
        )
        meta_blob = _json_blob(obs.metadata)
        assert team_id in meta_blob, (
            f"org-scoped team key must stamp team_id on Langfuse metadata; "
            f"team_id={team_id!r} metadata={meta_blob[:400]}"
        )
        _ = org_id
        _assert_logs_spend(
            client,
            key=key,
            outcome=outcome,
            obs_cost=observation_spend(obs),
            scope="org-team",
        )
