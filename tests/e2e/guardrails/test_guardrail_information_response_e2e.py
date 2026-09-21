"""Live e2e: an opted-in chat response includes the guardrail execution details."""

from __future__ import annotations

import time

import pytest

from e2e_config import unique_marker
from e2e_http import Success, unwrap
from guardrails_client import (
    BlockedWordBody,
    ContentFilterParamsBody,
    GuardrailsClient,
)
from lifecycle import ResourceManager

pytestmark = pytest.mark.e2e

MODEL = "gpt-4.1-mini"
GUARDRAIL_PROPAGATION_DEADLINE_SECONDS = 40.0
GUARDRAIL_PROPAGATION_POLL_INTERVAL_SECONDS = 5.0


def _register_content_filter(client: GuardrailsClient, resources: ResourceManager, *, name: str) -> None:
    guardrail_id = client.register(
        name,
        ContentFilterParamsBody(
            mode="pre_call",
            default_on=False,
            blocked_words=[BlockedWordBody(keyword=f"never-match-{unique_marker()}", action="MASK")],
        ),
    )
    resources.defer(lambda: client.delete_guardrail(guardrail_id))


class TestGuardrailInformationResponse:
    @pytest.mark.covers(
        "guardrail.litellm_content_filter.pre_call.returns_guardrail_information",
        exercised_on=["chat_completions"],
    )
    def test_flag_returns_guardrail_information_for_the_guardrail_that_ran(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        name = f"e2e-guardrail-information-{unique_marker()}"
        _register_content_filter(client, resources, name=name)
        deadline = time.monotonic() + GUARDRAIL_PROPAGATION_DEADLINE_SECONDS
        last_result = None

        while True:
            last_result = client.chat(
                scoped_key,
                MODEL,
                "Reply with the single word OK.",
                guardrails=[name],
                include_guardrail_response=True,
                max_tokens=16,
            )
            if isinstance(last_result, Success):
                response = unwrap(last_result)
                assert response.guardrail_information is not None, (
                    f"opted-in response must include guardrail information; response: {response}"
                )
                entries = [entry for entry in response.guardrail_information if entry.guardrail_name == name]
                if len(entries) == 1:
                    entry = entries[0]
                    assert entry.guardrail_status == "success", (
                        f"guardrail information should report a successful run, got {entry!r}; response: {response}"
                    )
                    assert entry.duration is not None and entry.duration >= 0, (
                        f"guardrail information should report a non-negative duration; response: {response}"
                    )
                    return
            if time.monotonic() >= deadline:
                pytest.fail(
                    f"guardrail information did not report exactly one successful {name!r} entry within "
                    f"{GUARDRAIL_PROPAGATION_DEADLINE_SECONDS}s; response: {last_result}"
                )
            time.sleep(GUARDRAIL_PROPAGATION_POLL_INTERVAL_SECONDS)

    def test_without_flag_response_has_no_guardrail_information(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        name = f"e2e-guardrail-information-default-{unique_marker()}"
        _register_content_filter(client, resources, name=name)

        response = unwrap(
            client.chat(
                scoped_key,
                MODEL,
                "Reply with the single word OK.",
                guardrails=[name],
                max_tokens=16,
            )
        )

        assert response.guardrail_information is None, (
            f"guardrail information must remain absent without include_guardrail_response, got {response}"
        )
