"""Live e2e: an opted-in chat response includes the guardrail execution details."""

from __future__ import annotations

import time
from typing import Final

import pytest

from e2e_config import unique_marker
from e2e_http import unwrap
from guardrails_client import (
    BlockedWordBody,
    ContentFilterParamsBody,
    GuardrailsClient,
)
from lifecycle import ResourceManager
from models import ChatResponse, GuardrailInformationEntry

pytestmark = pytest.mark.e2e

GUARDRAIL_PROPAGATION_DEADLINE_SECONDS: Final = 40.0
GUARDRAIL_PROPAGATION_POLL_INTERVAL_SECONDS: Final = 5.0


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


def _opted_in_entries(
    client: GuardrailsClient,
    key: str,
    model: str,
    name: str,
) -> tuple[ChatResponse, tuple[GuardrailInformationEntry, ...]]:
    response = unwrap(
        client.chat(
            key,
            model,
            "Reply with the single word OK.",
            guardrails=[name],
            include_guardrail_response=True,
            max_tokens=16,
        )
    )
    entries = tuple(entry for entry in response.guardrail_information or () if entry.guardrail_name == name)
    return response, entries


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
        model = client.create_backend_model(
            resources,
            prefix="e2e-guardrail-info-backend",
            backend="openai/gpt-4.1-mini",
            api_key="os.environ/OPENAI_API_KEY",
        )
        deadline = time.monotonic() + GUARDRAIL_PROPAGATION_DEADLINE_SECONDS

        while True:
            response, entries = _opted_in_entries(client, scoped_key, model, name)
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
                    f"{GUARDRAIL_PROPAGATION_DEADLINE_SECONDS}s; response: {response}"
                )
            time.sleep(GUARDRAIL_PROPAGATION_POLL_INTERVAL_SECONDS)

    def test_without_flag_response_has_no_guardrail_information(
        self, client: GuardrailsClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        name = f"e2e-guardrail-information-default-{unique_marker()}"
        _register_content_filter(client, resources, name=name)
        model = client.create_backend_model(
            resources,
            prefix="e2e-guardrail-info-backend",
            backend="openai/gpt-4.1-mini",
            api_key="os.environ/OPENAI_API_KEY",
        )

        response = unwrap(
            client.chat(
                scoped_key,
                model,
                "Reply with the single word OK.",
                guardrails=[name],
                max_tokens=16,
            )
        )

        assert "guardrail_information" not in response.model_fields_set, (
            f"guardrail information must remain absent without include_guardrail_response, got {response}"
        )
