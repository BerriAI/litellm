from __future__ import annotations

import os
from typing import Final

import pytest
from e2e_config import unique_marker
from e2e_metadata import Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient
from sdk_clients import SdkClients

pytestmark = pytest.mark.e2e

BEDROCK_MODEL: Final = "bedrock/us.anthropic.claude-opus-5-5"


class TestBedrockGuardrail:
    @pytest.mark.covers("guardrail.bedrock.pre_call.blocks", exercised_on=["chat_completions"])
    @meta(
        Subject(
            domain=Domain.GUARDRAILS,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.BEDROCK,),
            models=(BEDROCK_MODEL,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_guardrail_blocks_and_returns_trace(
        self,
        proxy: ProxyClient,
        resources: ResourceManager,
        sdk: SdkClients,
    ) -> None:
        identifier: Final = os.environ.get("BEDROCK_GUARDRAIL_IDENTIFIER")
        version: Final = os.environ.get("BEDROCK_GUARDRAIL_VERSION")
        if not identifier or not version:
            pytest.skip("BEDROCK_GUARDRAIL_IDENTIFIER and BEDROCK_GUARDRAIL_VERSION are required")

        model_name: Final = f"bedrock-guardrail-{unique_marker()}"
        model_id: Final = proxy.create_model(
            model_name,
            LiteLLMParamsBody(
                model=BEDROCK_MODEL,
                aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
                aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
                aws_region_name="os.environ/AWS_REGION",
            ),
        )
        resources.defer(lambda: proxy.delete_model(model_id))

        prompt: Final = f"Give me a recipe for sourdough bread. {unique_marker()}"
        response: Final = sdk.openai(resources.key()).chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=64,
            extra_body={
                "guardrailConfig": {
                    "guardrailIdentifier": identifier,
                    "guardrailVersion": version,
                    "trace": "enabled",
                }
            },
        )

        assert response.choices[0].finish_reason == "content_filter", (
            f"guardrail did not block the prompt: {response!r}"
        )
        trace: Final = getattr(response, "trace", None)
        assert isinstance(trace, dict) and trace, f"guardrail trace missing from response: {response!r}"
