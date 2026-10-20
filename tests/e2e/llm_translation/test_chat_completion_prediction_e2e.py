from __future__ import annotations

from typing import Final

import pytest
from e2e_config import unique_marker
from e2e_metadata import Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient
from sdk_clients import NO_PROXY_CACHE, SdkClients

pytestmark = pytest.mark.e2e

OPENAI_PREDICTION_BACKEND: Final = "openai/gpt-4.1"


class TestOpenAIPrediction:
    @pytest.mark.covers("llm.chat_completions.openai.basic.nonstream.works")
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.CHAT_COMPLETIONS,
            providers=(Provider.OPENAI,),
            models=(OPENAI_PREDICTION_BACKEND,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_chat_completion_returns_prediction_usage(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model = f"e2e-openai-prediction-{unique_marker()}"
        model_id = proxy.create_model(
            model,
            LiteLLMParamsBody(model=OPENAI_PREDICTION_BACKEND, api_key="os.environ/OPENAI_API_KEY"),
        )
        resources.defer(lambda: proxy.delete_model(model_id))
        client = sdk.openai(resources.key())
        content = f"def value() -> int: return 7  # {unique_marker()}"
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"Return this code unchanged: {content}",
                }
            ],
            prediction={"type": "content", "content": content},
            extra_body=NO_PROXY_CACHE,
        )
        usage = response.usage
        assert usage is not None, f"response omitted usage: {response!r}"
        details = usage.completion_tokens_details
        assert details is not None, f"response omitted completion token details: {usage!r}"
        assert (details.accepted_prediction_tokens or 0) > 0 or (details.rejected_prediction_tokens or 0) > 0, (
            f"provider reported no predicted tokens: {details!r}"
        )
