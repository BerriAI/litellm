"""On launch day a Kimi K3 request through the proxy returned HTTP 200 with an empty
x-litellm-response-cost header and no spend row, because the Bedrock registry had no
complete entry for the profile the customer called. This row registers each callable
Kimi K3 inference profile through the model-management route, sends one chat completion
through the OpenAI SDK, and proves the cost header is priced from the served registry
rates and that the spend row for the request carries the same number.
"""

from typing import Final

import pytest
from e2e_config import unique_marker
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient
from sdk_clients import NO_PROXY_CACHE, SdkClients, response_header

pytestmark = pytest.mark.e2e

KIMI_K3_PROFILES: Final = ("bedrock/global.moonshotai.kimi-k3", "bedrock/us.moonshotai.kimi-k3")


def _approx_equal(actual: float, expected: float) -> bool:
    return abs(actual - expected) <= max(1e-9, abs(expected) * 1e-2)


class TestBedrockKimiK3:
    @pytest.mark.covers("llm.chat_completions.bedrock_converse.basic.nonstream.cost_logged")
    @pytest.mark.parametrize("model_id", KIMI_K3_PROFILES)
    def test_kimi_k3_is_charged(
        self,
        proxy: ProxyClient,
        sdk: SdkClients,
        scoped_key: str,
        resources: ResourceManager,
        model_id: str,
    ) -> None:
        alias: Final = f"kimi-k3-{unique_marker()}"
        created: Final = proxy.create_model(
            alias,
            LiteLLMParamsBody(
                model=model_id,
                aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
                aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
                aws_region_name="os.environ/AWS_REGION",
            ),
        )
        resources.defer(lambda: proxy.delete_model(created))

        raw: Final = sdk.openai(scoped_key).chat.completions.with_raw_response.create(
            model=alias,
            messages=[{"role": "user", "content": f"Reply pong {unique_marker()}"}],
            max_tokens=20,
            extra_body=NO_PROXY_CACHE,
        )
        completion: Final = raw.parse()

        price: Final = proxy.model_cost_map()[model_id.removeprefix("bedrock/")]
        assert price.input_cost_per_token and price.output_cost_per_token, (
            f"{model_id} has no priced row in the proxy's cost map: {price}"
        )
        usage: Final = completion.usage
        assert usage is not None, f"response carries no usage: {completion}"
        assert usage.prompt_tokens > 0 and usage.completion_tokens > 0, f"usage is empty: {usage}"
        cached: Final = (usage.prompt_tokens_details.cached_tokens or 0) if usage.prompt_tokens_details else 0
        expected: Final = (
            (usage.prompt_tokens - cached) * price.input_cost_per_token
            + cached * (price.cache_read_input_token_cost or 0.0)
            + usage.completion_tokens * price.output_cost_per_token
        )

        header_cost: Final = float(response_header(raw.headers, "x-litellm-response-cost") or 0)
        assert header_cost > 0, "HTTP 200 with an empty cost header is the launch-day symptom"
        assert _approx_equal(header_cost, expected), f"header {header_cost} vs registry {expected} at {usage}"

        rows: Final = proxy.poll_logs_for_request_id(completion.id)
        assert rows and rows[0].spend is not None and _approx_equal(rows[0].spend, header_cost), rows
