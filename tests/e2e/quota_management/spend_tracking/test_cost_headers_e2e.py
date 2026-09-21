"""Live e2e: the per-component x-litellm-response-cost-* headers keep their contract.

Pins the header contract shipped in #36965: alongside the x-litellm-response-cost
total, every response carries the component costs (input, output, cache-read,
cache-creation, reasoning, tool-usage), where input covers only fresh tokens (the
cache components are subtracted out) so the components sum to the total, and
reasoning stays a subset of output.

The deployment carries distinct custom rates per component, a prime call fills the
provider's prefix cache, and the measured call re-reads it, so the cache-read
header is exercised with a real nonzero value instead of passing vacuously. The
backend is gpt-5.5 because it reports cached tokens on the second call; the
gpt-5.6 line reports cache writes and never a read, which would leave the
cache-read header at zero forever. The raw-transport send is used because the
typed chat client validates bodies and drops headers.

OpenAI publishes a primed prefix asynchronously and routes lookups by
prompt_cache_key, so a measure fired the instant the prime returns can miss a
prefix that is about to become readable. Each round pins a cache key and re-reads
the prefix it already paid to prime before spending a fresh one.
"""

import time

import pytest

from cost_rows import approx_equal, cacheable_prefix, register_priced_model
from e2e_config import unique_marker
from e2e_http import StreamingResponse
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, ChatResponse, LiteLLMParamsBody
from spend_e2e_client import SpendClient

pytestmark = pytest.mark.e2e

BACKEND = "openai/gpt-5.5"
OPENAI_API_KEY = "os.environ/OPENAI_API_KEY"
CACHE_ATTEMPTS = 3
CACHE_REREADS = 3
CACHE_SETTLE_SECONDS = 2.0

INPUT_RATE = 4e-05
OUTPUT_RATE = 8e-05
CACHE_READ_RATE = 1e-05
CACHE_WRITE_RATE = 5e-05

COMPONENT_HEADERS = (
    "x-litellm-response-cost-input",
    "x-litellm-response-cost-cache-read",
    "x-litellm-response-cost-cache-creation",
    "x-litellm-response-cost-output",
    "x-litellm-response-cost-tool-usage",
)


def _header_cost(response: StreamingResponse, name: str) -> float:
    value = response.headers.get(name)
    return float(value) if value not in (None, "", "None") else 0.0


class TestCostHeaders:
    @pytest.mark.covers("quota_management.spend_tracking.cost_headers.additive_components")
    def test_component_cost_headers_sum_to_total(
        self, client: SpendClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        model = register_priced_model(
            client.proxy,
            resources,
            "header-priced",
            LiteLLMParamsBody(
                model=BACKEND,
                api_key=OPENAI_API_KEY,
                input_cost_per_token=INPUT_RATE,
                output_cost_per_token=OUTPUT_RATE,
                cache_read_input_token_cost=CACHE_READ_RATE,
                cache_creation_input_token_cost=CACHE_WRITE_RATE,
            ),
        )

        def priced_call(content: str, cache_key: str) -> StreamingResponse:
            response = client.proxy.transport.send(
                "/chat/completions",
                headers=client.proxy.transport.bearer(scoped_key),
                json=ChatBody(
                    model=model,
                    messages=[ChatMessage(role="user", content=content)],
                    max_completion_tokens=4000,
                    prompt_cache_key=cache_key,
                ),
            )
            assert response.ok, f"chat failed (status {response.status_code}): {response.body[:300]}"
            return response

        def prime_then_reread() -> StreamingResponse | None:
            marker = unique_marker()
            prefix = cacheable_prefix(marker)
            priced_call(f"{prefix}\nReply with the single word ready.", marker)
            for _ in range(CACHE_REREADS):
                time.sleep(CACHE_SETTLE_SECONDS)
                response = priced_call(f"{prefix}\nReply with the single word measured.", marker)
                if _header_cost(response, "x-litellm-response-cost-cache-read") > 0:
                    return response
            return None

        rounds = (prime_then_reread() for _ in range(CACHE_ATTEMPTS))
        measured = next((response for response in rounds if response is not None), None)
        if measured is None:
            pytest.fail(
                f"no cache read landed across {CACHE_ATTEMPTS} prime rounds of "
                f"{CACHE_REREADS} re-reads each; the cache-read cost header was never "
                "exercised with a nonzero value"
            )

        total = measured.response_cost
        assert total is not None and total > 0, (
            f"x-litellm-response-cost missing or zero: {measured.headers}"
        )
        component_sum = sum(_header_cost(measured, name) for name in COMPONENT_HEADERS)
        assert approx_equal(component_sum, total), (
            f"component headers sum to {component_sum}, not the total {total}: "
            f"{ {name: measured.headers.get(name) for name in COMPONENT_HEADERS} }"
        )

        reasoning = _header_cost(measured, "x-litellm-response-cost-reasoning")
        output = _header_cost(measured, "x-litellm-response-cost-output")
        assert reasoning <= output * 1.01, (
            f"reasoning header {reasoning} exceeds output header {output}; "
            "reasoning must be a subset of output"
        )

        usage = ChatResponse.model_validate_json(measured.body).usage
        assert usage is not None, f"measured response carried no usage: {measured.body[:300]}"
        cached_tokens = (
            usage.prompt_tokens_details.cached_tokens or 0 if usage.prompt_tokens_details else 0
        )
        cache_creation_tokens = usage.cache_creation_input_tokens or 0
        assert cached_tokens > 0, f"cache-read header nonzero but usage shows no cached tokens: {usage}"
        assert approx_equal(
            _header_cost(measured, "x-litellm-response-cost-cache-read"),
            cached_tokens * CACHE_READ_RATE,
        ), (
            f"cache-read header {measured.headers.get('x-litellm-response-cost-cache-read')} != "
            f"{cached_tokens} cached tokens * {CACHE_READ_RATE}"
        )
        fresh_tokens = (usage.prompt_tokens or 0) - cached_tokens - cache_creation_tokens
        assert approx_equal(
            _header_cost(measured, "x-litellm-response-cost-input"), fresh_tokens * INPUT_RATE
        ), (
            f"input header {measured.headers.get('x-litellm-response-cost-input')} != "
            f"{fresh_tokens} fresh tokens * {INPUT_RATE}; the input component is not "
            "subtracting the cache components"
        )
