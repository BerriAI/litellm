from typing import Final

import litellm
from litellm.litellm_core_utils.exception_mapping_utils import keep_routing_classification
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.router_rust.surface.retries import router_policy_retries
from litellm.types.router import RetryPolicy

_POLICY: Final = RetryPolicy(RateLimitErrorRetries=3, DefaultRetries=1)


def _rate_limit() -> litellm.RateLimitError:
    return litellm.RateLimitError(message="throttlingException {}", llm_provider="bedrock", model="m")


def test_a_mapped_error_takes_the_retries_of_the_raw_error_it_keeps_for_routing() -> None:
    raw: Final = BedrockError(status_code=429, message="throttlingException {}")
    mapped: Final = _rate_limit()
    keep_routing_classification(mapped, raw)

    assert router_policy_retries(raw, _POLICY, None, None) == 1
    assert router_policy_retries(mapped, _POLICY, None, None) == 1


def test_an_error_that_keeps_nothing_for_routing_takes_its_own_class_retries() -> None:
    assert router_policy_retries(_rate_limit(), _POLICY, None, None) == 3
