import pytest
from pydantic import ValidationError

from litellm.proxy.public_relay.api_types import ApiKeyCreateRequest, ModelPriceCreateRequest


def test_price_requires_a_billable_output_kind() -> None:
    with pytest.raises(ValidationError):
        ModelPriceCreateRequest(model_name="relay-model", input_micros_per_million=1)


def test_price_output_default_cannot_exceed_maximum() -> None:
    with pytest.raises(ValidationError):
        ModelPriceCreateRequest(
            model_name="relay-model",
            input_micros_per_million=1,
            output_micros_per_million=1,
            default_max_output_tokens=4097,
            max_output_tokens=4096,
        )


def test_content_logging_is_opt_in() -> None:
    assert ApiKeyCreateRequest(alias="production").log_content is False
