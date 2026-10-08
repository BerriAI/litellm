from typing import Final

import pytest
from pydantic import ValidationError

from litellm.types.proxy.model_offerings import ModelOfferingsConfig


@pytest.mark.parametrize(
    "change",
    (
        {"version": True},
        {"config_poll_seconds": 0},
        {"inventory_poll_seconds": "300"},
        {
            "offerings": [
                {"model_name": "duplicate", "source": "manual", "provider": "supplier", "upstream_model": "backend"}
            ]
            * 2
        },
        {
            "offerings": [
                {
                    "model_name": "selected",
                    "source": "manual",
                    "provider": "supplier",
                    "upstream_model": "backend",
                    "model_info": {"context_window": -1},
                }
            ]
        },
        {"providers": {"supplier": {"provider": "chatgpt", "api_key": "ignored-key"}}},
        {"providers": {"supplier": {"provider": "openai"}}},
        {"providers": {" supplier ": {"provider": "openrouter"}}},
        {"offerings": [{"model_name": "*", "source": "manual", "provider": "supplier", "upstream_model": "backend"}]},
        {
            "offerings": [
                {"model_name": "selected", "source": "manual", "provider": "missing", "upstream_model": "backend"}
            ]
        },
    ),
)
def test_invalid_operator_edits_are_rejected(change) -> None:
    config: Final = {
        "version": 1,
        "providers": {"supplier": {"provider": "openai", "api_base": "https://supplier.test/v1"}},
        "offerings": [],
    }
    with pytest.raises(ValidationError):
        ModelOfferingsConfig.model_validate({**config, **change})
