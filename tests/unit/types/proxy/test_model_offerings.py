from typing import Final

import pytest
from pydantic import ValidationError

from litellm.types.proxy.model_offerings import ModelOfferingsConfig, OpenRouterProviderPolicy


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


@pytest.mark.parametrize(
    "change",
    (
        {"provider": "openai"},
        {"only": []},
        {"only": ["endpoint", "endpoint"]},
        {"only": ["endpoint "]},
        {"only": ["https://supplier.test"]},
        {"only": ["supplier/*"]},
        {"allow_fallbacks": True},
        {"allow_fallbacks": 0},
        {"api_key": "not-allowed"},
    ),
)
def test_provider_policy_rejects_nonexact_or_unsafe_configuration(change: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        OpenRouterProviderPolicy.model_validate(
            {"provider": "openrouter", "only": ["synthetic/fp8"], "allow_fallbacks": False, **change}
        )


@pytest.mark.parametrize("supplier,mode", (("openai", "chat"), ("openrouter", None), ("openrouter", "embedding")))
def test_provider_policy_requires_supported_supplier_and_api(supplier: str, mode: str | None) -> None:
    with pytest.raises(ValidationError, match="matching supplier and explicit chat mode"):
        ModelOfferingsConfig.model_validate(
            {
                "version": 1,
                "providers": {"supplier": {"provider": supplier, "api_base": "https://supplier.test/v1"}},
                "offerings": [
                    {
                        "model_name": "selected",
                        "source": "manual",
                        "provider": "supplier",
                        "upstream_model": "backend",
                        "model_info": {"mode": mode},
                        "provider_policy": {"provider": "openrouter", "only": ["synthetic/fp8"]},
                    }
                ],
            }
        )
