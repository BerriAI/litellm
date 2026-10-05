from typing import Final

import pytest
from pydantic import ValidationError

from litellm._v2.routing._types import RouterConfig


def test_config_keeps_provider_extensions_and_fallback_order() -> None:
    config: Final = RouterConfig.model_validate(
        {
            "model_list": [
                {
                    "model_name": "primary",
                    "litellm_params": {"model": "provider/model", "api_key": "private-key", "provider_option": True},
                    "model_info": {"id": "first"},
                }
            ],
            "retry": {"retries": 2, "fallbacks": {"primary": ["backup-a", "backup-b"]}},
        }
    )
    serialized: Final = config.model_dump(mode="json")
    restored: Final = RouterConfig.model_validate(serialized)

    assert restored == config
    assert restored.retry.fallbacks["primary"] == ("backup-a", "backup-b")
    assert restored.model_list[0].litellm_params.model_extra == {"provider_option": True}
    assert "private-key" not in repr(config)
    with pytest.raises(ValidationError, match="frozen"):
        setattr(config, "timeout", 30.0)


@pytest.mark.parametrize(
    "field,value",
    (
        ("timeout", 0),
        ("timeout", float("inf")),
        ("strategy", "unknown"),
        ("retry", {"retries": -1}),
        ("retry", {"retries": 2**32}),
        ("cooldown", {"allowed_failures": 0, "duration": -1}),
        ("model_list", [{"model_name": "", "litellm_params": {"model": "provider/model"}}]),
    ),
)
def test_config_rejects_invalid_routing_inputs(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        RouterConfig.model_validate({"model_list": (), field: value})
