from typing import Final
from unittest.mock import patch

import pytest

import litellm
from litellm.exceptions import ModelNotMappedError
from litellm.router import Router


def test_get_router_model_info_returns_fallback_on_model_not_mapped_error() -> None:
    router: Final = Router(
        model_list=[
            {
                "model_name": "mcp-stdio-group",
                "litellm_params": {
                    "model": "openai/unmapped-mcp-stdio-model",
                    "api_key": "test-key",
                },
            }
        ]
    )

    with patch.object(
        litellm,
        "get_model_info",
        side_effect=ModelNotMappedError(message="Model 'unmapped-mcp-stdio-model' is not mapped", model="unmapped-mcp-stdio-model"),
    ):
        info: Final = router.get_router_model_info(model="mcp-stdio-group")

    assert info is not None
    assert info["max_tokens"] is None
    assert info["max_input_tokens"] is None
    assert info["max_output_tokens"] is None
    assert info["input_cost_per_token"] == 0
    assert info["output_cost_per_token"] == 0
    assert info["litellm_provider"] == "openai"
    assert info["mode"] == "chat"


def test_get_router_model_info_reraises_unrelated_exceptions() -> None:
    router: Final = Router(model_list=[])

    with (
        patch.object(litellm, "get_model_info", side_effect=RuntimeError("unexpected failure")),
        pytest.raises(RuntimeError, match="unexpected failure"),
    ):
        router.get_router_model_info(model="any-model")
