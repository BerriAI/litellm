import datetime
from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.llm_response_utils.response_metadata import update_response_metadata
from litellm.types.utils import ModelResponse, Usage


def test_update_response_metadata_prices_per_second_deployment_from_its_stamped_duration(monkeypatch):
    """
    The cost cached on ``_hidden_params`` is computed after ``_response_ms`` is stamped, so a
    deployment priced only per second bills the response window it was given rather than 0 or
    whatever wider window the logging object holds.
    """
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))

    deployment_id: Final = "per-second-deployment-response-metadata"
    litellm.register_model(
        model_cost={
            deployment_id: {
                "input_cost_per_second": 0.02,
                "output_cost_per_second": 0.04,
                "litellm_provider": "openai",
                "mode": "chat",
            }
        }
    )
    start_time: Final = datetime.datetime(2026, 9, 21, 12, 0, 0)
    logging_obj: Final = Logging(
        model="gpt-5.4-nano",
        messages=[{"role": "user", "content": "Hello"}],
        stream=False,
        call_type="completion",
        start_time=start_time,
        litellm_call_id="per-second-response-metadata",
        function_id="f",
    )
    logging_obj.update_environment_variables(
        model="gpt-5.4-nano",
        litellm_params={
            "input_cost_per_second": 0.02,
            "output_cost_per_second": 0.04,
            "metadata": {"model_info": {"id": deployment_id}},
        },
        optional_params={},
        custom_llm_provider="openai",
    )
    logging_obj.model_call_details["end_time"] = start_time + datetime.timedelta(seconds=10)
    result: Final = ModelResponse(
        model="gpt-5.4-nano",
        usage=Usage(prompt_tokens=11, completion_tokens=7, total_tokens=18),
    )

    update_response_metadata(
        result=result,
        logging_obj=logging_obj,
        model="gpt-5.4-nano",
        kwargs={"model_info": {"id": deployment_id}},
        start_time=start_time,
        end_time=start_time + datetime.timedelta(seconds=2),
    )

    assert result._response_ms == pytest.approx(2000)
    assert result._hidden_params["response_cost"] == pytest.approx((0.02 + 0.04) * 2)
