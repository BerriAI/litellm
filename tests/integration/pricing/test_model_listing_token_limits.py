import uuid
from typing import Final

from integration._support.client import Gateway, object_value
from pydantic import JsonValue

from litellm import get_model_info


def _listed_model(gateway: Gateway, model: str) -> dict[str, JsonValue]:
    entries: Final = gateway.get("/v1/models")["data"]
    assert isinstance(entries, list)
    return next(object_value(entry) for entry in entries if object_value(entry)["id"] == model)


def test_v1_models_carries_cost_map_context_window_for_a_known_model(gateway: Gateway) -> None:
    catalog: Final = get_model_info("openai/gpt-4o-mini")
    assert isinstance(catalog["max_input_tokens"], int) and isinstance(catalog["max_output_tokens"], int), catalog
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        listed: Final = _listed_model(gateway, model)
        assert listed["max_input_tokens"] == catalog["max_input_tokens"], listed
        assert listed["max_output_tokens"] == catalog["max_output_tokens"], listed


def test_v1_models_carries_deployment_model_info_limits_for_an_unknown_model(gateway: Gateway) -> None:
    unknown: Final = f"openai/custom-{uuid.uuid4().hex}"
    with gateway.scenario() as scenario:
        model: Final = scenario.model(model=unknown, model_info={"max_input_tokens": 4321, "max_output_tokens": 987})
        listed: Final = _listed_model(gateway, model)
        assert listed["max_input_tokens"] == 4321, listed
        assert listed["max_output_tokens"] == 987, listed
