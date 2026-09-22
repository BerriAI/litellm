import uuid
from typing import Final

import pytest
from integration._support.client import JSON_OBJECT, Gateway, object_value, string_value


@pytest.mark.covers("quota_management.spend_tracking.spend_calculate.rejects_unpriced_model")
def test_spend_calculate_rejects_unpriced_model_with_400(gateway: Gateway) -> None:
    model: Final = f"openrouter/integration-unpriced-{uuid.uuid4().hex}"
    response: Final = gateway.request(
        "POST",
        "/spend/calculate",
        {"model": model, "messages": [{"role": "user", "content": "price this request"}]},
    )
    assert response.status_code == 400, response.text
    error: Final = object_value(JSON_OBJECT.validate_json(response.text)["error"])
    assert error["type"] == "invalid_request_error", response.text
    assert error["param"] == "model", response.text
    assert model in string_value(error["message"]), response.text
