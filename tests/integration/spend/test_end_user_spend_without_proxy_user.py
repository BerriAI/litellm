import uuid
from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows


@pytest.mark.covers("spend.end_user.charged_when_key_has_no_user_id_and_auth_cache_is_redis")
def test_end_user_spend_lands_for_key_without_user_id_when_auth_cache_is_redis(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        key: Final = scenario.key(models=[model])
        end_user: Final = f"integration-end-user-{uuid.uuid4().hex}"
        scenario.cleanups.callback(gateway.request, "POST", "/customer/delete", {"user_ids": [end_user]})
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": f"end user spend {end_user}"}], "user": end_user},
            key=key,
        )
        assert response.status_code == 200, response.text
        assert response.json()["usage"]["total_tokens"] == 40, response.text
        charged: Final = eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_EndUserTable" WHERE user_id=%s', (end_user,)),
            lambda values: len(values) == 1 and float(values[0]["spend"]) >= 0.06,
            seconds=70,
        )
        assert float(charged[0]["spend"]) == pytest.approx(20 * 0.001 + 20 * 0.002)
