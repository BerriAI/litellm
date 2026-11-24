import uuid
from hashlib import sha256
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, object_value
from integration._support.database import read_rows


def test_a_key_bound_to_a_user_id_with_no_user_row_serves_and_attributes_spend_to_that_id(gateway: Gateway) -> None:
    user_id: Final = f"integration-absent-{uuid.uuid4().hex}"
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.001, output_cost_per_token=0.002)
        key: Final = scenario.key(user_id=user_id, models=[model])
        assert read_rows('SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id=%s', (user_id,)) == []
        response: Final = gateway.chat(model, key=key, text=f"unknown user {uuid.uuid4().hex}")
        assert object_value(response["usage"])["total_tokens"] == 40
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT "user", spend FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (str(response["id"]),)
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert rows[0]["user"] == user_id
        assert float(str(rows[0]["spend"])) == pytest.approx(0.06)
        eventually(
            lambda: read_rows(
                'SELECT spend FROM "LiteLLM_VerificationToken" WHERE token=%s', (sha256(key.encode()).hexdigest(),)
            ),
            lambda values: len(values) == 1 and float(str(values[0]["spend"])) == pytest.approx(0.06),
            seconds=70,
        )
