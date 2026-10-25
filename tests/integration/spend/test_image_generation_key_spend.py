import json
from hashlib import sha256
from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server

PROMPT: Final = "a scripted sea otter"
PRICE_PER_IMAGE: Final = 0.25


def _image(request: Request) -> Reply:
    assert (request.method, request.target) == ("POST", "/images/generations")
    return Reply(body=json.dumps({"created": 1700000000, "data": [{"b64_json": "aW1n"}]}).encode())


def test_identical_image_generations_each_charge_the_key(gateway: Gateway) -> None:
    with wire_server(_image) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/dall-e-3",
            api_base=wire.url,
            api_key="synthetic-image-key",
            output_cost_per_image=PRICE_PER_IMAGE,
        )
        key: Final = scenario.key(models=[model])
        digest: Final = sha256(key.encode()).hexdigest()
        body: Final = {"model": model, "prompt": PROMPT, "size": "1024x1024", "n": 1}
        first: Final = gateway.request("POST", "/v1/images/generations", body, key=key)
        assert first.status_code == 200, first.text
        logged: Final = eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_SpendLogs" WHERE api_key=%s', (digest,)),
            lambda rows: len(rows) == 1,
            seconds=70,
        )
        charge: Final = float(str(logged[0]["spend"]))
        assert charge == pytest.approx(PRICE_PER_IMAGE)
        eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_VerificationToken" WHERE token=%s', (digest,)),
            lambda rows: float(str(rows[0]["spend"])) == pytest.approx(charge),
            seconds=70,
        )
        repeat: Final = gateway.request("POST", "/v1/images/generations", body, key=key)
        assert repeat.status_code == 200, repeat.text
        rows: Final = eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_SpendLogs" WHERE api_key=%s', (digest,)),
            lambda values: len(values) == 2,
            seconds=70,
        )
        assert [float(str(row["spend"])) for row in rows] == pytest.approx([charge, charge])
        eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_VerificationToken" WHERE token=%s', (digest,)),
            lambda values: float(str(values[0]["spend"])) == pytest.approx(2 * charge),
            seconds=70,
        )
        assert len(wire.drain()) == 2
