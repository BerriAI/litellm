import json
import uuid
from hashlib import sha256
from pathlib import Path
from typing import Final

import litellm
import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from openai import OpenAI
from openai.types import ContainerCreateResponse

# run.py sets LITELLM_LOCAL_MODEL_COST_MAP=True and reads model_prices_and_context_window_backup.json
CODE_INTERPRETER_SESSION_COST: Final = float(
    json.loads((Path(litellm.__file__).parent / "model_prices_and_context_window_backup.json").read_text())[
        "openai/container"
    ]["code_interpreter_cost_per_session"]
)
PROVIDER_KEY: Final = "synthetic-container-key"


def test_container_create_charges_one_code_interpreter_session(gateway: Gateway) -> None:
    assert CODE_INTERPRETER_SESSION_COST > 0
    container_id: Final = f"cntr_spend_{uuid.uuid4().hex}"
    scripted: Final = {"id": container_id, "object": "container", "created_at": 1, "status": "running", "name": "c"}

    def respond(request: Request) -> Reply:
        return Reply(body=json.dumps(scripted).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="openai/gpt-4o-mini", api_base=f"{wire.url}/v1", api_key=PROVIDER_KEY)
        key: Final = scenario.key()
        digest: Final = sha256(key.encode()).hexdigest()
        client: Final = OpenAI(base_url=str(gateway.client.base_url.join("/v1")), api_key=key, max_retries=0)
        eventually(lambda: tuple(entry.id for entry in client.models.list()), lambda ids: model in ids)

        raw: Final = client.containers.with_raw_response.create(
            name="c",
            expires_after={"anchor": "last_active_at", "minutes": 5},
            file_ids=["file-spend"],
            extra_body={"model": model},
        )
        assert raw.http_response.status_code == 200, raw.http_response.text
        created: Final = raw.parse()
        assert created.model_copy(update={"id": container_id}) == ContainerCreateResponse.model_validate(scripted), (
            raw.http_response.text
        )
        assert float(raw.headers["x-litellm-response-cost"]) == pytest.approx(CODE_INTERPRETER_SESSION_COST), dict(
            raw.headers
        )
        requests: Final = wire.drain()
        assert [(request.method, request.target, request.headers["authorization"]) for request in requests] == [
            ("POST", "/v1/containers", f"Bearer {PROVIDER_KEY}")
        ]
        assert json.loads(requests[0].body) == {
            "name": "c",
            "expires_after": {"anchor": "last_active_at", "minutes": 5},
            "file_ids": ["file-spend"],
        }, requests[0].body

        logged: Final = eventually(
            lambda: read_rows(
                'SELECT call_type, spend, model_group FROM "LiteLLM_SpendLogs" WHERE api_key=%s', (digest,)
            ),
            lambda rows: len(rows) == 1,
            seconds=70,
        )
        assert [(row["call_type"], row["model_group"]) for row in logged] == [("acreate_container", model)], logged
        assert float(str(logged[0]["spend"])) == pytest.approx(CODE_INTERPRETER_SESSION_COST), logged
        eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_VerificationToken" WHERE token=%s', (digest,)),
            lambda rows: float(str(rows[0]["spend"])) == pytest.approx(CODE_INTERPRETER_SESSION_COST),
            seconds=70,
        )
