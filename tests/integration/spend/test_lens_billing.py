from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from typing import Final

import pytest

from tests.integration._support.client import Gateway, eventually, object_value, string_value
from tests.integration._support.database import read_rows, write_rows


def delete_lens(engine_id: str) -> None:
    write_rows('DELETE FROM "LiteLLM_EngineRun" WHERE engine_id=%s', (engine_id,))
    write_rows('DELETE FROM "LiteLLM_Engine" WHERE id=%s', (engine_id,))
    assert read_rows('SELECT id FROM "LiteLLM_Engine" WHERE id=%s', (engine_id,)) == []


def test_lens_bills_selected_key_and_rechecks_its_permissions(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.000001, output_cost_per_token=0.000002)
        key: Final = scenario.key(models=[model], max_budget=1)
        key_id: Final = sha256(key.encode()).hexdigest()
        worker: Final = gateway.post(
            "/engine/workers/register", {"name": "Billing regression", "analysis_key_id": key_id}
        )
        worker_id: Final = string_value(object_value(worker["worker"])["id"])
        scenario.cleanups.callback(write_rows, 'DELETE FROM "LiteLLM_EngineWorker" WHERE id=%s', (worker_id,))
        engine: Final = gateway.post(
            "/engine",
            {
                "name": "Billing regression",
                "model": model,
                "enabled": False,
                "context": "Answers should be accurate",
                "source": "requests",
            },
        )
        engine_id: Final = string_value(engine["id"])
        scenario.cleanups.callback(delete_lens, engine_id)
        worker_key: Final = string_value(worker["token"])
        unauthorized: Final = gateway.request(
            "POST", "/engine/workers/register", {"name": "Denied", "analysis_key_id": key_id}, key=key
        )
        assert unauthorized.status_code == 403, unauthorized.text
        with ThreadPoolExecutor(max_workers=8) as pool:
            claims: Final = tuple(
                pool.map(
                    lambda _: gateway.request("POST", "/engine/worker/claim?protocol_version=2", {}, key=worker_key),
                    range(8),
                )
            )
        assert all(response.status_code == 200 for response in claims)
        winners: Final = tuple(response.json() for response in claims if response.json() is not None)
        assert len(winners) == 1
        claim: Final = object_value(winners[0])
        assert claim["engine_id"] == engine_id
        job_id: Final = string_value(object_value(claim["job"])["id"])
        path: Final = f"/engine/worker/{engine_id}/{job_id}/model"
        result: Final = gateway.post(path, {"prompt": "Inspect this run", "purpose": "extract"}, key=worker_key)
        expected: Final = 20 * 0.000001 + 20 * 0.000002
        assert result["cost"] == pytest.approx(expected)
        rows: Final = eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_VerificationToken" WHERE token=%s', (key_id,)),
            lambda values: len(values) == 1 and values[0]["spend"] == pytest.approx(expected),
            seconds=70,
        )
        assert rows[0]["spend"] == pytest.approx(expected)
        assert gateway.get(f"/engine/{engine_id}")["spent"] == pytest.approx(expected)
        raw_hash: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": "Not a bearer credential"}],
            },
            key=key_id,
        )
        assert raw_hash.status_code == 401, raw_hash.text
        gateway.post("/key/update", {"key": key, "max_budget": expected / 2})
        exhausted: Final = gateway.request(
            "POST", path, {"prompt": "Must not run", "purpose": "extract"}, key=worker_key
        )
        assert exhausted.status_code == 402, exhausted.text
        gateway.post("/key/update", {"key": key, "max_budget": 1, "models": ["unavailable-analysis-model"]})
        restricted: Final = gateway.request(
            "POST", path, {"prompt": "Must not run", "purpose": "extract"}, key=worker_key
        )
        assert restricted.status_code == 403, restricted.text
        gateway.post("/key/block", {"key": key})
        blocked: Final = gateway.request("POST", path, {"prompt": "Must not run", "purpose": "extract"}, key=worker_key)
        assert blocked.status_code == 400, blocked.text
        assert gateway.get(f"/engine/{engine_id}")["spent"] == pytest.approx(expected)
        replacement: Final = scenario.key(models=[model], rpm_limit=1)
        replacement_id: Final = sha256(replacement.encode()).hexdigest()
        changed: Final = gateway.request(
            "PUT", f"/engine/workers/{worker_id}/billing-key", {"analysis_key_id": replacement_id}
        )
        assert changed.status_code == 200, changed.text
        billed_replacement: Final = gateway.post(
            path, {"prompt": "Inspect another run", "purpose": "extract"}, key=worker_key
        )
        assert billed_replacement["cost"] == pytest.approx(expected)
        limited: Final = gateway.request("POST", path, {"prompt": "Must not run", "purpose": "extract"}, key=worker_key)
        assert limited.status_code == 429, limited.text
        second_rows: Final = eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_VerificationToken" WHERE token=%s', (replacement_id,)),
            lambda values: len(values) == 1 and values[0]["spend"] == pytest.approx(expected),
            seconds=70,
        )
        assert second_rows[0]["spend"] == pytest.approx(expected)
        revoked: Final = gateway.request("DELETE", f"/engine/workers/{worker_id}")
        assert revoked.status_code == 200, revoked.text
        denied_worker: Final = gateway.request(
            "POST", path, {"prompt": "Must not run", "purpose": "extract"}, key=worker_key
        )
        assert denied_worker.status_code == 401, denied_worker.text
        forbidden_change: Final = gateway.request(
            "PUT", f"/engine/workers/{worker_id}/billing-key", {"analysis_key_id": replacement_id}
        )
        assert forbidden_change.status_code == 409, forbidden_change.text
        gateway.post(f"/engine/{engine_id}/cancel", {})
