import json
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from typing import Final

import pytest

from tests.integration._support.client import Gateway, eventually, object_value, string_value
from tests.integration._support.database import read_rows, write_rows
from tests.integration._support.process import owned_proxy
from tests.integration.pricing.test_off_peak_pricing import off_peak_window


def delete_lens(lens_id: str) -> None:
    write_rows('DELETE FROM "LiteLLM_LensRun" WHERE lens_id=%s', (lens_id,))
    write_rows('DELETE FROM "LiteLLM_Lens" WHERE id=%s', (lens_id,))
    assert read_rows('SELECT id FROM "LiteLLM_Lens" WHERE id=%s', (lens_id,)) == []


@pytest.mark.parametrize("off_peak", (False, True))
def test_lens_bills_selected_key_and_rechecks_its_permissions(gateway: Gateway, off_peak: bool) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(
            input_cost_per_token=0.000001,
            output_cost_per_token=0.000002,
            model_info={
                "off_peak_pricing": {
                    **off_peak_window(-1, 1),
                    "input_cost_per_token": 0.0000005,
                    "output_cost_per_token": 0.000001,
                }
            }
            if off_peak
            else None,
        )
        key: Final = scenario.key(models=[model], max_budget=1)
        key_id: Final = sha256(key.encode()).hexdigest()
        worker: Final = gateway.post(
            "/lens/workers/register", {"name": "Billing regression", "analysis_key_id": key_id}
        )
        worker_id: Final = string_value(object_value(worker["worker"])["id"])
        scenario.cleanups.callback(write_rows, 'DELETE FROM "LiteLLM_LensWorker" WHERE id=%s', (worker_id,))
        lens: Final = gateway.post(
            "/lens",
            {
                "name": "Billing regression",
                "model": model,
                "enabled": False,
                "context": "Answers should be accurate",
                "source": "requests",
            },
        )
        lens_id: Final = string_value(lens["id"])
        scenario.cleanups.callback(delete_lens, lens_id)
        worker_key: Final = string_value(worker["token"])
        unauthorized: Final = gateway.request(
            "POST", "/lens/workers/register", {"name": "Denied", "analysis_key_id": key_id}, key=key
        )
        assert unauthorized.status_code == 403, unauthorized.text
        with ThreadPoolExecutor(max_workers=8) as pool:
            claims: Final = tuple(
                pool.map(
                    lambda _: gateway.request("POST", "/lens/worker/claim?protocol_version=2", {}, key=worker_key),
                    range(8),
                )
            )
        assert all(response.status_code == 200 for response in claims)
        winners: Final = tuple(response.json() for response in claims if response.json() is not None)
        assert len(winners) == 1
        claim: Final = object_value(winners[0])
        assert claim["lens_id"] == lens_id
        job_id: Final = string_value(object_value(claim["job"])["id"])
        path: Final = f"/lens/worker/{lens_id}/{job_id}/model"
        result: Final = gateway.post(path, {"prompt": "Inspect this run", "purpose": "extract"}, key=worker_key)
        expected: Final = (20 * 0.000001 + 20 * 0.000002) * (0.5 if off_peak else 1)
        assert result["cost"] == pytest.approx(expected)
        rows: Final = eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_VerificationToken" WHERE token=%s', (key_id,)),
            lambda values: len(values) == 1 and values[0]["spend"] == pytest.approx(expected),
            seconds=70,
        )
        assert rows[0]["spend"] == pytest.approx(expected)
        assert gateway.get(f"/lens/{lens_id}")["spent"] == pytest.approx(expected)
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
        assert gateway.get(f"/lens/{lens_id}")["spent"] == pytest.approx(expected)
        replacement: Final = scenario.key(models=[model], rpm_limit=1)
        replacement_id: Final = sha256(replacement.encode()).hexdigest()
        changed: Final = gateway.request(
            "PUT", f"/lens/workers/{worker_id}/billing-key", {"analysis_key_id": replacement_id}
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
        revoked: Final = gateway.request("DELETE", f"/lens/workers/{worker_id}")
        assert revoked.status_code == 200, revoked.text
        denied_worker: Final = gateway.request(
            "POST", path, {"prompt": "Must not run", "purpose": "extract"}, key=worker_key
        )
        assert denied_worker.status_code == 401, denied_worker.text
        forbidden_change: Final = gateway.request(
            "PUT", f"/lens/workers/{worker_id}/billing-key", {"analysis_key_id": replacement_id}
        )
        assert forbidden_change.status_code == 409, forbidden_change.text
        gateway.post(f"/lens/{lens_id}/cancel", {})


@pytest.mark.parametrize("cancel_on_disconnect", (False, True))
def test_worker_spend_logs_do_not_expose_investigation_content(
    gateway: Gateway, tmp_path: Path, cancel_on_disconnect: bool
) -> None:
    config: Final = tmp_path / "lens-privacy.json"
    config.write_text(
        json.dumps(
            {
                "model_list": [],
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    "store_model_in_db": True,
                    "store_prompts_in_spend_logs": True,
                    "cancel_on_disconnect": cancel_on_disconnect,
                    "proxy_batch_write_at": 1,
                    "proxy_batch_polling_interval": 1,
                    "allowed_ips": ["127.0.0.1"],
                },
            }
        )
    )
    with owned_proxy(gateway, tmp_path, {}, config=config) as isolated, isolated.scenario() as scenario:
        model: Final = scenario.model(input_cost_per_token=0.000001, output_cost_per_token=0.000002)
        key: Final = scenario.key(models=[model])
        key_id: Final = sha256(key.encode()).hexdigest()
        marker: Final = "PRIVATE_OTHER_TEAM_TRACE_CONTENT"
        ordinary: Final = isolated.chat(model, key=key, text=marker)
        retained: Final = eventually(
            lambda: read_rows(
                'SELECT proxy_server_request FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                (string_value(ordinary["id"]),),
            ),
            lambda rows: len(rows) == 1,
            seconds=70,
        )
        assert marker in str(retained[0]), "Control must prove this proxy retains ordinary prompts"
        worker: Final = isolated.post("/lens/workers/register", {"analysis_key_id": key_id})
        worker_id: Final = string_value(object_value(worker["worker"])["id"])
        scenario.cleanups.callback(write_rows, 'DELETE FROM "LiteLLM_LensWorker" WHERE id=%s', (worker_id,))
        lens: Final = isolated.post(
            "/lens", {"name": "Log privacy", "model": model, "enabled": False, "context": "Find problems"}
        )
        lens_id: Final = string_value(lens["id"])
        scenario.cleanups.callback(delete_lens, lens_id)
        worker_token: Final = string_value(worker["token"])
        claim: Final = isolated.post("/lens/worker/claim?protocol_version=2", {}, key=worker_token)
        job_id: Final = string_value(object_value(claim["job"])["id"])
        result: Final = isolated.post(
            f"/lens/worker/{lens_id}/{job_id}/model", {"prompt": marker, "purpose": "extract"}, key=worker_token
        )
        assert result["content"], "The worker must still receive model output"
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT spend, proxy_server_request, response FROM "LiteLLM_SpendLogs" WHERE api_key=%s AND request_id<>%s',
                (key_id, string_value(ordinary["id"])),
            ),
            lambda rows: len(rows) == 1,
            seconds=70,
        )
        assert float(rows[0]["spend"]) == pytest.approx(result["cost"])
        assert marker not in str(rows[0])
        assert result["content"] not in str(rows[0]["response"])
        isolated.post(f"/lens/{lens_id}/cancel", {})
