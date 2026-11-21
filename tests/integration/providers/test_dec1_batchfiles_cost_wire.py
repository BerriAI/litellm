"""Contract: what a completed OpenAI or Azure batch costs, and where that cost lands.

When a retrieve sees `status: completed`, the proxy downloads `output_file_id`, prices every succeeded line at
batch rates and writes ONE spend row with `call_type = 'aretrieve_batch'` for the creating key, with the summed
prompt and completion tokens. Batch rates come from `input_cost_per_token_batches` /
`output_cost_per_token_batches` when the deployment (or the cost map) has them and are half the standard
per-token prices otherwise (`litellm/cost_calculator.py:3306` `batch_cost_calculator`, `_batch_or_half`).
The cost is added to the key's `spend` like any other request.

Existing coverage this file does not repeat: the half-price fallback and cached-input cases
(`cost_calculation/cases/batch_realtime.py`), reasoning tokens and error-file failure counts
(`spend/test_batch_completion_accounting.py`). Assumed: a model-routed batch (`model` form field and body
field) is billed on the retrieve that first sees it completed, as in `test_batch_completion_accounting.py`.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway, Scenario, eventually
from integration._support.wire import Wire, wire_server
from integration.providers.dec1_batchfiles_support import (
    Script,
    batch_spend_rows,
    chat_row,
    hashed,
    json_of,
    jsonl,
    metadata_of,
    openai_batch,
    openai_file,
    openai_output_line,
    reply,
    text_reply,
    unique,
)
from pydantic import JsonValue

_AZURE_VERSION: Final = "2024-10-21"


def _openai_script(stem: str, output: bytes, prefix: str = "/v1") -> Script:
    batch_id: Final = f"batch_{stem}"
    completed: Final = openai_batch(
        batch_id,
        "completed",
        output_file_id=f"file-out-{stem}",
        completed_at=1763377200,
        request_counts={"total": 3, "completed": 2, "failed": 1},
    )
    script: Final = Script()
    script.on("POST", f"{prefix}/files", reply(openai_file(f"file-in-{stem}")))
    script.on("POST", f"{prefix}/batches", reply(openai_batch(batch_id, "validating")))
    script.on("GET", f"{prefix}/batches/{batch_id}", reply(completed))
    script.on("GET", f"{prefix}/files/file-out-{stem}/content", text_reply(output.decode()))
    return script


def _run_batch(gateway: Gateway, key: str, model: str) -> dict[str, JsonValue]:
    uploaded: Final = gateway.request_multipart(
        "/v1/files",
        {"purpose": "batch", "model": model},
        {
            "file": (
                "in.jsonl",
                jsonl(chat_row("r1", model), chat_row("r2", model), chat_row("r3", model)),
                "application/jsonl",
            )
        },
        key=key,
    )
    assert uploaded.status_code == 200, uploaded.text
    created: Final = gateway.request(
        "POST",
        "/v1/batches",
        {
            "input_file_id": json_of(uploaded)["id"],
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
            "model": model,
        },
        key=key,
    )
    assert created.status_code == 200, created.text
    retrieved: Final = gateway.request("GET", f"/v1/batches/{json_of(created)['id']}", key=key)
    assert retrieved.status_code == 200, retrieved.text
    return json_of(retrieved)


def _output(model: str) -> bytes:
    return jsonl(
        openai_output_line("r1", model, 100, 20),
        openai_output_line("r2", model, 50, 10),
        openai_output_line("r3", model, 0, 0, status=400),
    )


def _openai_model(scenario: Scenario, wire: Wire, **pricing: JsonValue) -> str:
    return scenario.model(
        model="openai/gpt-4o-mini", api_key="sk-dec1-batchfiles", api_base=f"{wire.url}/v1", **pricing
    )


def test_deployment_batch_prices_win_over_half_of_the_standard_prices(gateway: Gateway) -> None:
    """A deployment sets `input_cost_per_token: 0.01`, `output_cost_per_token: 0.02`,
    `input_cost_per_token_batches: 0.001` and `output_cost_per_token_batches: 0.003`; its batch completes with
    two succeeded lines (100+20 and 50+10 tokens) and one 400 line.

    The key's single `aretrieve_batch` row must have prompt_tokens 150, completion_tokens 30 and spend
    `150 * 0.001 + 30 * 0.003 = 0.24`, not the half-price `150 * 0.005 + 30 * 0.01 = 1.05`; the 400 line
    adds no tokens and no cost.
    """
    stem: Final = unique("dep-batch-price")
    with gateway.scenario() as scenario:
        script: Final = Script()
        with wire_server(script) as wire:
            model: Final = _openai_model(
                scenario,
                wire,
                input_cost_per_token=0.01,
                output_cost_per_token=0.02,
                input_cost_per_token_batches=0.001,
                output_cost_per_token_batches=0.003,
            )
            for route, answer in _openai_script(stem, _output("gpt-4o-mini")).routes.items():
                script.on(route[0], route[1], answer)
            key: Final = scenario.key(models=[model])
            assert _run_batch(gateway, key, model)["status"] == "completed"
            rows: Final = batch_spend_rows(gateway, key)
            assert len(rows) == 1, rows
            assert (rows[0]["prompt_tokens"], rows[0]["completion_tokens"]) == (150, 30)
            spend: Final = rows[0]["spend"]
            assert isinstance(spend, (int, float, str)) and float(spend) == pytest.approx(0.24)
            metadata: Final = metadata_of(rows[0])
            assert (metadata["batch_successful_requests"], metadata["batch_failed_requests"]) == (2, 1)


def test_completed_batch_cost_is_added_to_the_key_spend(gateway: Gateway) -> None:
    """A deployment priced at `input_cost_per_token_batches: 0.001`, `output_cost_per_token_batches: 0.002`
    completes a batch costing `150 * 0.001 + 30 * 0.002 = 0.21`.

    `LiteLLM_VerificationToken.spend` for the key must reach 0.21 (spend lands asynchronously, so it is
    polled). A rewrite that writes the spend row but skips the key increment would let batch callers run
    past `max_budget`.
    """
    stem: Final = unique("key-spend")
    with gateway.scenario() as scenario:
        script: Final = Script()
        with wire_server(script) as wire:
            model: Final = _openai_model(
                scenario, wire, input_cost_per_token_batches=0.001, output_cost_per_token_batches=0.002
            )
            for route, answer in _openai_script(stem, _output("gpt-4o-mini")).routes.items():
                script.on(route[0], route[1], answer)
            key: Final = scenario.key(models=[model])
            _run_batch(gateway, key, model)
            rows: Final = eventually(
                lambda: gateway.rows('SELECT spend FROM "LiteLLM_VerificationToken" WHERE token = %s', (hashed(key),)),
                lambda found: len(found) == 1 and float(str(found[0]["spend"])) >= 0.21 - 1e-9,
                seconds=70,
            )
            assert float(str(rows[0]["spend"])) == pytest.approx(0.21)


def test_retrieving_a_completed_batch_again_writes_no_second_spend_row(gateway: Gateway) -> None:
    """A completed batch is retrieved three times by its creating key.

    Exactly one `aretrieve_batch` spend row may exist for the key afterwards: the managed object row is
    claimed (`batch_processed = true`) by the first costing, so later retrieves bill nothing.
    UNVERIFIED for model-routed (not managed) batches; the cost case suite pins the two-retrieve case for
    its own setup.
    """
    stem: Final = unique("retrieve-thrice")
    with gateway.scenario() as scenario:
        script: Final = Script()
        with wire_server(script) as wire:
            model: Final = _openai_model(
                scenario, wire, input_cost_per_token_batches=0.001, output_cost_per_token_batches=0.002
            )
            for route, answer in _openai_script(stem, _output("gpt-4o-mini")).routes.items():
                script.on(route[0], route[1], answer)
            key: Final = scenario.key(models=[model])
            body: Final = _run_batch(gateway, key, model)
            for _ in range(2):
                again = gateway.request("GET", f"/v1/batches/{body['id']}", key=key)
                assert again.status_code == 200, again.text
            batch_spend_rows(gateway, key)
            settled: Final = eventually(
                lambda: gateway.rows(
                    'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE api_key = %s AND call_type = %s',
                    (hashed(key), "aretrieve_batch"),
                ),
                lambda found: len(found) >= 1,
                seconds=10,
            )
            assert len(settled) == 1, settled


def test_an_azure_batch_is_billed_from_its_output_file_on_the_azure_route(gateway: Gateway) -> None:
    """An Azure deployment (`azure/<deployment>`, `api_version` 2024-10-21) priced at
    `input_cost_per_token_batches: 0.001` and `output_cost_per_token_batches: 0.002` completes a batch.

    The proxy must download the output from `GET /openai/files/file-out-<stem>/content?api-version=2024-10-21`
    and write one `aretrieve_batch` row with 150 prompt tokens, 30 completion tokens and spend 0.21.
    """
    stem: Final = unique("azure-cost")
    with gateway.scenario() as scenario:
        script: Final = Script()
        with wire_server(script) as wire:
            model: Final = scenario.model(
                model="azure/dec1-batchfiles-deployment",
                api_key="dec1-batchfiles-azure-key",
                api_base=wire.url,
                api_version=_AZURE_VERSION,
                input_cost_per_token_batches=0.001,
                output_cost_per_token_batches=0.002,
            )
            for route, answer in _openai_script(stem, _output("gpt-4o-mini"), prefix="/openai").routes.items():
                script.on(route[0], route[1], answer)
            key: Final = scenario.key(models=[model])
            assert _run_batch(gateway, key, model)["status"] == "completed"
            rows: Final = batch_spend_rows(gateway, key)
            downloads: Final = [
                request
                for request in wire.drain()
                if request.method == "GET" and request.target.startswith(f"/openai/files/file-out-{stem}/content")
            ]
            assert downloads and f"api-version={_AZURE_VERSION}" in downloads[0].target, downloads
            assert (rows[0]["prompt_tokens"], rows[0]["completion_tokens"]) == (150, 30)
            spend: Final = rows[0]["spend"]
            assert isinstance(spend, (int, float, str)) and float(spend) == pytest.approx(0.21)
