"""Contract: Anthropic Message Batches seen through the OpenAI `/v1/batches` routes.

The proxy cannot create an Anthropic batch through `/v1/batches` (the create transformation raises
`Batch creation not yet implemented for Anthropic`), but it retrieves one: `GET /v1/batches/<id>` for an
`anthropic/` deployment becomes `GET {api_base}/v1/messages/batches/<id>` with `anthropic-version: 2023-06-01`
and an `anthropic-beta` header that always includes `message-batches-2024-09-24`, authenticated with the
deployment key. The Message Batch is mapped onto an OpenAI batch: `processing_status` `in_progress` stays
`in_progress`, `canceling` becomes `cancelling`, `ended` becomes `completed`, `endpoint` is `/v1/messages`,
`output_file_id` is the batch id itself (results are read from `/v1/messages/batches/<id>/results`),
`request_counts.total` is the sum of all five Anthropic counters, `completed` is `succeeded` and `failed` is
`errored`. Cancel is not supported for Anthropic through `/v1/batches`.

Read from `litellm/llms/anthropic/batches/transformation.py`, `litellm/batches/main.py` (retrieve
`anthropic` branch and the cancel else branch) and `litellm/batches/batch_utils.py:853`. Assumed: a batch id
encoded with the deployment's model (`batch_` + base64url(`litellm:<id>;model,<model>`)) routes the retrieve
to that deployment, as it does for OpenAI in `test_files_batches_wire.py`; the tests build that id
themselves because the proxy cannot create one.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway, Scenario
from integration._support.wire import Wire, wire_server
from integration.providers.dec1_batchfiles_support import (
    ANTHROPIC_KEY,
    CREATED_UNIX,
    ENDED_ISO,
    ENDED_UNIX,
    Script,
    anthropic_message_batch,
    batch_spend_rows,
    json_of,
    jsonl,
    model_encoded_id,
    reply,
    sent,
    text_reply,
    unique,
)
from pydantic import JsonValue

_BACKEND: Final = "claude-haiku-4-5-20251001"
_CANCEL_ISO: Final = "2026-11-17T10:30:00Z"


def _deployment(scenario: Scenario, wire: Wire, **pricing: JsonValue) -> str:
    return scenario.model(model=f"anthropic/{_BACKEND}", api_key=ANTHROPIC_KEY, api_base=wire.url, **pricing)


def _retrieve(gateway: Gateway, batch_id: str, model: str, *, key: str | None = None) -> dict[str, JsonValue]:
    response: Final = gateway.request("GET", f"/v1/batches/{model_encoded_id(batch_id, model, 'batch_')}", key=key)
    assert response.status_code == 200, response.text
    return json_of(response)


def test_retrieve_reads_the_message_batch_with_the_batches_beta_and_the_deployment_key(gateway: Gateway) -> None:
    """An anthropic deployment and a Message Batch `msgbatch_<x>` that is `in_progress` upstream.

    The proxy must send one `GET /v1/messages/batches/msgbatch_<x>` with `anthropic-version: 2023-06-01`,
    an `anthropic-beta` header containing `message-batches-2024-09-24`, and `x-api-key` equal to the
    deployment key.
    """
    batch_id: Final = unique("msgbatch")
    script: Final = Script()
    script.on("GET", f"/v1/messages/batches/{batch_id}", reply(anthropic_message_batch(batch_id, "in_progress")))
    with wire_server(script) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        _retrieve(gateway, batch_id, model)
        gets: Final = sent(wire.drain(), "GET", f"/v1/messages/batches/{batch_id}")
        assert len(gets) == 1, gets
        headers: Final = gets[0].headers
        assert headers.get("anthropic-version") == "2023-06-01"
        assert "message-batches-2024-09-24" in headers.get("anthropic-beta", "")
        assert headers.get("x-api-key") == ANTHROPIC_KEY


@pytest.mark.parametrize(
    ("processing_status", "status"),
    [("in_progress", "in_progress"), ("canceling", "cancelling"), ("ended", "completed")],
)
def test_retrieve_maps_processing_status_to_an_openai_status(
    gateway: Gateway, processing_status: str, status: str
) -> None:
    """Anthropic reports the given `processing_status` for a Message Batch with no succeeded requests.

    The caller must get the mapped OpenAI status. For `ended` the batch has zero succeeded requests, so the
    proxy has nothing to bill and still answers `completed`.
    """
    batch_id: Final = unique("msgbatch")
    script: Final = Script()
    script.on(
        "GET",
        f"/v1/messages/batches/{batch_id}",
        reply(
            anthropic_message_batch(batch_id, processing_status, ended_at=ENDED_ISO if status == "completed" else None)
        ),
    )
    script.on("GET", f"/v1/messages/batches/{batch_id}/results", text_reply(""))
    with wire_server(script) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        assert _retrieve(gateway, batch_id, model)["status"] == status


def test_retrieve_reports_the_messages_endpoint_and_the_batch_id_as_output_file(gateway: Gateway) -> None:
    """Anthropic reports an `in_progress` Message Batch.

    The OpenAI batch the caller gets must say `endpoint: "/v1/messages"` and `completion_window: "24h"`, and
    its `output_file_id` must point back at the same batch (the results file is read by batch id), encoded
    with the model like every other id on this path.
    """
    batch_id: Final = unique("msgbatch")
    script: Final = Script()
    script.on("GET", f"/v1/messages/batches/{batch_id}", reply(anthropic_message_batch(batch_id, "in_progress")))
    with wire_server(script) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        body: Final = _retrieve(gateway, batch_id, model)
        assert (body["endpoint"], body["completion_window"]) == ("/v1/messages", "24h")
        assert body["output_file_id"] == model_encoded_id(batch_id, model, "file-")


def test_retrieve_sums_all_five_anthropic_counters_into_total(gateway: Gateway) -> None:
    """Anthropic reports `request_counts: {processing: 1, succeeded: 2, errored: 3, canceled: 4, expired: 5}`.

    The caller must see `request_counts == {"total": 15, "completed": 2, "failed": 3}`: canceled and expired
    requests count toward the total but toward neither completed nor failed.
    """
    batch_id: Final = unique("msgbatch")
    counts: Final = {"processing": 1, "succeeded": 2, "errored": 3, "canceled": 4, "expired": 5}
    script: Final = Script()
    script.on(
        "GET",
        f"/v1/messages/batches/{batch_id}",
        reply(anthropic_message_batch(batch_id, "in_progress", counts=counts)),
    )
    with wire_server(script) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        assert _retrieve(gateway, batch_id, model)["request_counts"] == {"total": 15, "completed": 2, "failed": 3}


def test_retrieve_converts_iso_timestamps_to_unix_seconds(gateway: Gateway) -> None:
    """Anthropic reports an `in_progress` batch created at `2026-11-17T10:00:00Z`.

    The caller must see `created_at` and `in_progress_at` both equal to that instant in unix seconds and
    `completed_at` null. A port that passes the ISO strings through would break every OpenAI SDK client,
    which types these fields as integers.
    """
    batch_id: Final = unique("msgbatch")
    script: Final = Script()
    script.on("GET", f"/v1/messages/batches/{batch_id}", reply(anthropic_message_batch(batch_id, "in_progress")))
    with wire_server(script) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        body: Final = _retrieve(gateway, batch_id, model)
        assert (body["created_at"], body["in_progress_at"], body["completed_at"]) == (CREATED_UNIX, CREATED_UNIX, None)


def test_an_ended_batch_reports_ended_at_as_completed_at(gateway: Gateway) -> None:
    """Anthropic reports `processing_status: ended` with `ended_at: 2026-11-17T11:00:00Z` and nothing succeeded.

    The caller must see `status: completed` and `completed_at` equal to `ended_at` in unix seconds.
    """
    batch_id: Final = unique("msgbatch")
    script: Final = Script()
    script.on(
        "GET",
        f"/v1/messages/batches/{batch_id}",
        reply(anthropic_message_batch(batch_id, "ended", counts={"errored": 1}, ended_at=ENDED_ISO)),
    )
    script.on("GET", f"/v1/messages/batches/{batch_id}/results", text_reply(""))
    with wire_server(script) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        body: Final = _retrieve(gateway, batch_id, model)
        assert (body["status"], body["completed_at"]) == ("completed", ENDED_UNIX)


def test_a_batch_ended_by_cancel_is_reported_completed_not_cancelled(gateway: Gateway) -> None:
    """Anthropic reports `processing_status: ended` with `cancel_initiated_at` set and `canceled: 3`.

    Anthropic has no separate cancelled status, so the proxy reports `completed` and `cancelled_at` null
    (`cancelled_at` is only filled while `processing_status` is `canceling`). Pinned so a rewrite keeps the
    current mapping until it is changed on purpose.
    """
    batch_id: Final = unique("msgbatch")
    script: Final = Script()
    script.on(
        "GET",
        f"/v1/messages/batches/{batch_id}",
        reply(
            anthropic_message_batch(
                batch_id, "ended", counts={"canceled": 3}, ended_at=ENDED_ISO, cancel_initiated_at=_CANCEL_ISO
            )
        ),
    )
    script.on("GET", f"/v1/messages/batches/{batch_id}/results", text_reply(""))
    with wire_server(script) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        body: Final = _retrieve(gateway, batch_id, model)
        assert (body["status"], body["cancelled_at"]) == ("completed", None)


def test_an_anthropic_404_keeps_its_status(gateway: Gateway) -> None:
    """Anthropic answers the retrieve with 404 `not_found_error`.

    The caller must get 404, not 500, so clients can tell a missing batch from a provider outage.
    UNVERIFIED: relies on `AnthropicError` carrying the upstream status through the proxy's exception mapper.
    """
    batch_id: Final = unique("msgbatch")
    script: Final = Script()
    script.on(
        "GET",
        f"/v1/messages/batches/{batch_id}",
        reply({"type": "error", "error": {"type": "not_found_error", "message": "batch not found"}}, 404),
    )
    with wire_server(script) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        response: Final = gateway.request("GET", f"/v1/batches/{model_encoded_id(batch_id, model, 'batch_')}")
        assert response.status_code == 404, response.text


def test_cancel_on_an_anthropic_batch_is_a_400_and_never_reaches_anthropic(gateway: Gateway) -> None:
    """The caller cancels an Anthropic batch by its model-encoded id.

    The answer must be 400 with `LiteLLM doesn't support anthropic for 'cancel_batch'`, and no
    `POST /v1/messages/batches/<id>/cancel` may reach the upstream. Anthropic cancel is only reachable
    through the `/anthropic/v1/messages/batches/<id>/cancel` pass-through.
    """
    batch_id: Final = unique("msgbatch")
    script: Final = Script()
    with wire_server(script) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        response: Final = gateway.request("POST", f"/v1/batches/{model_encoded_id(batch_id, model, 'batch_')}/cancel")
        assert response.status_code == 400, response.text
        assert "LiteLLM doesn't support anthropic for 'cancel_batch'" in response.text
        assert sent(wire.drain(), "POST", f"/v1/messages/batches/{batch_id}/cancel") == []


def _result_line(custom_id: str, input_tokens: int, output_tokens: int) -> dict[str, JsonValue]:
    return {
        "custom_id": custom_id,
        "result": {
            "type": "succeeded",
            "message": {
                "id": f"msg_{custom_id}",
                "type": "message",
                "role": "assistant",
                "model": _BACKEND,
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            },
        },
    }


def test_an_ended_batch_is_billed_from_its_results_file(gateway: Gateway) -> None:
    """An anthropic deployment priced at `input_cost_per_token_batches: 0.001` and
    `output_cost_per_token_batches: 0.002`; the batch has ended and its results file has two succeeded lines
    with usage `{input_tokens: 10, output_tokens: 4}` and `{6, 2}`.

    A key retrieving the batch must see `completed`, the proxy must read
    `GET /v1/messages/batches/<id>/results`, and the key gets one `aretrieve_batch` spend row with 16 prompt
    tokens, 6 completion tokens and spend `16 * 0.001 + 6 * 0.002 = 0.028`. UNVERIFIED: that a model-routed
    (not managed) Anthropic batch is billed on retrieve the same way an OpenAI one is.
    """
    batch_id: Final = unique("msgbatch")
    results: Final = jsonl(_result_line("r1", 10, 4), _result_line("r2", 6, 2)).decode()
    script: Final = Script()
    script.on(
        "GET",
        f"/v1/messages/batches/{batch_id}",
        reply(anthropic_message_batch(batch_id, "ended", counts={"succeeded": 2}, ended_at=ENDED_ISO)),
    )
    script.on("GET", f"/v1/messages/batches/{batch_id}/results", text_reply(results))
    with wire_server(script) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(
            scenario, wire, input_cost_per_token_batches=0.001, output_cost_per_token_batches=0.002
        )
        key: Final = scenario.key(models=[model])
        assert _retrieve(gateway, batch_id, model, key=key)["status"] == "completed"
        rows: Final = batch_spend_rows(gateway, key)
        assert sent(wire.drain(), "GET", f"/v1/messages/batches/{batch_id}/results"), "results were never read"
        assert len(rows) == 1, rows
        assert (rows[0]["prompt_tokens"], rows[0]["completion_tokens"]) == (16, 6)
        spend: Final = rows[0]["spend"]
        assert isinstance(spend, (int, float, str)) and float(spend) == pytest.approx(0.028)
