"""Contract: file-backed Gemini batch jobs on `/v1/batches` (#51032, landed 2026-11-11).

A caller creates a batch on a `gemini/` deployment through the OpenAI Batch API. The proxy posts
`{api_base}/v1beta/models/<model>:batchGenerateContent` with the deployment key in `x-goog-api-key` and the
body `{"batch": {"display_name": ..., "input_config": {"file_name": "files/<id>"}}}`; the uploaded JSONL runs
unchanged because Gemini accepts OpenAI-shaped rows. Gemini answers with a `batches/<id>` operation, which the
proxy maps onto an OpenAI batch: the `batches/` and `files/` prefixes are stripped, `BATCH_STATE_*` maps to
an OpenAI status, `batchStats` maps to `request_counts`, `endTime` lands on the timestamp of the terminal
status only, and an operation `error` becomes `errors.data[0]`. Retrieval is `GET {api_base}/v1beta/batches/<id>`.
Cancel is not implemented for Gemini. When a retrieved batch is completed the proxy downloads the output file
from `{api_base}/download/v1beta/files/<id>:download?alt=media` and bills it: Gemini writes `usage` with
`promptTokens`/`completionTokens`/`totalTokens`, and the tokens in `totalTokens` beyond the other two are
thinking tokens, billed as output.

Read from `litellm/llms/gemini/batches/transformation.py`, `litellm/llms/gemini/files/transformation.py:396`
and `litellm/batches/batch_utils.py`. Assumed: a `model` in the create body routes a raw provider file id
to that deployment and the proxy returns model-encoded ids (as on the OpenAI path in
`test_files_batches_wire.py`), so retrieve and cancel reach the same deployment without naming it again.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway, Scenario
from integration._support.wire import Wire, wire_server
from integration.providers.dec1_batchfiles_support import (
    CREATED_UNIX,
    ENDED_ISO,
    ENDED_UNIX,
    GEMINI_KEY,
    Script,
    batch_spend_rows,
    body_of,
    gemini_operation,
    json_of,
    jsonl,
    metadata_of,
    model_encoded_id,
    query_of,
    reply,
    sent,
    text_reply,
    unique,
)
from pydantic import JsonValue

_BACKEND: Final = "gemini-2.5-flash"
_CREATE_PATH: Final = f"/v1beta/models/{_BACKEND}:batchGenerateContent"


def _deployment(scenario: Scenario, wire: Wire, **pricing: JsonValue) -> str:
    return scenario.model(model=f"gemini/{_BACKEND}", api_key=GEMINI_KEY, api_base=wire.url, **pricing)


def _create(gateway: Gateway, model: str, file_id: str, **fields: JsonValue) -> dict[str, JsonValue]:
    response: Final = gateway.request(
        "POST",
        "/v1/batches",
        {
            "input_file_id": file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
            "model": model,
            **fields,
        },
    )
    assert response.status_code == 200, response.text
    return json_of(response)


def test_create_posts_batch_generate_content_with_the_deployment_key_header(gateway: Gateway) -> None:
    """A gemini deployment exists; the caller creates a batch from `files/<id>` with `model` in the body.

    The upstream must receive exactly one `POST /v1beta/models/gemini-2.5-flash:batchGenerateContent` that
    carries `x-goog-api-key: <deployment key>` and no `key` query parameter. A rewrite that moves the key to
    the query string would leak it into access logs.
    """
    batch_id: Final = unique("gembatch")
    script: Final = Script()
    script.on("POST", _CREATE_PATH, reply(gemini_operation(batch_id, "BATCH_STATE_PENDING")))
    with wire_server(script) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        _create(gateway, model, "files/dec1-input")
        posts: Final = sent(wire.drain(), "POST", _CREATE_PATH)
        assert len(posts) == 1, posts
        assert posts[0].headers.get("x-goog-api-key") == GEMINI_KEY
        assert "key" not in query_of(posts[0])


def test_create_body_names_the_input_file_and_a_default_display_name(gateway: Gateway) -> None:
    """The caller creates a batch from the bare id `dec1-input` and sends no metadata.

    The upstream body must be exactly `{"batch": {"display_name": "litellm-dec1-input", "input_config":
    {"file_name": "files/dec1-input"}}}`: the bare id gains the `files/` prefix and the default display name is
    `litellm-` plus the bare id.
    """
    script: Final = Script()
    script.on("POST", _CREATE_PATH, reply(gemini_operation(unique("gembatch"), "BATCH_STATE_PENDING")))
    with wire_server(script) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        _create(gateway, model, "dec1-input")
        (post,) = sent(wire.drain(), "POST", _CREATE_PATH)
        assert body_of(post) == {
            "batch": {"display_name": "litellm-dec1-input", "input_config": {"file_name": "files/dec1-input"}}
        }


def test_create_forwards_a_display_name_given_in_metadata(gateway: Gateway) -> None:
    """The caller sends `metadata: {"display_name": "nightly-eval"}`.

    The upstream body must carry `batch.display_name == "nightly-eval"`; display_name is the only metadata
    key Gemini batch jobs accept.
    """
    script: Final = Script()
    script.on("POST", _CREATE_PATH, reply(gemini_operation(unique("gembatch"), "BATCH_STATE_PENDING")))
    with wire_server(script) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        _create(gateway, model, "files/dec1-input", metadata={"display_name": "nightly-eval"})
        (post,) = sent(wire.drain(), "POST", _CREATE_PATH)
        batch: Final = body_of(post)["batch"]
        assert isinstance(batch, dict) and batch["display_name"] == "nightly-eval"


def test_create_with_metadata_other_than_display_name_is_refused_before_gemini(gateway: Gateway) -> None:
    """The caller sends `metadata: {"team": "blue"}`.

    The create must fail (the transformation raises `Gemini batch jobs carry no metadata besides
    display_name, got ['team']`) and nothing may reach `batchGenerateContent`. UNVERIFIED: the exact status
    (400 or 500) depends on how the proxy maps a `ValueError` from the batch handler.
    """
    script: Final = Script()
    script.on("POST", _CREATE_PATH, reply(gemini_operation(unique("gembatch"), "BATCH_STATE_PENDING")))
    with wire_server(script) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        response: Final = gateway.request(
            "POST",
            "/v1/batches",
            {
                "input_file_id": "files/dec1-input",
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
                "model": model,
                "metadata": {"team": "blue"},
            },
        )
        assert response.status_code in (400, 500), response.text
        assert "display_name" in response.text
        assert sent(wire.drain(), "POST", _CREATE_PATH) == []


def test_create_returns_an_openai_batch_with_model_encoded_ids(gateway: Gateway) -> None:
    """Gemini answers `batches/<id>` in `BATCH_STATE_PENDING` with input file `files/dec1-input`.

    The caller must get `object: "batch"`, `status: "validating"`, `endpoint: "/v1/chat/completions"`,
    `completion_window: "24h"`, `created_at` equal to the operation's `createTime` in unix seconds, and ids
    encoded with the model: `batch_` + base64url(`litellm:<id>;model,<model>`) and `file-` +
    base64url(`litellm:dec1-input;model,<model>`) (the `batches/` and `files/` prefixes stripped).
    """
    batch_id: Final = unique("gembatch")
    script: Final = Script()
    script.on("POST", _CREATE_PATH, reply(gemini_operation(batch_id, "BATCH_STATE_PENDING")))
    with wire_server(script) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        body: Final = _create(gateway, model, "files/dec1-input")
        assert (body["object"], body["status"], body["endpoint"], body["completion_window"]) == (
            "batch",
            "validating",
            "/v1/chat/completions",
            "24h",
        )
        assert body["created_at"] == CREATED_UNIX
        assert body["id"] == model_encoded_id(batch_id, model, "batch_")
        assert body["input_file_id"] == model_encoded_id("dec1-input", model, "file-")


@pytest.mark.parametrize(
    ("state", "status"),
    [
        ("BATCH_STATE_UNSPECIFIED", "validating"),
        ("BATCH_STATE_PENDING", "validating"),
        ("BATCH_STATE_RUNNING", "in_progress"),
        ("BATCH_STATE_FAILED", "failed"),
        ("BATCH_STATE_CANCELLED", "cancelled"),
        ("BATCH_STATE_EXPIRED", "expired"),
    ],
)
def test_retrieve_maps_each_gemini_state_to_an_openai_status(gateway: Gateway, state: str, status: str) -> None:
    """A batch exists; Gemini answers `GET /v1beta/batches/<id>` with the given `BATCH_STATE_*`.

    The caller's `GET /v1/batches/<encoded id>` must return the mapped OpenAI status (`_STATUS_MAP` in the
    Gemini batch transformation). `BATCH_STATE_SUCCEEDED` is covered by the billing test because a completed
    batch also triggers the output download.
    """
    batch_id: Final = unique("gembatch")
    script: Final = Script()
    script.on("POST", _CREATE_PATH, reply(gemini_operation(batch_id, "BATCH_STATE_PENDING")))
    script.on("GET", f"/v1beta/batches/{batch_id}", reply(gemini_operation(batch_id, state, end_time=ENDED_ISO)))
    with wire_server(script) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        created: Final = _create(gateway, model, "files/dec1-input")
        retrieved: Final = gateway.request("GET", f"/v1/batches/{created['id']}")
        assert retrieved.status_code == 200, retrieved.text
        assert json_of(retrieved)["status"] == status


def test_retrieve_is_a_get_on_the_bare_batch_resource_with_the_key_header(gateway: Gateway) -> None:
    """The caller retrieves a Gemini batch by the encoded id the create returned.

    The upstream must receive `GET /v1beta/batches/<bare id>` (no `models/` segment, no `batches/batches/`)
    with `x-goog-api-key: <deployment key>`, and the caller gets back the same encoded batch id.
    """
    batch_id: Final = unique("gembatch")
    script: Final = Script()
    script.on("POST", _CREATE_PATH, reply(gemini_operation(batch_id, "BATCH_STATE_PENDING")))
    script.on("GET", f"/v1beta/batches/{batch_id}", reply(gemini_operation(batch_id, "BATCH_STATE_RUNNING")))
    with wire_server(script) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        created: Final = _create(gateway, model, "files/dec1-input")
        wire.drain()
        retrieved: Final = gateway.request("GET", f"/v1/batches/{created['id']}")
        assert retrieved.status_code == 200, retrieved.text
        assert json_of(retrieved)["id"] == created["id"]
        gets: Final = sent(wire.drain(), "GET", f"/v1beta/batches/{batch_id}")
        assert len(gets) == 1 and gets[0].headers.get("x-goog-api-key") == GEMINI_KEY, gets


def test_retrieve_maps_batch_stats_to_request_counts(gateway: Gateway) -> None:
    """Gemini reports `batchStats: {requestCount: 5, successfulRequestCount: 3, failedRequestCount: 1}` while
    running.

    The caller must see `request_counts == {"total": 5, "completed": 3, "failed": 1}`; missing stats default
    to zero rather than null.
    """
    batch_id: Final = unique("gembatch")
    stats: Final = {"requestCount": 5, "successfulRequestCount": 3, "failedRequestCount": 1}
    script: Final = Script()
    script.on("POST", _CREATE_PATH, reply(gemini_operation(batch_id, "BATCH_STATE_PENDING")))
    script.on(
        "GET", f"/v1beta/batches/{batch_id}", reply(gemini_operation(batch_id, "BATCH_STATE_RUNNING", stats=stats))
    )
    with wire_server(script) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        created: Final = _create(gateway, model, "files/dec1-input")
        body: Final = json_of(gateway.request("GET", f"/v1/batches/{created['id']}"))
        assert body["request_counts"] == {"total": 5, "completed": 3, "failed": 1}


def test_end_time_lands_only_on_the_terminal_status_timestamp(gateway: Gateway) -> None:
    """Gemini reports `BATCH_STATE_FAILED` with `endTime`.

    The caller must see `failed_at` equal to `endTime` in unix seconds and `completed_at`, `expired_at` and
    `cancelled_at` all null. A port that copies `endTime` onto `completed_at` would make a failed batch look
    finished to clients that poll `completed_at`.
    """
    batch_id: Final = unique("gembatch")
    script: Final = Script()
    script.on("POST", _CREATE_PATH, reply(gemini_operation(batch_id, "BATCH_STATE_PENDING")))
    script.on(
        "GET",
        f"/v1beta/batches/{batch_id}",
        reply(gemini_operation(batch_id, "BATCH_STATE_FAILED", end_time=ENDED_ISO)),
    )
    with wire_server(script) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        created: Final = _create(gateway, model, "files/dec1-input")
        body: Final = json_of(gateway.request("GET", f"/v1/batches/{created['id']}"))
        assert (body["failed_at"], body["completed_at"], body["expired_at"], body["cancelled_at"]) == (
            ENDED_UNIX,
            None,
            None,
            None,
        )


def test_an_operation_error_becomes_the_batch_errors_list(gateway: Gateway) -> None:
    """Gemini reports a failed batch with operation `error: {code: 3, message: "bad input file"}`.

    The caller must see `errors == {"object": "list", "data": [{"code": "3", "message": "bad input file", ...}]}`
    (the numeric code as a string).
    """
    batch_id: Final = unique("gembatch")
    script: Final = Script()
    script.on("POST", _CREATE_PATH, reply(gemini_operation(batch_id, "BATCH_STATE_PENDING")))
    script.on(
        "GET",
        f"/v1beta/batches/{batch_id}",
        reply(gemini_operation(batch_id, "BATCH_STATE_FAILED", error={"code": 3, "message": "bad input file"})),
    )
    with wire_server(script) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        created: Final = _create(gateway, model, "files/dec1-input")
        errors: Final = json_of(gateway.request("GET", f"/v1/batches/{created['id']}"))["errors"]
        assert isinstance(errors, dict) and errors["object"] == "list", errors
        data: Final = errors["data"]
        assert isinstance(data, list) and len(data) == 1
        first: Final = data[0]
        assert isinstance(first, dict) and (first["code"], first["message"]) == ("3", "bad input file")


def test_cancel_on_a_gemini_batch_is_a_400_that_names_the_supported_providers(gateway: Gateway) -> None:
    """The caller cancels a Gemini batch by its encoded id.

    Cancel has no Gemini implementation: the answer must be 400 with the message `LiteLLM doesn't support
    gemini for 'cancel_batch'. Only 'openai', 'azure', 'vertex_ai', and 'bedrock' are supported.`, and no
    request may reach Gemini. Read from `litellm/batches/main.py` (cancel_batch else branch).
    """
    batch_id: Final = unique("gembatch")
    script: Final = Script()
    script.on("POST", _CREATE_PATH, reply(gemini_operation(batch_id, "BATCH_STATE_PENDING")))
    with wire_server(script) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        created: Final = _create(gateway, model, "files/dec1-input")
        wire.drain()
        cancelled: Final = gateway.request("POST", f"/v1/batches/{created['id']}/cancel")
        assert cancelled.status_code == 400, cancelled.text
        assert "LiteLLM doesn't support gemini for 'cancel_batch'" in cancelled.text
        assert [request for request in wire.drain() if request.method == "POST"] == []


def _gemini_output_row(custom_id: str, prompt: int, completion: int, total: int) -> dict[str, JsonValue]:
    return {
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "body": {
                "id": f"gen-{custom_id}",
                "object": "chat.completion",
                "model": _BACKEND,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"promptTokens": prompt, "completionTokens": completion, "totalTokens": total},
            },
        },
    }


def test_completed_batch_downloads_the_output_file_and_bills_thinking_tokens_as_output(gateway: Gateway) -> None:
    """A gemini deployment priced at `input_cost_per_token_batches: 0.001` and `output_cost_per_token_batches:
    0.002`; Gemini reports `BATCH_STATE_SUCCEEDED` with `responsesFile: files/<out>`, and the output file has
    two rows with usage `{promptTokens: 10, completionTokens: 4, totalTokens: 20}` and `{5, 3, 8}`.

    The retrieve must answer `completed` with `output_file_id` encoding the bare output id, the proxy must
    download `GET /download/v1beta/files/<out>:download?alt=media`, and the key gets one `aretrieve_batch`
    spend row with prompt_tokens 15 and completion_tokens 13 (the 6 thinking tokens of the first row are
    output) and spend `15 * 0.001 + 13 * 0.002 = 0.041`.
    """
    batch_id: Final = unique("gembatch")
    output_id: Final = unique("gemout")
    output: Final = jsonl(_gemini_output_row("r1", 10, 4, 20), _gemini_output_row("r2", 5, 3, 8)).decode()
    script: Final = Script()
    script.on("POST", _CREATE_PATH, reply(gemini_operation(batch_id, "BATCH_STATE_PENDING")))
    script.on(
        "GET",
        f"/v1beta/batches/{batch_id}",
        reply(
            gemini_operation(
                batch_id,
                "BATCH_STATE_SUCCEEDED",
                responses_file=f"files/{output_id}",
                end_time=ENDED_ISO,
                stats={"requestCount": 2, "successfulRequestCount": 2},
            )
        ),
    )
    script.on("GET", f"/download/v1beta/files/{output_id}:download", text_reply(output))
    with wire_server(script) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(
            scenario, wire, input_cost_per_token_batches=0.001, output_cost_per_token_batches=0.002
        )
        key: Final = scenario.key(models=[model])
        created_response: Final = gateway.request(
            "POST",
            "/v1/batches",
            {
                "input_file_id": "files/dec1-input",
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
                "model": model,
            },
            key=key,
        )
        assert created_response.status_code == 200, created_response.text
        retrieved: Final = gateway.request("GET", f"/v1/batches/{json_of(created_response)['id']}", key=key)
        assert retrieved.status_code == 200, retrieved.text
        body: Final = json_of(retrieved)
        assert (body["status"], body["completed_at"]) == ("completed", ENDED_UNIX)
        assert body["output_file_id"] == model_encoded_id(output_id, model, "file-")
        rows: Final = batch_spend_rows(gateway, key)
        downloads: Final = sent(wire.drain(), "GET", f"/download/v1beta/files/{output_id}:download")
        assert downloads and query_of(downloads[0]).get("alt") == ["media"], downloads
        assert len(rows) == 1, rows
        assert (rows[0]["prompt_tokens"], rows[0]["completion_tokens"]) == (15, 13)
        spend: Final = rows[0]["spend"]
        assert isinstance(spend, (int, float, str)) and float(spend) == pytest.approx(0.041)
        assert metadata_of(rows[0])["batch_successful_requests"] == 2
