import json
from collections.abc import Callable, Iterable
from typing import Final

import anthropic
import openai
import pytest
from integration._support.client import Gateway, Scenario
from integration._support.wire import Reply, wire_server
from pydantic import JsonValue

from integration.providers._vertex_batch_rows import (
    BOM,
    BODY_NOT_OBJECT,
    CHAT_URL,
    ROUTES,
    Route,
    chat_row,
    closed_port,
    encoded,
    error_of,
    gcs_peer,
    is_openai_upload,
    jsonl,
    openai_files_peer,
    openai_uploaded_lines,
    roleless_row,
    stored_custom_ids,
    unconvertible_line_message,
    upload,
    vertex_model,
)


def _with_body(body: JsonValue) -> bytes:
    return encoded({"custom_id": "bad", "method": "POST", "url": CHAT_URL, "body": body})


def _row_case(case: str, model: str) -> tuple[bytes, str]:
    valid_body: Final = json.dumps(chat_row("bad", model, "fine")["body"])
    match case:
        case "body_null":
            return _with_body(None), BODY_NOT_OBJECT
        case "body_int":
            return _with_body(7), BODY_NOT_OBJECT
        case "body_list":
            return _with_body([{"model": model}]), BODY_NOT_OBJECT
        case "body_empty_string":
            return _with_body(""), BODY_NOT_OBJECT
        case "body_5kb_string":
            return _with_body("x" * 5120), BODY_NOT_OBJECT
        case "body_twice_null_last":
            twice: Final = f'{{"custom_id": "bad", "method": "POST", "url": "{CHAT_URL}", "body": {valid_body}, "body": null}}'
            return twice.encode(), BODY_NOT_OBJECT
        case "message_without_role":
            return encoded(roleless_row("bad", model)), "missing field 'role'"
        case "responses_body_null":
            return encoded({"custom_id": "bad", "method": "POST", "url": "/v1/responses", "body": None}), BODY_NOT_OBJECT
        case _:
            raise AssertionError(case)


ROW_CASES: Final = (
    "body_null",
    "body_int",
    "body_list",
    "body_empty_string",
    "body_5kb_string",
    "body_twice_null_last",
    "message_without_role",
    "responses_body_null",
)


@pytest.mark.parametrize("case", ROW_CASES)
@pytest.mark.parametrize("route", ROUTES)
def test_row_vertex_cannot_convert_answers_400_naming_its_file_line_every_time(
    gateway: Gateway, route: Route, case: str
) -> None:
    with wire_server(gcs_peer) as wire, gateway.scenario() as scenario:
        model: Final = vertex_model(gateway, scenario, wire.url)
        bad_line, detail = _row_case(case, model)
        content: Final = jsonl(
            encoded(chat_row("first", model, "one")), b"", bad_line, encoded(chat_row("last", model, "two"))
        )
        responses: Final = tuple(upload(gateway, route, model, content) for _ in range(2))
        assert tuple(response.status_code for response in responses) == (400, 400), [r.text for r in responses]
        errors: Final = tuple(error_of(response) for response in responses)
        assert tuple(error["type"] for error in errors) == ("invalid_request_error",) * 2, errors
        assert str(errors[0]["message"]).startswith(unconvertible_line_message(3) + detail), errors[0]
        assert errors[0]["message"] == errors[1]["message"]
        assert wire.drain() == (), "a batch file with an unconvertible row reached the bucket"


def test_embeddings_row_vertex_cannot_convert_names_its_file_line(gateway: Gateway) -> None:
    block: Final[dict[str, JsonValue]] = {"type": "file", "file": {"file_id": "gs://scripted-bucket/clip.mp4"}}
    with wire_server(gcs_peer) as wire, gateway.scenario() as scenario:
        model: Final = vertex_model(gateway, scenario, wire.url, backend="gemini-embedding-2-preview")
        fine: Final = encoded(
            {"custom_id": "fine", "method": "POST", "url": "/v1/embeddings", "body": {"model": model, "input": "bus"}}
        )
        bare: Final = encoded(
            {"custom_id": "bare", "method": "POST", "url": "/v1/embeddings", "body": {"model": model, "input": block}}
        )
        response: Final = upload(gateway, "target_model_names", model, jsonl(fine, b"   ", b"", bare))
        assert response.status_code == 400, response.text
        assert str(error_of(response)["message"]).startswith(unconvertible_line_message(4)), response.text
        assert wire.drain() == ()


def _clean_and_messy(model: str) -> tuple[bytes, bytes]:
    first: Final = encoded(chat_row("first", model, "one"))
    second: Final = encoded(chat_row("second", model, "two"))
    third: Final = encoded(chat_row("third", model, "three"))
    messy: Final = b"".join(
        (BOM + first + b"\r\n", b"\r\n", b" \t\r\n", BOM + second + b"  \r\n", b"    " + third + b"\n", b"\n")
    )
    return jsonl(first, second, third), messy


@pytest.mark.parametrize("route", ROUTES)
def test_byte_order_marks_blank_lines_and_crlf_upload_the_same_rows_as_a_clean_file(
    gateway: Gateway, route: Route
) -> None:
    with wire_server(gcs_peer) as wire, gateway.scenario() as scenario:
        model: Final = vertex_model(gateway, scenario, wire.url)
        clean, messy = _clean_and_messy(model)
        clean_response: Final = upload(gateway, route, model, clean)
        assert clean_response.status_code == 200, clean_response.text
        clean_uploads: Final = wire.drain()
        messy_response: Final = upload(gateway, route, model, messy)
        assert messy_response.status_code == 200, messy_response.text
        messy_uploads: Final = wire.drain()
        assert (len(clean_uploads), len(messy_uploads)) == (1, 1), (clean_uploads, messy_uploads)
        assert messy_uploads[0].body == clean_uploads[0].body
        assert stored_custom_ids(messy_uploads[0]) == ("first", "second", "third")
        stored_texts: Final = tuple(
            json.loads(line)["request"]["contents"][0]["parts"][0]["text"]
            for line in messy_uploads[0].body.decode().split("\n")
        )
        assert stored_texts == ("one", "two", "three")


def _deeply_nested(model: str) -> bytes:
    depth: Final = 100_000
    return (
        b'{"custom_id": "deep", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "'
        + model.encode()
        + b'", "messages": '
        + b"[" * depth
        + b"]" * depth
        + b"}}"
    )


def _long_integer(model: str) -> bytes:
    body: Final = json.dumps(chat_row("long", model, "x")["body"]).encode()
    return b'{"custom_id": "long", "method": "POST", "url": "/v1/chat/completions", "body": ' + (
        body[:-1] + b', "seed": ' + b"7" * 5000 + b"}}"
    )


@pytest.mark.parametrize("row", (_deeply_nested, _long_integer), ids=("nested_100k_deep", "integer_5000_digits"))
def test_row_too_deep_or_long_to_parse_answers_400_naming_its_line(
    gateway: Gateway, row: Callable[[str], bytes]
) -> None:
    with wire_server(gcs_peer) as wire, gateway.scenario() as scenario:
        model: Final = vertex_model(gateway, scenario, wire.url)
        response: Final = upload(
            gateway, "model", model, jsonl(encoded(chat_row("first", model, "one")), b"", row(model))
        )
        assert response.status_code == 400, response.text
        assert error_of(response) == {
            "message": "Batch input file line 3 is not valid JSON. The file was not forwarded to the provider.",
            "type": "invalid_request_error",
            "param": "file",
            "code": "400",
        }, response.text
        assert wire.drain() == ()
        assert gateway.client.get("/health/liveliness").status_code == 200
        follow_up: Final = upload(gateway, "model", model, jsonl(encoded(chat_row("after", model, "ok"))))
        assert follow_up.status_code == 200, follow_up.text
        assert len(wire.drain()) == 1


VALIDATION_CASES: Final[tuple[tuple[str, bytes, str, str], ...]] = (
    ("truncated_json", b'{"custom_id": "bad", ', "Batch input file line 3 is not valid JSON.", "file"),
    ("invalid_utf8", b'{"custom_id": "\xff\xfe"}', "Batch input file line 3 is not valid JSON.", "file"),
    ("json_array", b'[{"custom_id": "bad"}]', "Batch input file line 3 must be a JSON object.", "file"),
    (
        "body_missing",
        b'{"custom_id": "bad", "method": "POST", "url": "/v1/chat/completions"}',
        "Missing required parameter: 'body' (batch input file line 3).",
        "body",
    ),
)


@pytest.mark.parametrize(
    ("bad_line", "message", "param"),
    tuple(case[1:] for case in VALIDATION_CASES),
    ids=tuple(case[0] for case in VALIDATION_CASES),
)
@pytest.mark.parametrize("route", ROUTES)
def test_rows_the_upload_validation_rejects_keep_their_400(
    gateway: Gateway, route: Route, bad_line: bytes, message: str, param: str
) -> None:
    with wire_server(gcs_peer) as wire, gateway.scenario() as scenario:
        model: Final = vertex_model(gateway, scenario, wire.url)
        response: Final = upload(gateway, route, model, jsonl(encoded(chat_row("first", model, "one")), b"", bad_line))
        assert response.status_code == 400, response.text
        error: Final = error_of(response)
        assert str(error["message"]).startswith(message), response.text
        assert str(error["message"]).endswith("The file was not forwarded to the provider."), response.text
        assert (error["type"], error["param"]) == ("invalid_request_error", param), response.text
        assert wire.drain() == ()


@pytest.mark.parametrize("route", ROUTES)
def test_empty_object_body_still_uploads(gateway: Gateway, route: Route) -> None:
    with wire_server(gcs_peer) as wire, gateway.scenario() as scenario:
        model: Final = vertex_model(gateway, scenario, wire.url)
        empty_body: Final = encoded({"custom_id": "empty", "method": "POST", "url": CHAT_URL, "body": {}})
        response: Final = upload(gateway, route, model, jsonl(encoded(chat_row("first", model, "one")), empty_body))
        assert response.status_code == 200, response.text
        uploads: Final = wire.drain()
        assert len(uploads) == 1
        assert stored_custom_ids(uploads[0]) == ("first", "empty")


def test_passthrough_upload_stores_the_callers_bytes_unchanged(gateway: Gateway) -> None:
    content: Final = b"".join(
        (
            BOM + b'{"request": {"contents": [{"role": "user", "parts": [{"text": "one"}]}]}}\r\n',
            b"\r\n",
            b'  {"request": {"contents": [{"role": "user", "parts": [{"text": "two"}]}]}}  \n',
        )
    )
    with wire_server(gcs_peer) as wire, gateway.scenario() as scenario:
        model: Final = vertex_model(gateway, scenario, wire.url)
        response: Final = upload(gateway, "target_model_names", model, content, passthrough="true")
        assert response.status_code == 200, response.text
        uploads: Final = wire.drain()
        assert len(uploads) == 1
        assert uploads[0].body == content


GCS_FAILURES: Final[tuple[tuple[str, Reply | None, int], ...]] = (
    (
        "bucket_answers_403",
        Reply(status=403, body=json.dumps({"error": {"code": 403, "message": "scripted denial"}}).encode()),
        403,
    ),
    ("bucket_drops_the_connection", Reply(drop_connection=True), 500),
    ("bucket_port_closed", None, 500),
)


@pytest.mark.parametrize(
    ("reply", "status"), tuple(case[1:] for case in GCS_FAILURES), ids=tuple(case[0] for case in GCS_FAILURES)
)
@pytest.mark.parametrize("route", ROUTES)
def test_bucket_failures_keep_their_status_and_never_blame_a_row(
    gateway: Gateway, route: Route, reply: Reply | None, status: int
) -> None:
    with wire_server(lambda _request: reply or Reply()) as wire, gateway.scenario() as scenario:
        url: Final = wire.url if reply is not None else f"http://127.0.0.1:{closed_port()}"
        model: Final = vertex_model(gateway, scenario, url)
        response: Final = upload(gateway, route, model, jsonl(encoded(chat_row("first", model, "one"))))
        assert response.status_code == status, response.text
        assert "Batch input file line" not in response.text, response.text


def _bom_then_blank_lines(first: bytes, fourth: bytes) -> bytes:
    return b"".join((BOM + first + b"\n", b"\n", b"  \n", fourth + b"\n"))


def _leading_blank_line(first: bytes, fourth: bytes) -> bytes:
    return b"".join((b"\n", first + b"\n", fourth + b"\n"))


@pytest.mark.parametrize("layout", (_bom_then_blank_lines, _leading_blank_line), ids=("bom_then_blank", "leading_blank"))
def test_router_rewrite_sends_openai_only_the_rows(gateway: Gateway, layout: Callable[[bytes, bytes], bytes]) -> None:
    with wire_server(openai_files_peer) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-4o-mini", api_key="integration-provider-key", api_base=f"{wire.url}/v1"
        )
        content: Final = layout(encoded(chat_row("first", model, "one")), encoded(chat_row("fourth", model, "four")))
        response: Final = upload(gateway, "target_model_names", model, content)
        assert response.status_code == 200, response.text
        uploads: Final = tuple(request for request in wire.drain() if is_openai_upload(request))
        assert len(uploads) == 1, [request.target for request in uploads]
        rows: Final = tuple(json.loads(line) for line in openai_uploaded_lines(uploads[0]))
        assert [(row["custom_id"], row["body"]["model"]) for row in rows] == [
            ("first", "gpt-4o-mini"),
            ("fourth", "gpt-4o-mini"),
        ]


def test_one_upload_to_openai_then_vertex_sends_openai_only_the_rows_and_names_the_vertex_file_line(
    gateway: Gateway,
) -> None:
    with (
        wire_server(openai_files_peer) as openai_wire,
        wire_server(gcs_peer) as gcs_wire,
        gateway.scenario() as scenario,
    ):
        openai_target: Final = scenario.model(
            model="openai/gpt-4o-mini", api_key="integration-provider-key", api_base=f"{openai_wire.url}/v1"
        )
        vertex_target: Final = vertex_model(gateway, scenario, gcs_wire.url)
        content: Final = jsonl(
            b"", encoded(chat_row("first", openai_target, "one")), b"", encoded(roleless_row("bad", openai_target))
        )
        response: Final = upload(gateway, "target_model_names", f"{openai_target},{vertex_target}", content)
        assert response.status_code == 400, response.text
        message: Final = str(error_of(response)["message"])
        assert message.startswith(unconvertible_line_message(4) + "missing field 'role'"), message
        uploads: Final = tuple(request for request in openai_wire.drain() if is_openai_upload(request))
        assert len(uploads) == 1, [request.target for request in uploads]
        rows: Final = tuple(json.loads(line) for line in openai_uploaded_lines(uploads[0]))
        assert [(row["custom_id"], row["body"]["model"]) for row in rows] == [
            ("first", "gpt-4o-mini"),
            ("bad", "gpt-4o-mini"),
        ]
        assert gcs_wire.drain() == ()


def _unreachable_anthropic_model(scenario: Scenario) -> str:
    return scenario.model(
        model="anthropic/claude-haiku-4-5",
        api_key="synthetic-anthropic-key",
        api_base=f"http://127.0.0.1:{closed_port()}",
        num_retries=0,
    )


def _proxy_url(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/")


def _consume(reply: object, stream: bool) -> None:
    if stream and isinstance(reply, Iterable):
        tuple(reply)


def _chat(client: openai.OpenAI, model: str, stream: bool) -> None:
    _consume(
        client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "hello"}], stream=stream, max_tokens=8
        ),
        stream,
    )


def _messages(client: anthropic.Anthropic, model: str, stream: bool) -> None:
    _consume(
        client.messages.create(
            model=model, messages=[{"role": "user", "content": "hello"}], max_tokens=8, stream=stream
        ),
        stream,
    )


def _responses(client: openai.OpenAI, model: str, stream: bool) -> None:
    _consume(client.responses.create(model=model, input="hello", stream=stream, max_output_tokens=16), stream)


@pytest.mark.parametrize("stream", (False, True), ids=("whole", "stream"))
def test_chat_transport_failure_keeps_its_500(gateway: Gateway, stream: bool) -> None:
    with gateway.scenario() as scenario:
        model: Final = _unreachable_anthropic_model(scenario)
        with openai.OpenAI(base_url=f"{_proxy_url(gateway)}/v1", api_key=gateway.key, max_retries=0) as client:
            with pytest.raises(openai.InternalServerError) as raised:
                _chat(client, model, stream)
        assert raised.value.status_code == 500
        assert "Batch input file" not in str(raised.value)
        raw: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "hi"}], "stream": stream},
        )
        assert raw.status_code == 500, raw.text


@pytest.mark.parametrize("stream", (False, True), ids=("whole", "stream"))
def test_messages_transport_failure_keeps_its_500(gateway: Gateway, stream: bool) -> None:
    with gateway.scenario() as scenario:
        model: Final = _unreachable_anthropic_model(scenario)
        with anthropic.Anthropic(base_url=_proxy_url(gateway), api_key=gateway.key, max_retries=0) as client:
            with pytest.raises(anthropic.InternalServerError) as raised:
                _messages(client, model, stream)
        assert raised.value.status_code == 500
        assert "Batch input file" not in str(raised.value)


@pytest.mark.parametrize("stream", (False, True), ids=("whole", "stream"))
def test_responses_transport_failure_keeps_its_500(gateway: Gateway, stream: bool) -> None:
    with gateway.scenario() as scenario:
        model: Final = _unreachable_anthropic_model(scenario)
        with openai.OpenAI(base_url=f"{_proxy_url(gateway)}/v1", api_key=gateway.key, max_retries=0) as client:
            with pytest.raises(openai.InternalServerError) as raised:
                _responses(client, model, stream)
        assert raised.value.status_code == 500
        assert "Batch input file" not in str(raised.value)
