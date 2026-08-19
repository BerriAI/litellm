import io
import json

import pytest
from fastapi import HTTPException

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.openai_files_endpoints.batch_guardrails import (
    RedactionRequired,
    UnparseableRecord,
    UnscannableRecord,
    raise_public,
    scan_batch_input_file,
)


def _record(custom_id, content="hello", url="/v1/chat/completions"):
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": url,
        "body": {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": content}],
        },
    }


def _jsonl(*records):
    return io.BytesIO("\n".join(json.dumps(r) for r in records).encode())


class FakeProxyLogging:
    """Stands in for ProxyLogging so the scan can be driven without a live proxy."""

    def __init__(self, on_record=None):
        self.on_record = on_record or (lambda data: None)
        self.seen = []

    async def pre_call_hook(self, user_api_key_dict, data, call_type, guardrails_only=False):
        self.seen.append((call_type, json.dumps(data.get("messages"), sort_keys=True)))
        self.on_record(data)
        return data

    def has_pre_call_guardrails(self, request_metadata):
        return True


def _redact_containing(needle):
    def _hook(data):
        for message in data.get("messages") or []:
            if isinstance(message.get("content"), str) and needle in message["content"]:
                message["content"] = message["content"].replace(needle, "***")

    return _hook


def _raise_on(needle, exc):
    def _hook(data):
        for message in data.get("messages") or []:
            if isinstance(message.get("content"), str) and needle in message["content"]:
                raise exc

    return _hook


async def _scan(source, logging_obj, metadata=None):
    return await scan_batch_input_file(
        file_source=source,
        request_metadata=metadata if metadata is not None else {},
        user_api_key_dict=UserAPIKeyAuth(api_key="sk-test"),
        proxy_logging_obj=logging_obj,
    )


@pytest.mark.asyncio
async def test_clean_file_passes_and_rewinds_the_handle():
    source = _jsonl(_record("a"), _record("b"), _record("c"))
    logging_obj = FakeProxyLogging()

    assert await _scan(source, logging_obj) is None
    assert len(logging_obj.seen) == 3
    assert source.tell() == 0, "handle must be rewound so the upload still sees the whole file"


@pytest.mark.asyncio
async def test_every_record_is_scanned_not_just_the_first():
    records = [_record(f"r{i}") for i in range(70)]
    logging_obj = FakeProxyLogging()

    assert await _scan(_jsonl(*records), logging_obj) is None
    assert len(logging_obj.seen) == 70, "records past the first scan window must still be scanned"


@pytest.mark.asyncio
async def test_redaction_is_reported_with_line_and_custom_id():
    source = _jsonl(_record("keep-1"), _record("dirty", content="my secret is here"), _record("keep-2"))

    failure = await _scan(source, FakeProxyLogging(_redact_containing("secret")))

    assert failure == RedactionRequired(line_number=2, custom_id="dirty")


@pytest.mark.asyncio
async def test_body_carrying_its_own_metadata_is_not_reported_as_redacted():
    record = _record("has-meta")
    record["body"]["metadata"] = {"team": "finance"}

    failure = await _scan(_jsonl(record), FakeProxyLogging(), metadata={"guardrails": ["x"]})

    assert failure is None, "the metadata the proxy injects must not be diffed as record content"


@pytest.mark.asyncio
async def test_guardrail_writing_bookkeeping_into_metadata_is_not_a_redaction():
    def _touch_metadata(data):
        data["litellm_metadata"]["applied_guardrails"] = ["some-guard"]

    assert await _scan(_jsonl(_record("a")), FakeProxyLogging(_touch_metadata)) is None


@pytest.mark.asyncio
async def test_records_own_metadata_is_left_out_of_the_scan_and_the_diff():
    """Guardrail dispatch writes bookkeeping into `metadata`; diffing it would reject every such record."""
    record = _record("has-meta")
    record["body"]["metadata"] = {"team": "finance"}
    seen = []

    def _write_bookkeeping(data):
        seen.append("metadata" in data)
        data.setdefault("metadata", {})["applied_guardrails"] = ["g"]

    assert await _scan(_jsonl(record), FakeProxyLogging(_write_bookkeeping)) is None
    assert seen == [False], "the record's own metadata must not be handed to guardrail dispatch"
    assert record["body"]["metadata"] == {"team": "finance"}


@pytest.mark.asyncio
async def test_request_metadata_is_narrowed_to_what_guardrails_read():
    """An OTel-enabled proxy puts a lock-bearing span here; a per-record copy of it is a crash."""
    import threading

    seen = []
    metadata = {
        "guardrails": ["g"],
        "tags": ["t"],
        "litellm_parent_otel_span": threading.RLock(),
        "user_api_key": "sk-secret",
    }

    failure = await _scan(
        _jsonl(_record("a")),
        FakeProxyLogging(lambda d: seen.append(dict(d["litellm_metadata"]))),
        metadata=metadata,
    )

    assert failure is None
    assert seen == [{"guardrails": ["g"], "tags": ["t"]}]


@pytest.mark.asyncio
async def test_guardrail_that_adds_a_key_is_detected():
    def _add_key(data):
        data["mock_response"] = "intercepted"

    failure = await _scan(_jsonl(_record("a")), FakeProxyLogging(_add_key))

    assert failure == RedactionRequired(line_number=1, custom_id="a")


@pytest.mark.asyncio
async def test_guardrail_that_adds_a_null_valued_key_is_detected():
    """A null value must not read the same as a missing key, or dropping one hides a change."""

    def _add_null_key(data):
        data["response_format"] = None

    failure = await _scan(_jsonl(_record("a")), FakeProxyLogging(_add_null_key))

    assert failure == RedactionRequired(line_number=1, custom_id="a")


@pytest.mark.asyncio
async def test_guardrail_that_drops_a_null_valued_key_is_detected():
    def _drop_null_key(data):
        data.pop("response_format")

    record = _record("a")
    record["body"]["response_format"] = None

    failure = await _scan(_jsonl(record), FakeProxyLogging(_drop_null_key))

    assert failure == RedactionRequired(line_number=1, custom_id="a")


@pytest.mark.asyncio
async def test_guardrail_that_only_reorders_a_nested_dict_is_not_a_redaction():
    def _reorder(data):
        message = data["messages"][0]
        data["messages"][0] = {key: message[key] for key in reversed(list(message))}

    assert await _scan(_jsonl(_record("a")), FakeProxyLogging(_reorder)) is None


@pytest.mark.asyncio
async def test_non_utf8_bytes_are_a_client_error_not_a_crash():
    assert await _scan(io.BytesIO(b"\xff\xfe not utf8\n"), FakeProxyLogging()) == UnparseableRecord(line_number=1)


@pytest.mark.asyncio
async def test_record_without_a_url_falls_back_to_its_body_shape():
    logging_obj = FakeProxyLogging()
    record = _record("no-url")
    del record["url"]

    assert await _scan(_jsonl(record), logging_obj) is None
    assert logging_obj.seen[0][0] == "acompletion"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body, expected_call_type",
    [
        ({"messages": [{"role": "user", "content": "x"}]}, "acompletion"),
        ({"prompt": "x"}, "atext_completion"),
        ({"input": "x"}, "aembedding"),
    ],
)
async def test_empty_url_falls_back_to_its_body_shape(body, expected_call_type):
    logging_obj = FakeProxyLogging()
    record = {"custom_id": "c", "url": "", "body": {"model": "m", **body}}

    assert await _scan(_jsonl(record), logging_obj) is None
    assert logging_obj.seen[0][0] == expected_call_type


@pytest.mark.asyncio
async def test_blocking_guardrail_outranks_an_earlier_refused_record():
    """PR 2 turns RedactionRequired into a non-failure; a block must not be lost behind it."""
    blocked = HTTPException(status_code=403, detail={"error": "Violated guardrail policy"})

    def _hook(data):
        content = data["messages"][0]["content"]
        if content == "raiser":
            raise blocked
        if content == "redact":
            data["messages"][0]["content"] = "***"

    source = _jsonl(_record("a", content="redact"), _record("b", content="raiser"))

    with pytest.raises(HTTPException) as raised:
        await _scan(source, FakeProxyLogging(_hook))

    assert raised.value is blocked


@pytest.mark.asyncio
async def test_handle_is_rewound_even_when_a_record_is_refused():
    source = _jsonl(_record("a", content="secret"))

    await _scan(source, FakeProxyLogging(_redact_containing("secret")))

    assert source.tell() == 0


@pytest.mark.asyncio
async def test_unparseable_record_is_rejected():
    source = io.BytesIO(b'{"custom_id": "ok", "url": "/v1/chat/completions", "body": {}}\n{ not json\n')

    assert await _scan(source, FakeProxyLogging()) == UnparseableRecord(line_number=2)


@pytest.mark.asyncio
async def test_record_without_a_body_object_is_rejected():
    source = io.BytesIO(b'{"custom_id": "no-body", "url": "/v1/chat/completions"}\n')

    assert await _scan(source, FakeProxyLogging()) == UnparseableRecord(line_number=1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url, expected_call_type",
    [
        ("/v1/chat/completions", "acompletion"),
        ("/v1/completions", "atext_completion"),
        ("/v1/embeddings", "aembedding"),
        ("/v1/responses", "aresponses"),
        ("/v1/messages", "anthropic_messages"),
    ],
)
async def test_supported_urls_scan_under_the_matching_call_type(url, expected_call_type):
    logging_obj = FakeProxyLogging()

    assert await _scan(_jsonl(_record("a", url=url)), logging_obj) is None
    assert logging_obj.seen[0][0] == expected_call_type


@pytest.mark.asyncio
async def test_unrecognized_url_falls_back_to_the_body_shape():
    """A record we can still read is a record we can still scan, so the url alone must not reject it."""
    logging_obj = FakeProxyLogging()

    assert await _scan(_jsonl(_record("img", url="/v1/images/generations")), logging_obj) is None
    assert logging_obj.seen[0][0] == "acompletion"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    ["/chat/completions", "/v1/chat/completions/", "https://api.openai.com/v1/chat/completions"],
)
async def test_url_variants_callers_actually_write_are_accepted(url):
    logging_obj = FakeProxyLogging()

    assert await _scan(_jsonl(_record("v", url=url)), logging_obj) is None
    assert logging_obj.seen[0][0] == "acompletion"


@pytest.mark.asyncio
async def test_query_string_on_a_known_url_does_not_change_the_call_type():
    """The body carries `messages`, so only stripping the query string can yield aembedding."""
    logging_obj = FakeProxyLogging()
    record = {
        "custom_id": "q",
        "url": "/v1/embeddings?api-version=1",
        "body": {"model": "m", "input": "x", "messages": [{"role": "user", "content": "y"}]},
    }

    assert await _scan(_jsonl(record), logging_obj) is None
    assert logging_obj.seen[0][0] == "aembedding"


@pytest.mark.asyncio
async def test_record_whose_body_cannot_be_read_is_rejected():
    source = _jsonl({"custom_id": "opaque", "url": "/v1/rerank", "body": {"model": "m", "documents": ["a"]}})

    failure = await _scan(source, FakeProxyLogging())

    assert failure == UnscannableRecord(line_number=1, custom_id="opaque", url="/v1/rerank")


@pytest.mark.asyncio
async def test_url_less_record_whose_body_shape_is_unknown_is_rejected():
    record = {"custom_id": "opaque", "body": {"model": "m", "something_else": 1}}

    assert await _scan(_jsonl(record), FakeProxyLogging()) == UnscannableRecord(
        line_number=1, custom_id="opaque", url=None
    )


@pytest.mark.asyncio
async def test_blocking_guardrail_exception_propagates_unwrapped():
    blocked = HTTPException(status_code=403, detail={"error": "Violated guardrail policy"})
    source = _jsonl(_record("a"), _record("b", content="tripwire"))

    with pytest.raises(HTTPException) as raised:
        await _scan(source, FakeProxyLogging(_raise_on("tripwire", blocked)))

    assert raised.value is blocked, "the guardrail's own exception must survive so its status code does"
    assert raised.value.status_code == 403


@pytest.mark.asyncio
async def test_earliest_refused_record_is_the_one_reported():
    source = _jsonl(_record("a"), _record("b", content="secret"), _record("c", content="secret"))

    failure = await _scan(source, FakeProxyLogging(_redact_containing("secret")))

    assert failure == RedactionRequired(line_number=2, custom_id="b")


@pytest.mark.asyncio
async def test_earliest_failing_record_wins_when_the_raise_comes_first():
    blocked = HTTPException(status_code=400, detail="blocked")

    def _hook(data):
        content = data["messages"][0]["content"]
        if content == "raiser":
            raise blocked
        if content == "redact":
            data["messages"][0]["content"] = "***"

    source = _jsonl(
        _record("a", content="raiser"),
        _record("b", content="redact"),
    )

    with pytest.raises(HTTPException) as raised:
        await _scan(source, FakeProxyLogging(_hook))

    assert raised.value is blocked


@pytest.mark.asyncio
async def test_records_are_not_mutated_by_the_scan():
    record = _record("a", content="my secret is here")
    payload = json.dumps(record)
    source = io.BytesIO(payload.encode())

    await _scan(source, FakeProxyLogging(_redact_containing("secret")))

    assert source.getvalue().decode() == payload, "the scan must never rewrite the uploaded bytes"


@pytest.mark.parametrize(
    "failure, fragment",
    [
        (UnparseableRecord(line_number=7), "line 7"),
        (UnscannableRecord(line_number=3, custom_id="x", url="/v1/audio/speech"), "custom_id x"),
        (RedactionRequired(line_number=2, custom_id=None), "line 2"),
    ],
)
def test_every_failure_maps_to_a_400_naming_the_record(failure, fragment):
    with pytest.raises(HTTPException) as raised:
        raise_public(failure)

    assert raised.value.status_code == 400
    assert fragment in raised.value.detail["error"]


@pytest.mark.asyncio
async def test_scan_does_not_mutate_the_parsed_record():
    """The guardrail must redact a copy. Mutating the record would corrupt what PR 2 writes out."""
    from litellm.proxy.openai_files_endpoints.batch_guardrails import _ParsedRecord, _scan_record

    record = _ParsedRecord(line_number=1, payload=_record("a", content="my secret is here"))

    failure = await _scan_record(
        record,
        {},
        UserAPIKeyAuth(api_key="sk-test"),
        FakeProxyLogging(_redact_containing("secret")),
    )

    assert failure == RedactionRequired(line_number=1, custom_id="a")
    assert record.payload["body"]["messages"][0]["content"] == "my secret is here", (
        "the guardrail redacted the record itself instead of a copy"
    )


@pytest.mark.asyncio
async def test_non_object_json_line_is_rejected():
    source = io.BytesIO(b"[1, 2, 3]\n")

    assert await _scan(source, FakeProxyLogging()) == UnparseableRecord(line_number=1)


@pytest.mark.asyncio
async def test_scan_is_bounded_so_a_huge_file_cannot_fan_out_without_limit():
    import asyncio

    from litellm.proxy.openai_files_endpoints.batch_guardrails import _SCAN_WINDOW

    in_flight = {"now": 0, "peak": 0}

    class CountingLogging(FakeProxyLogging):
        async def pre_call_hook(self, user_api_key_dict, data, call_type, guardrails_only=False):
            in_flight["now"] += 1
            in_flight["peak"] = max(in_flight["peak"], in_flight["now"])
            await asyncio.sleep(0)
            in_flight["now"] -= 1
            return data

    records = [_record(f"r{i}") for i in range(_SCAN_WINDOW * 3)]

    assert await _scan(_jsonl(*records), CountingLogging()) is None
    assert in_flight["peak"] <= _SCAN_WINDOW, (
        f"peak {in_flight['peak']} exceeded the scan window; a gigabyte file would fan out unbounded"
    )


@pytest.mark.asyncio
async def test_scan_runs_guardrails_only():
    """Rate limiters, budget hooks and the hanging-request alert must not fire once per record."""
    flags = []

    class FlagCapturingLogging(FakeProxyLogging):
        async def pre_call_hook(self, user_api_key_dict, data, call_type, guardrails_only=False):
            flags.append(guardrails_only)
            return data

    await _scan(_jsonl(_record("a"), _record("b")), FlagCapturingLogging())

    assert flags == [True, True]


@pytest.mark.asyncio
async def test_guardrail_that_returns_a_replacement_dict_is_detected():
    """async_pre_call_hook may return a NEW dict instead of mutating; that result is the real input."""

    class ReplacingLogging(FakeProxyLogging):
        async def pre_call_hook(self, user_api_key_dict, data, call_type, guardrails_only=False):
            replacement = json.loads(json.dumps(data))
            replacement["messages"][0]["content"] = "***"
            return replacement

    failure = await _scan(_jsonl(_record("a", content="my secret is here")), ReplacingLogging())

    assert failure == RedactionRequired(line_number=1, custom_id="a")
