"""
Unit tests for litellm/batches/batch_utils.py

batch_utils.py is the batch cost/usage/parsing layer: it turns a batch output
JSONL into spend (cost), token usage, and the list of models seen, and counts
tokens in batch *input* files for rate limiting. A silent bug here mis-bills
real money or lets callers slip past TPM limits, so these tests assert exact
numeric results rather than "ran without error".

Pure functions (parsing, token math, credential extraction, success checks) run
for real with exact-value assertions. The few true external seams - the cost
maps (litellm.completion_cost, batch_cost_calculator), the tokenizer
(token_counter), and remote file fetch (afile_content) - are mocked with
deterministic stand-ins so the arithmetic under test is the only variable.
"""

import json
import logging
from types import MappingProxyType

import httpx
import pytest
import respx


import litellm
import litellm.batches.batch_utils as bu
from litellm.types.utils import Usage

# --------------------------------------------------------------------------- #
# Builders for batch OUTPUT file rows.
# Shape: {"response": {"status_code": 200, "body": {... "usage": {...}}}}
# --------------------------------------------------------------------------- #


def _usage(p, c, t=None):
    return {
        "prompt_tokens": p,
        "completion_tokens": c,
        "total_tokens": t if t is not None else p + c,
    }


def _success_row(model="gpt-4o", usage=None, **body_extra):
    body = {"model": model, **body_extra}
    if usage is not None:
        body["usage"] = usage
    return {"response": {"status_code": 200, "body": body}}


def _failed_row(status_code=500, model="gpt-4o"):
    return {"response": {"status_code": status_code, "body": {"model": model}}}


# =========================================================================== #
# _batch_response_was_successful
# =========================================================================== #


@pytest.mark.parametrize(
    "row,expected",
    [
        ({"response": {"status_code": 200}}, True),
        ({"response": {"status_code": 500}}, False),
        ({"response": {"status_code": 429}}, False),
        ({"response": {}}, False),  # no status_code
        ({}, False),  # no response
        ({"response": None}, False),  # null response
    ],
)
def test_batch_response_was_successful(row, expected):
    assert bu._batch_response_was_successful(row) is expected


# =========================================================================== #
# _get_response_from_batch_job_output_file
# =========================================================================== #


def test_get_response_body_present():
    row = {"response": {"body": {"model": "gpt-4o", "usage": {"x": 1}}}}
    assert bu._get_response_from_batch_job_output_file(row) == {
        "model": "gpt-4o",
        "usage": {"x": 1},
    }


@pytest.mark.parametrize(
    "row",
    [
        {},  # no response
        {"response": {}},  # no body
        {"response": None},  # null response
        {"response": {"body": None}},  # null body
    ],
)
def test_get_response_body_missing_returns_empty(row):
    assert bu._get_response_from_batch_job_output_file(row) == {}


# =========================================================================== #
# _get_batch_job_usage_from_response_body
# =========================================================================== #


def test_get_usage_from_response_body():
    usage = bu._get_batch_job_usage_from_response_body(
        {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
    )
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (
        10,
        5,
        15,
    )


def test_get_usage_from_response_body_missing_is_zero():
    usage = bu._get_batch_job_usage_from_response_body({})
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (
        0,
        0,
        0,
    )


# =========================================================================== #
# _get_file_content_as_dictionary  (JSONL parsing)
# =========================================================================== #


def test_parse_jsonl_multiple_lines():
    content = b'{"a": 1}\n{"b": 2}\n{"c": 3}'
    assert bu._get_file_content_as_dictionary(content) == [
        {"a": 1},
        {"b": 2},
        {"c": 3},
    ]


def test_parse_jsonl_trailing_newline_skipped():
    # outer content is stripped; the trailing-newline empty line is dropped.
    content = b'{"a": 1}\n{"b": 2}\n'
    assert bu._get_file_content_as_dictionary(content) == [{"a": 1}, {"b": 2}]


def test_parse_jsonl_empty_content_is_empty_list():
    assert bu._get_file_content_as_dictionary(b"") == []


def test_parse_jsonl_malformed_lines_skipped():
    content = b'{"a": 1}\nnot valid json\n{"b": 2}\n'
    assert bu._get_file_content_as_dictionary(content) == [{"a": 1}, {"b": 2}]


# =========================================================================== #
# _iter_batch_input_lines / _iter_batch_output_entries  (JSONL parsing)
# =========================================================================== #


def test_iter_input_lines_skips_blank_and_strips():
    content = b'{"a":1}\n\n  \n{"b":2}\n'
    assert list(bu._iter_batch_input_lines(content)) == [b'{"a":1}', b'{"b":2}']


def test_iter_input_lines_handles_missing_trailing_newline():
    assert list(bu._iter_batch_input_lines(b'{"a":1}')) == [b'{"a":1}']


def test_iter_input_lines_empty():
    assert list(bu._iter_batch_input_lines(b"")) == []


def test_iter_output_entries_parses_each_row():
    content = b'{"body": {"model": "gpt-4o"}}\n{"body": {"model": "claude-3"}}\n'
    assert list(bu._iter_batch_output_entries(content)) == [
        {"body": {"model": "gpt-4o"}},
        {"body": {"model": "claude-3"}},
    ]


def test_iter_output_entries_skips_malformed_and_non_object_lines():
    content = b'{"ok": 1}\nnot-json\n[1, 2]\n{"ok": 2}\n'
    assert list(bu._iter_batch_output_entries(content)) == [{"ok": 1}, {"ok": 2}]


def test_iter_output_entries_skips_undecodable_line():
    content = b'{"ok": 1}\n{"note": "\xff-bad"}\n{"ok": 2}\n'
    assert list(bu._iter_batch_output_entries(content)) == [{"ok": 1}, {"ok": 2}]


# =========================================================================== #
# _estimate_batch_entry_tokens  (regression: an uncountable/malformed row must
# never contribute zero tokens, or a crafted batch could evade the TPM limit)
# =========================================================================== #


def test_estimate_tokens_scales_with_size():
    # 4 bytes per token, floored, with a minimum of 1.
    assert bu._estimate_batch_entry_tokens(b"a" * 40) == 10


def test_estimate_tokens_never_zero_for_short_rows():
    assert bu._estimate_batch_entry_tokens(b"") == 1
    assert bu._estimate_batch_entry_tokens(b"abc") == 1


# =========================================================================== #
# _aggregate_batch_cost_usage_models: models  (output file)
# =========================================================================== #


def test_output_models_uses_model_name_override(monkeypatch):
    monkeypatch.setattr(litellm, "completion_cost", lambda **kw: 0.0)
    _, _, models = bu._aggregate_batch_cost_usage_models(
        entries=[_success_row(model="ignored")], custom_llm_provider="openai", model_name="forced-model"
    )
    assert models == ["forced-model"]


def test_output_models_collects_from_successful_only(monkeypatch):
    monkeypatch.setattr(litellm, "completion_cost", lambda **kw: 0.0)
    rows = [
        _success_row(model="gpt-4o"),
        _failed_row(model="should-be-skipped"),
        _success_row(model="claude-3"),
    ]
    _, _, models = bu._aggregate_batch_cost_usage_models(entries=rows, custom_llm_provider="openai")
    assert models == ["gpt-4o", "claude-3"]


def test_output_models_skips_successful_without_model(monkeypatch):
    monkeypatch.setattr(litellm, "completion_cost", lambda **kw: 0.0)
    rows = [{"response": {"status_code": 200, "body": {}}}]
    _, _, models = bu._aggregate_batch_cost_usage_models(entries=rows, custom_llm_provider="openai")
    assert models == []


# =========================================================================== #
# _extract_file_access_credentials
# =========================================================================== #


def test_extract_credentials_only_known_keys():
    params = {
        "api_key": "sk-1",
        "api_base": "https://b",
        "vertex_project": "proj",
        "gcs_bucket_name": "my-bucket",
        "bucket_name": "my-alias-bucket",
        "model": "gpt-4o",  # not a credential key
        "unrelated": "x",
    }
    assert bu._extract_file_access_credentials(params) == {
        "api_key": "sk-1",
        "api_base": "https://b",
        "vertex_project": "proj",
        "gcs_bucket_name": "my-bucket",
        "bucket_name": "my-alias-bucket",
    }


@pytest.mark.parametrize("params", [None, {}])
def test_extract_credentials_empty(params):
    assert bu._extract_file_access_credentials(params) == {}


def test_extract_credentials_all_supported_keys():
    keys = {
        "api_key",
        "api_base",
        "api_version",
        "organization",
        "azure_ad_token",
        "azure_ad_token_provider",
        "vertex_project",
        "vertex_location",
        "vertex_credentials",
        "gcs_bucket_name",
        "bucket_name",
        "timeout",
        "max_retries",
    }
    params = {k: f"val-{k}" for k in keys}
    assert bu._extract_file_access_credentials(params) == params


# =========================================================================== #
# _count_prompt_or_input_tokens  (regression-critical: list[list[int]] used to
# count as zero and let callers slip past TPM limits). token_counter stubbed to
# len(text) so every shape has an exact expected value.
# =========================================================================== #


@pytest.fixture
def fake_token_counter(monkeypatch):
    def _tc(model=None, text=None, messages=None, **kw):
        if messages is not None:
            return len(messages)
        if text is not None:
            return len(text)
        return 0

    monkeypatch.setattr(bu, "token_counter", _tc)
    return _tc


def test_count_tokens_str(fake_token_counter):
    assert bu._count_prompt_or_input_tokens("m", "hello") == 5  # len("hello")


def test_count_tokens_list_of_str(fake_token_counter):
    assert bu._count_prompt_or_input_tokens("m", ["ab", "cde"]) == 5  # 2 + 3


def test_count_tokens_list_of_int(fake_token_counter):
    # pre-tokenized prompt: each int counts as one token.
    assert bu._count_prompt_or_input_tokens("m", [1, 2, 3, 4]) == 4


def test_count_tokens_list_of_list_of_int(fake_token_counter):
    # the bug-fix shape: nested pre-tokenized prompts, each int = 1 token.
    assert bu._count_prompt_or_input_tokens("m", [[1, 2, 3], [4, 5]]) == 5


def test_count_tokens_mixed_nested(fake_token_counter):
    # nested list with ints + a string: 2 ints (=2) + len("xyz")=3 -> 5
    assert bu._count_prompt_or_input_tokens("m", [[1, 2, "xyz"]]) == 5


def test_count_tokens_unsupported_shape_is_zero(fake_token_counter):
    assert bu._count_prompt_or_input_tokens("m", 12345) == 0
    assert bu._count_prompt_or_input_tokens("m", {"a": 1}) == 0


# =========================================================================== #
# _count_entry_tokens  (per-entry rate-limit token counting). The individual
# prompt/input/embedding shapes are covered in test_batch_file_validation.py;
# here we pin the body-field precedence and the empty/fallback behavior.
# =========================================================================== #


def test_count_entry_messages_path(fake_token_counter):
    entry = {"body": {"model": "gpt-4o", "messages": [{"role": "user"}, {"role": "x"}]}}
    assert bu._count_entry_tokens(entry) == 2  # len(messages)


def test_count_entry_prompt_path(fake_token_counter):
    assert bu._count_entry_tokens({"body": {"model": "gpt-4o", "prompt": "abcd"}}) == 4


def test_count_entry_input_path(fake_token_counter):
    assert bu._count_entry_tokens({"body": {"model": "gpt-4o", "input": "ab"}}) == 2


def test_count_entry_messages_beats_prompt(fake_token_counter):
    # messages present -> prompt/input are ignored (messages is checked first).
    entry = {
        "body": {
            "model": "gpt-4o",
            "messages": [{"role": "user"}],
            "prompt": "this-should-be-ignored",
        }
    }
    assert bu._count_entry_tokens(entry) == 1


def test_count_entry_prompt_beats_input(fake_token_counter):
    entry = {"body": {"model": "gpt-4o", "prompt": "abc", "input": "this-is-longer"}}
    assert bu._count_entry_tokens(entry) == 3


def test_count_entry_empty_body_is_zero(fake_token_counter):
    assert bu._count_entry_tokens({"body": {}}) == 0
    assert bu._count_entry_tokens({}) == 0


def test_count_entry_uses_model_name_fallback(monkeypatch):
    # No body.model -> the model_name argument is forwarded to the token counter.
    captured = {}

    def _tc(model=None, text=None, messages=None, **kw):
        captured["model"] = model
        return len(text or "")

    monkeypatch.setattr(bu, "token_counter", _tc)
    bu._count_entry_tokens({"body": {"prompt": "ab"}}, model_name="fallback-model")
    assert captured["model"] == "fallback-model"


# =========================================================================== #
# _aggregate_batch_cost_usage_models: usage  (output usage aggregation)
# =========================================================================== #


def test_total_usage_sums_successful_only(monkeypatch):
    monkeypatch.setattr(litellm, "completion_cost", lambda **kw: 0.0)
    rows = [
        _success_row(usage=_usage(10, 5)),  # 15
        _failed_row(),  # excluded
        _success_row(usage=_usage(20, 10)),  # 30
    ]
    _, usage, _ = bu._aggregate_batch_cost_usage_models(entries=rows, custom_llm_provider="openai")
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (
        30,
        15,
        45,
    )


def test_total_usage_and_cost_normalize_mixed_responses_and_chat():
    responses_row = _success_row(
        usage={
            "input_tokens": 20,
            "output_tokens": 7,
            "total_tokens": 27,
            "input_tokens_details": {"cached_tokens": 3},
        }
    )
    chat_row = _success_row(usage=_usage(10, 5))

    cost, usage, _ = bu._aggregate_batch_cost_usage_models(
        entries=[responses_row, chat_row],
        custom_llm_provider="openai",
        model_info={
            "input_cost_per_token_batches": 0.00125,
            "output_cost_per_token_batches": 0.005,
        },
    )

    assert usage.prompt_tokens == 30
    assert usage.completion_tokens == 12
    assert usage.total_tokens == 42
    assert usage.cache_read_input_tokens == 3
    assert cost == pytest.approx((30 * 0.00125) + (12 * 0.005))


def test_total_usage_empty_is_zero():
    cost, usage, models = bu._aggregate_batch_cost_usage_models(entries=[], custom_llm_provider="openai")
    assert cost == 0.0
    assert models == []
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (
        0,
        0,
        0,
    )


# =========================================================================== #
# _aggregate_batch_cost_usage_models: cost  (cost maps mocked)
# =========================================================================== #


def test_cost_from_content_completion_cost_path(monkeypatch):
    # model_info is None -> litellm.completion_cost per successful row.
    calls = []

    def _completion_cost(**kw):
        calls.append(kw)
        return 0.5

    monkeypatch.setattr(litellm, "completion_cost", _completion_cost)
    rows = [
        _success_row(usage=_usage(10, 5)),
        _failed_row(),  # excluded -> not costed
        _success_row(usage=_usage(20, 10)),
    ]

    total, _, _ = bu._aggregate_batch_cost_usage_models(entries=rows, custom_llm_provider="openai")

    assert total == 1.0  # 2 successful * 0.5
    assert len(calls) == 2  # failed row not costed


def test_empty_body_line_does_not_zero_whole_batch():
    """A status-200 row with an empty body makes litellm.completion_cost raise;
    that line must be skipped instead of zeroing the whole batch."""
    rows = [
        _success_row(usage=_usage(10, 5)),
        {
            "custom_id": "request-poison-empty",
            "response": {"status_code": 200, "request_id": "inject-empty-body", "body": {}},
        },
        _success_row(usage=_usage(20, 10)),
    ]

    cost, usage, models = bu._aggregate_batch_cost_usage_models(entries=rows, custom_llm_provider="openai")

    assert cost > 0.0
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (30, 15, 45)
    assert models == ["gpt-4o", "gpt-4o"]


def test_cost_from_content_model_info_path(monkeypatch):
    # model_info set -> batch_cost_calculator(prompt_cost, completion_cost).
    import litellm.cost_calculator as cc

    monkeypatch.setattr(cc, "batch_cost_calculator", lambda **kw: (0.1, 0.2))
    rows = [
        _success_row(usage=_usage(10, 5)),
        _success_row(usage=_usage(20, 10)),
    ]

    total, _, _ = bu._aggregate_batch_cost_usage_models(
        entries=rows,
        custom_llm_provider="openai",
        model_info={"input_cost_per_token": 0.0},  # type: ignore[arg-type]  # truthy -> model_info path
    )

    assert total == pytest.approx(0.6)  # 2 * (0.1 + 0.2)


def test_aggregate_consumes_entries_in_a_single_pass(monkeypatch):
    """A one-shot generator: any implementation that iterates the entries twice
    (e.g. separate cost and usage passes) sees nothing on the second pass and
    returns wrong totals for at least one of cost/usage/models."""
    monkeypatch.setattr(litellm, "completion_cost", lambda **kw: 0.5)
    one_shot = (row for row in [_success_row(usage=_usage(10, 5)), _failed_row(), _success_row(usage=_usage(20, 10))])

    cost, usage, models = bu._aggregate_batch_cost_usage_models(entries=one_shot, custom_llm_provider="openai")

    assert cost == 1.0
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (30, 15, 45)
    assert models == ["gpt-4o", "gpt-4o"]


# =========================================================================== #
# calculate_batch_cost_and_usage  (dispatch: vertex-disable-transform vs generic)
# =========================================================================== #


@pytest.mark.asyncio
async def test_calculate_vertex_disable_transform_path(monkeypatch):
    monkeypatch.setattr(litellm, "disable_vertex_batch_output_transformation", True, raising=False)
    monkeypatch.setattr(
        bu,
        "calculate_vertex_ai_batch_cost_and_usage",
        lambda content, model: (9.9, Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3)),
    )
    # generic path must NOT be taken
    monkeypatch.setattr(
        bu,
        "_aggregate_batch_cost_usage_models",
        lambda **kw: pytest.fail("generic path should not run"),
    )

    cost, usage, models = await bu.calculate_batch_cost_and_usage(
        file_content_dictionary=[], custom_llm_provider="vertex_ai", model_name="gemini-2.0-flash-001"
    )
    assert cost == 9.9
    assert usage.total_tokens == 3
    assert models == ["gemini-2.0-flash-001"]


@pytest.mark.asyncio
async def test_calculate_vertex_disable_transform_needs_model_name(monkeypatch):
    """Without a model_name the raw-vertex path cannot price lines; the generic
    aggregation path must run even with the disable flag set."""
    monkeypatch.setattr(litellm, "disable_vertex_batch_output_transformation", True, raising=False)
    monkeypatch.setattr(
        bu,
        "calculate_vertex_ai_batch_cost_and_usage",
        lambda content, model: pytest.fail("raw vertex path should not run"),
    )

    cost, usage, models = await bu.calculate_batch_cost_and_usage(
        file_content_dictionary=[], custom_llm_provider="vertex_ai"
    )
    assert cost == 0.0
    assert usage.total_tokens == 0
    assert models == []


# =========================================================================== #
# calculate_vertex_ai_batch_cost_and_usage  (usageMetadata aggregation)
# =========================================================================== #


def test_vertex_cost_and_usage_aggregation(monkeypatch):
    import litellm.cost_calculator as cc

    monkeypatch.setattr(cc, "batch_cost_calculator", lambda **kw: (0.1, 0.2))
    responses = [
        {
            "response": {
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                    "totalTokenCount": 15,
                }
            }
        },
        {
            "response": {
                "usageMetadata": {
                    "promptTokenCount": 20,
                    "candidatesTokenCount": 10,
                    "totalTokenCount": 30,
                }
            }
        },
    ]

    cost, usage = bu.calculate_vertex_ai_batch_cost_and_usage(responses, "gemini-x")

    assert cost == pytest.approx(0.6)  # 2 * (0.1 + 0.2)
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (
        30,
        15,
        45,
    )


def test_vertex_cost_skips_none_response_body(monkeypatch):
    import litellm.cost_calculator as cc

    monkeypatch.setattr(cc, "batch_cost_calculator", lambda **kw: (1.0, 0.0))
    responses = [
        {"response": None},  # skipped
        {
            "response": {
                "usageMetadata": {
                    "promptTokenCount": 7,
                    "candidatesTokenCount": 3,
                    "totalTokenCount": 10,
                }
            }
        },
    ]

    cost, usage = bu.calculate_vertex_ai_batch_cost_and_usage(responses, "gemini-x")

    assert cost == pytest.approx(1.0)  # only one line costed
    assert usage.total_tokens == 10


def test_vertex_usage_total_token_fallback(monkeypatch):
    # no totalTokenCount -> falls back to prompt + completion.
    import litellm.cost_calculator as cc

    monkeypatch.setattr(cc, "batch_cost_calculator", lambda **kw: (0.0, 0.0))
    responses = [{"response": {"usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 4}}}]

    _, usage = bu.calculate_vertex_ai_batch_cost_and_usage(responses, "gemini-x")
    assert usage.total_tokens == 12


def test_vertex_cost_error_in_line_is_swallowed(monkeypatch):
    # a cost error on one line must not abort aggregation; usage still tallies.
    import litellm.cost_calculator as cc

    def _boom(**kw):
        raise RuntimeError("price map miss")

    monkeypatch.setattr(cc, "batch_cost_calculator", _boom)
    responses = [
        {
            "response": {
                "usageMetadata": {
                    "promptTokenCount": 5,
                    "candidatesTokenCount": 5,
                    "totalTokenCount": 10,
                }
            }
        }
    ]

    cost, usage = bu.calculate_vertex_ai_batch_cost_and_usage(responses, "gemini-x")
    assert cost == 0.0
    assert usage.total_tokens == 10


# =========================================================================== #
# calculate_batch_cost_and_usage  (async orchestrator)
# =========================================================================== #


@pytest.mark.asyncio
async def test_calculate_batch_cost_and_usage_orchestration(monkeypatch):
    rows = [_success_row(model="gpt-4o", usage=_usage(10, 5))]
    monkeypatch.setattr(litellm, "completion_cost", lambda **kw: 2.5)

    cost, usage, models = await bu.calculate_batch_cost_and_usage(
        file_content_dictionary=rows, custom_llm_provider="openai"
    )

    assert cost == 2.5
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (10, 5, 15)
    assert models == ["gpt-4o"]


# =========================================================================== #
# _fetch_batch_output_file_content  (file fetch + credential merge)
# =========================================================================== #


def _batch(output_file_id):
    from litellm.types.llms.openai import Batch

    return Batch(
        id="b",
        completion_window="24h",
        created_at=1,
        endpoint="/v1/chat/completions",
        input_file_id="f",
        object="batch",
        status="completed",
        output_file_id=output_file_id,
    )


def _vertex_openai_row(custom_id, model, prompt_tokens, completion_tokens):
    return {
        "id": f"batch_req_{custom_id}",
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "request_id": custom_id,
            "body": {
                "id": f"chatcmpl-{custom_id}",
                "object": "chat.completion",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": _usage(prompt_tokens, completion_tokens),
            },
        },
        "error": None,
    }


def _vertex_jsonl(rows):
    return "\n".join(json.dumps(row) for row in rows).encode()


@pytest.mark.asyncio
async def test_output_file_content_vertex_fetches_via_afile_content(monkeypatch):
    import litellm.files.main as files_main

    rows = [_vertex_openai_row("request-1", "gemini-3.6-flash", 10, 5)]
    captured: dict = {}

    async def fake_afile_content(**kw):
        captured.update(kw)
        return type("R", (), {"content": _vertex_jsonl(rows)})()

    monkeypatch.setattr(files_main, "afile_content", fake_afile_content)

    result = await bu._fetch_batch_output_file_content(
        _batch("gs://litellm-bucket/output/predictions.jsonl"),
        custom_llm_provider="vertex_ai",
        litellm_params={
            "vertex_project": "proj-1",
            "vertex_location": "us-central1",
            "vertex_credentials": "/path/to/creds.json",
            "gcs_bucket_name": "litellm-bucket",
            "model": "vertex_ai/gemini-3.6-flash",
        },
    )

    assert bu._get_file_content_as_dictionary(result) == rows
    assert captured["file_id"] == "gs://litellm-bucket/output/predictions.jsonl"
    assert captured["custom_llm_provider"] == "vertex_ai"
    assert captured["vertex_project"] == "proj-1"
    assert captured["vertex_location"] == "us-central1"
    assert captured["vertex_credentials"] == "/path/to/creds.json"
    assert captured["gcs_bucket_name"] == "litellm-bucket"
    assert "model" not in captured


@pytest.mark.asyncio
async def test_output_file_content_vertex_unified_file_id_extracts_gcs_uri(monkeypatch):
    import base64

    import litellm.files.main as files_main

    captured: dict = {}

    async def fake_afile_content(**kw):
        captured.update(kw)
        return type("R", (), {"content": b'{"a": 1}'})()

    monkeypatch.setattr(files_main, "afile_content", fake_afile_content)
    unified_id = (
        "litellm_proxy:application/jsonl;unified_id,uuid-1;target_model_names,vertex-model;"
        "llm_output_file_id,gs://litellm-bucket/output/predictions.jsonl;llm_output_file_model_id,model-1"
    )
    encoded_id = base64.urlsafe_b64encode(unified_id.encode()).decode().rstrip("=")

    await bu._fetch_batch_output_file_content(_batch(encoded_id), custom_llm_provider="vertex_ai")

    assert captured["file_id"] == "gs://litellm-bucket/output/predictions.jsonl"
    assert captured["custom_llm_provider"] == "vertex_ai"


@pytest.mark.asyncio
async def test_output_file_content_model_encoded_file_id_decoded_to_provider_id(monkeypatch):
    import litellm.files.main as files_main
    from litellm.proxy.openai_files_endpoints.common_utils import encode_file_id_with_model

    captured: dict = {}

    async def fake_afile_content(**kw):
        captured.update(kw)
        return type("R", (), {"content": b'{"a": 1}'})()

    monkeypatch.setattr(files_main, "afile_content", fake_afile_content)
    encoded_id = encode_file_id_with_model("file-Y3FHrMpi7uCkDpY6fgWGeR", "my-batch-model")

    await bu._fetch_batch_output_file_content(_batch(encoded_id), custom_llm_provider="openai")

    assert captured["file_id"] == "file-Y3FHrMpi7uCkDpY6fgWGeR"
    assert captured["custom_llm_provider"] == "openai"


@pytest.mark.asyncio
async def test_output_file_content_raw_openai_file_id_passes_through(monkeypatch):
    import litellm.files.main as files_main

    captured: dict = {}

    async def fake_afile_content(**kw):
        captured.update(kw)
        return type("R", (), {"content": b'{"a": 1}'})()

    monkeypatch.setattr(files_main, "afile_content", fake_afile_content)

    await bu._fetch_batch_output_file_content(_batch("file-abc123"), custom_llm_provider="openai")

    assert captured["file_id"] == "file-abc123"


def _vertex_predictions_row(custom_id, prompt_tokens, completion_tokens):
    return {
        "request": {
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "labels": {"litellm_custom_id": custom_id},
        },
        "status": "",
        "response": {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": "ok"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": prompt_tokens,
                "candidatesTokenCount": completion_tokens,
                "totalTokenCount": prompt_tokens + completion_tokens,
            },
            "modelVersion": "gemini-3.6-flash",
        },
        "processed_time": "2026-07-30T00:00:00.000000+00:00",
    }


@pytest.fixture
def respx_interceptable_httpx_client(monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield
    litellm.in_memory_llm_clients_cache.flush_cache()


@pytest.mark.asyncio
@respx.mock
async def test_output_file_content_vertex_managed_uri_accepted_by_real_validation(respx_interceptable_httpx_client):
    managed_output_uri = (
        "gs://litellm-bucket/litellm-vertex-files/publishers/google/models/"
        "gemini-3.6-flash/abc-123/prediction-model/predictions.jsonl"
    )
    rows = [
        _vertex_predictions_row("request-1", 10, 5),
        _vertex_predictions_row("request-2", 20, 10),
    ]
    route = respx.get(url__regex=r"https://storage\.googleapis\.com/storage/v1/b/litellm-bucket/o/.*").mock(
        return_value=httpx.Response(200, content=_vertex_jsonl(rows))
    )

    file_content = await bu._fetch_batch_output_file_content(
        _batch(managed_output_uri),
        custom_llm_provider="vertex_ai",
        litellm_params={
            "api_key": "test-token",
            "vertex_project": "proj-1",
            "vertex_location": "us-central1",
            "gcs_bucket_name": "litellm-bucket",
        },
    )
    result = bu._get_file_content_as_dictionary(file_content)

    assert route.call_count == 1
    request = route.calls.last.request
    assert request.url.raw_path == (
        b"/storage/v1/b/litellm-bucket/o/"
        b"litellm-vertex-files%2Fpublishers%2Fgoogle%2Fmodels%2Fgemini-3.6-flash"
        b"%2Fabc-123%2Fprediction-model%2Fpredictions.jsonl?alt=media"
    )
    assert [row["custom_id"] for row in result] == ["request-1", "request-2"]
    assert all(row["response"]["status_code"] == 200 for row in result)
    assert all(row["response"]["body"]["model"] == "gemini-3.6-flash" for row in result)
    assert [row["response"]["body"]["usage"]["prompt_tokens"] for row in result] == [10, 20]
    assert [row["response"]["body"]["usage"]["completion_tokens"] for row in result] == [5, 10]


@pytest.mark.asyncio
@respx.mock
async def test_output_file_content_vertex_foreign_bucket_rejected_by_real_validation():
    with pytest.raises(Exception, match="does not match the configured storage bucket"):
        await bu._fetch_batch_output_file_content(
            _batch("gs://attacker-bucket/litellm-vertex-files/x/predictions.jsonl"),
            custom_llm_provider="vertex_ai",
            litellm_params={
                "api_key": "test-token",
                "vertex_project": "proj-1",
                "vertex_location": "us-central1",
                "gcs_bucket_name": "litellm-bucket",
            },
        )

    assert respx.mock.calls.call_count == 0


@pytest.mark.asyncio
async def test_handle_completed_vertex_batch_computes_cost_usage_and_models(monkeypatch):
    import litellm.files.main as files_main

    rows = [
        _vertex_openai_row("request-1", "gemini-3.6-flash", 10, 5),
        _vertex_openai_row("request-2", "gemini-3.6-flash", 20, 10),
    ]

    async def fake_afile_content(**kw):
        return type("R", (), {"content": _vertex_jsonl(rows)})()

    monkeypatch.setattr(files_main, "afile_content", fake_afile_content)

    cost, usage, models = await bu._handle_completed_batch(
        _batch("gs://litellm-bucket/output/predictions.jsonl"),
        custom_llm_provider="vertex_ai",
        litellm_params={"vertex_project": "proj-1", "vertex_location": "us-central1"},
    )

    pricing = litellm.model_cost["vertex_ai/gemini-3.6-flash"]
    batch_input = pricing["input_cost_per_token_batches"]
    batch_output = pricing["output_cost_per_token_batches"]

    assert batch_input < pricing["input_cost_per_token"]
    assert batch_output < pricing["output_cost_per_token"]
    assert cost > 0
    assert cost == pytest.approx(30 * batch_input + 15 * batch_output)
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (30, 15, 45)
    assert models == ["gemini-3.6-flash", "gemini-3.6-flash"]


@pytest.mark.asyncio
async def test_output_file_content_no_output_file_id_raises():
    with pytest.raises(ValueError, match="Output file id is None"):
        await bu._fetch_batch_output_file_content(_batch(None), custom_llm_provider="openai")


@pytest.mark.asyncio
async def test_output_file_content_fetches_and_parses(monkeypatch):
    import litellm.files.main as files_main
    import litellm.proxy.openai_files_endpoints.common_utils as cu

    captured: dict = {}

    async def fake_afile_content(**kw):
        captured.update(kw)
        return type("R", (), {"content": b'{"a": 1}\n{"b": 2}'})()

    monkeypatch.setattr(files_main, "afile_content", fake_afile_content)
    monkeypatch.setattr(cu, "_is_base64_encoded_unified_file_id", lambda fid: False)

    result = await bu._fetch_batch_output_file_content(
        _batch("file-out"),
        custom_llm_provider="azure",
        litellm_params={"api_key": "sk-az", "api_base": "https://az", "model": "x"},
    )

    assert result == b'{"a": 1}\n{"b": 2}'
    # afile_content received the file id + extracted credentials (not "model").
    assert captured["file_id"] == "file-out"
    assert captured["custom_llm_provider"] == "azure"
    assert captured["api_key"] == "sk-az"
    assert captured["api_base"] == "https://az"
    assert "model" not in captured


@pytest.mark.asyncio
async def test_output_file_content_unified_file_id_extraction(monkeypatch):
    # a base64 unified id carries the real provider file id inside
    # "llm_output_file_id,<FID>;" - it must be unwrapped before the fetch.
    import litellm.files.main as files_main
    import litellm.proxy.openai_files_endpoints.common_utils as cu

    captured: dict = {}

    async def fake_afile_content(**kw):
        captured.update(kw)
        return type("R", (), {"content": b'{"a": 1}'})()

    monkeypatch.setattr(files_main, "afile_content", fake_afile_content)
    monkeypatch.setattr(
        cu,
        "_is_base64_encoded_unified_file_id",
        lambda fid: "litellm_proxy;llm_output_file_id,real-file-99;rest",
    )

    await bu._fetch_batch_output_file_content(_batch("encoded-blob"), custom_llm_provider="openai")

    assert captured["file_id"] == "real-file-99"


# =========================================================================== #
# _handle_completed_batch  (async orchestrator: fetch -> single-pass aggregate)
# =========================================================================== #


@pytest.mark.asyncio
async def test_handle_completed_batch_orchestration(monkeypatch):
    rows = [_success_row(model="gpt-4o", usage=_usage(10, 5))]

    async def fake_fetch(batch, custom_llm_provider, litellm_params=None):
        return _vertex_jsonl(rows)

    monkeypatch.setattr(bu, "_fetch_batch_output_file_content", fake_fetch)
    monkeypatch.setattr(litellm, "completion_cost", lambda **kw: 3.3)

    cost, usage, models = await bu._handle_completed_batch(_batch("of"), custom_llm_provider="openai")

    assert cost == 3.3
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (10, 5, 15)
    assert models == ["gpt-4o"]


@pytest.mark.asyncio
async def test_handle_completed_batch_no_output_file_is_zero(monkeypatch):
    """
    Regression: an all-error batch completes with output_file_id=None (results go
    to a separate error_file_id). _handle_completed_batch must report an empty
    result set - zero cost, zero usage, no models - instead of letting the file
    fetch raise "Output file id is None" on every aretrieve_batch logging poll.
    """
    # The output-file fetch must not even be attempted when there is no output file.
    async def _must_not_fetch(*args, **kwargs):
        pytest.fail("_fetch_batch_output_file_content should not be called")

    monkeypatch.setattr(bu, "_fetch_batch_output_file_content", _must_not_fetch)

    cost, usage, models = await bu._handle_completed_batch(_batch(None), custom_llm_provider="openai")

    assert cost == 0.0
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (0, 0, 0)
    assert models == []


@pytest.mark.asyncio
async def test_handle_completed_batch_vertex_disable_transform_path(monkeypatch):
    raw_rows = [{"response": {"usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 2}}}]

    async def fake_fetch(batch, custom_llm_provider, litellm_params=None):
        return _vertex_jsonl(raw_rows)

    monkeypatch.setattr(bu, "_fetch_batch_output_file_content", fake_fetch)
    monkeypatch.setattr(litellm, "disable_vertex_batch_output_transformation", True, raising=False)
    seen: dict = {}

    def fake_vertex_calc(content, model):
        seen["content"] = content
        seen["model"] = model
        return 7.7, Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3)

    monkeypatch.setattr(bu, "calculate_vertex_ai_batch_cost_and_usage", fake_vertex_calc)

    cost, usage, models = await bu._handle_completed_batch(
        _batch("gs://litellm-bucket/output/predictions.jsonl"),
        custom_llm_provider="vertex_ai",
        model_name="gemini-x",
    )

    assert cost == 7.7
    assert usage.total_tokens == 3
    assert models == ["gemini-x"]
    assert seen["content"] == raw_rows
    assert seen["model"] == "gemini-x"


def _anthropic_usage(input_tokens, output_tokens, cache_creation=0, cache_read=0):
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_input_tokens": cache_creation,
        "cache_read_input_tokens": cache_read,
    }


def _anthropic_succeeded_row(model="claude-sonnet-4-5-20250929", usage=None):
    return {
        "custom_id": "req-1",
        "result": {
            "type": "succeeded",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": usage or _anthropic_usage(10, 5),
            },
        },
    }


def _anthropic_errored_row():
    return {
        "custom_id": "req-2",
        "result": {
            "type": "errored",
            "error": {"type": "invalid_request_error", "message": "bad request"},
        },
    }


_ANTHROPIC_MODEL_INFO = {
    "input_cost_per_token": 3e-6,
    "output_cost_per_token": 15e-6,
    "cache_read_input_token_cost": 3e-7,
    "cache_creation_input_token_cost": 3.75e-6,
}


@pytest.mark.parametrize(
    "row,expected",
    [
        (_anthropic_succeeded_row(), True),
        (_anthropic_errored_row(), False),
        ({"custom_id": "x", "result": {"type": "canceled"}}, False),
        ({"custom_id": "x", "result": {"type": "expired"}}, False),
        ({"custom_id": "x"}, False),
        ({"custom_id": "x", "result": None}, False),
    ],
)
def test_anthropic_result_line_success_check(row, expected):
    """
    LIT-4008 regression: anthropic batch results JSONL lines are not
    OpenAI-shaped; success is result.type == "succeeded", not
    response.status_code == 200. Pre-fix every anthropic line parsed as
    unsuccessful, so completed batches were billed $0 forever.
    """
    assert bu._batch_response_was_successful(row, custom_llm_provider="anthropic") is expected


def test_anthropic_response_body_is_result_message():
    row = _anthropic_succeeded_row(model="claude-sonnet-4-5-20250929")
    body = bu._get_response_from_batch_job_output_file(row, custom_llm_provider="anthropic")
    assert body["model"] == "claude-sonnet-4-5-20250929"
    assert body["usage"] == _anthropic_usage(10, 5)


def test_anthropic_usage_conversion_includes_cache_tokens():
    body = {"model": "claude-sonnet-4-5-20250929", "usage": _anthropic_usage(1000, 200, cache_creation=2000, cache_read=8000)}
    usage = bu._get_batch_job_usage_from_response_body(body, custom_llm_provider="anthropic")
    assert usage.prompt_tokens == 11000
    assert usage.completion_tokens == 200
    assert usage.total_tokens == 11200
    assert usage.prompt_tokens_details.cached_tokens == 8000
    assert usage.prompt_tokens_details.cache_creation_tokens == 2000


def test_bedrock_model_output_line_success_check():
    row = {
        "recordId": "1",
        "modelOutput": {"model": "claude-sonnet-4-6", "usage": {"input_tokens": 13, "output_tokens": 5}},
    }
    assert bu._batch_response_was_successful(row, custom_llm_provider="bedrock") is True
    assert bu._get_response_from_batch_job_output_file(row, custom_llm_provider="bedrock")["model"] == "claude-sonnet-4-6"


def test_bedrock_cost_uses_deployment_model_name():
    row = {
        "recordId": "1",
        "modelOutput": {"model": "claude-sonnet-4-6", "usage": {"input_tokens": 13, "output_tokens": 5}},
    }
    cost, _, models = bu._aggregate_batch_cost_usage_models(
        entries=[row],
        custom_llm_provider="bedrock",
        model_name="us.anthropic.claude-sonnet-4-6",
        model_info={},
    )
    assert cost > 0
    assert models == ["us.anthropic.claude-sonnet-4-6"]


def test_anthropic_total_usage_sums_succeeded_only(monkeypatch):
    import litellm.cost_calculator as cc

    monkeypatch.setattr(cc, "batch_cost_calculator", lambda **kw: (0.0, 0.0))
    rows = [
        _anthropic_succeeded_row(usage=_anthropic_usage(10, 5)),
        _anthropic_errored_row(),
        _anthropic_succeeded_row(usage=_anthropic_usage(20, 10, cache_read=100)),
    ]
    _, usage, _ = bu._aggregate_batch_cost_usage_models(entries=rows, custom_llm_provider="anthropic")
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (130, 15, 145)


def test_anthropic_total_usage_aggregates_cache_token_details(monkeypatch):
    import litellm.cost_calculator as cc

    monkeypatch.setattr(cc, "batch_cost_calculator", lambda **kw: (0.0, 0.0))
    rows = [
        _anthropic_succeeded_row(usage=_anthropic_usage(1000, 200, cache_creation=2000, cache_read=8000)),
        _anthropic_errored_row(),
        _anthropic_succeeded_row(usage=_anthropic_usage(50, 20, cache_creation=300, cache_read=700)),
    ]
    _, usage, _ = bu._aggregate_batch_cost_usage_models(entries=rows, custom_llm_provider="anthropic")
    assert usage.prompt_tokens_details.cached_tokens == 8700
    assert usage.prompt_tokens_details.cache_creation_tokens == 2300
    assert usage.cache_read_input_tokens == 8700
    assert usage.cache_creation_input_tokens == 2300


def test_total_usage_without_cache_tokens_has_no_prompt_details(monkeypatch):
    monkeypatch.setattr(litellm, "completion_cost", lambda **kw: 0.0)
    rows = [
        {
            "custom_id": "req-1",
            "response": {"status_code": 200, "body": {"model": "gpt-5.2", "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}},
        }
    ]
    _, usage, _ = bu._aggregate_batch_cost_usage_models(entries=rows, custom_llm_provider="openai")
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (10, 5, 15)
    assert usage.prompt_tokens_details is None


def test_anthropic_cost_applies_batch_discount_and_cache_pricing():
    """Anthropic batches bill at 50% of the regular rate for base input,
    cache reads, cache writes, and output tokens alike."""
    rows = [
        _anthropic_succeeded_row(usage=_anthropic_usage(1000, 200, cache_creation=2000, cache_read=8000)),
        _anthropic_errored_row(),
    ]

    total, _, _ = bu._aggregate_batch_cost_usage_models(
        entries=rows,
        custom_llm_provider="anthropic",
        model_info=_ANTHROPIC_MODEL_INFO,  # type: ignore[arg-type]
    )

    expected_half_price = (1000 * 3e-6 + 8000 * 3e-7 + 2000 * 3.75e-6 + 200 * 15e-6) / 2
    assert total == pytest.approx(expected_half_price)


def test_anthropic_cost_without_model_info_uses_batch_cost_calculator(monkeypatch):
    import litellm.cost_calculator as cc

    seen = []

    def _fake_batch_cost_calculator(**kw):
        seen.append(kw)
        return (0.1, 0.2)

    monkeypatch.setattr(cc, "batch_cost_calculator", _fake_batch_cost_calculator)
    monkeypatch.setattr(
        litellm,
        "completion_cost",
        lambda **kw: pytest.fail("anthropic rows must not go through completion_cost"),
    )

    total, _, _ = bu._aggregate_batch_cost_usage_models(
        entries=[_anthropic_succeeded_row()], custom_llm_provider="anthropic"
    )

    assert total == pytest.approx(0.3)
    assert seen[0]["model"] == "claude-sonnet-4-5-20250929"
    assert seen[0]["custom_llm_provider"] == "anthropic"
    assert seen[0]["usage"].prompt_tokens == 10


def test_anthropic_batch_models_collected_from_succeeded_rows(monkeypatch):
    import litellm.cost_calculator as cc

    monkeypatch.setattr(cc, "batch_cost_calculator", lambda **kw: (0.0, 0.0))
    rows = [
        _anthropic_succeeded_row(model="claude-sonnet-4-5-20250929"),
        _anthropic_errored_row(),
    ]
    _, _, models = bu._aggregate_batch_cost_usage_models(entries=rows, custom_llm_provider="anthropic")
    assert models == ["claude-sonnet-4-5-20250929"]


@pytest.mark.asyncio
async def test_calculate_batch_cost_and_usage_anthropic_end_to_end():
    rows = [
        _anthropic_succeeded_row(usage=_anthropic_usage(1000, 200, cache_creation=2000, cache_read=8000)),
        _anthropic_errored_row(),
    ]

    cost, usage, models = await bu.calculate_batch_cost_and_usage(
        file_content_dictionary=rows,
        custom_llm_provider="anthropic",
        model_name="claude-sonnet-4-5",
        model_info=_ANTHROPIC_MODEL_INFO,  # type: ignore[arg-type]
    )

    assert cost == pytest.approx(1000 * 3e-6 / 2 + 8000 * 3e-7 / 2 + 2000 * 3.75e-6 / 2 + 200 * 15e-6 / 2)
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (11000, 200, 11200)
    assert models == ["claude-sonnet-4-5"]


def test_extract_credentials_forwards_the_trusted_model_credential_snapshot():
    """Bedrock resolves a batch's output bucket only from the immutable server-side
    snapshot, never from a request param, so cost accounting on the retrieve path cannot
    read the output file unless this key is forwarded. Without it the accounting raises
    "S3 bucket_name is required" for a bucket the deployment has configured, and the
    batch's cost is never recorded."""
    snapshot = MappingProxyType({"s3_bucket_name": "configured-bucket", "aws_region_name": "us-east-1"})

    credentials = bu._extract_file_access_credentials({"_litellm_internal_model_credentials": snapshot})

    assert credentials["_litellm_internal_model_credentials"] is snapshot


def test_extract_credentials_forwards_the_deployment_aws_credentials():
    """The retrieve path's logging object carries the deployment's AWS keys in its
    litellm_params, and the S3 read of the output file signs with whatever afile_content
    receives. Dropping them here sent the read to the ambient credential chain, so a
    deployment whose only AWS credentials live in its litellm_params never recorded
    batch cost on retrieve even once the bucket resolved."""
    params = {
        "aws_access_key_id": "AKIA-deployment",
        "aws_secret_access_key": "secret-deployment",
        "aws_session_token": "token-deployment",
        "aws_region_name": "us-west-2",
        "aws_role_name": "arn:aws:iam::123456789012:role/batch-reader",
        "model": "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
    }

    credentials = bu._extract_file_access_credentials(params)

    assert credentials == {key: value for key, value in params.items() if key != "model"}


@pytest.mark.asyncio
async def test_output_file_content_bedrock_reads_with_deployment_aws_credentials(monkeypatch):
    import litellm.files.main as files_main

    captured: dict = {}

    async def fake_afile_content(**kw):
        captured.update(kw)
        return type("R", (), {"content": b""})()

    monkeypatch.setattr(files_main, "afile_content", fake_afile_content)
    snapshot = MappingProxyType({"s3_bucket_name": "configured-bucket", "aws_region_name": "us-west-2"})

    await bu._fetch_batch_output_file_content(
        _batch("s3://configured-bucket/litellm-batch-outputs/job-1/out.jsonl.out"),
        custom_llm_provider="bedrock",
        litellm_params={
            "aws_access_key_id": "AKIA-deployment",
            "aws_secret_access_key": "secret-deployment",
            "aws_session_token": "token-deployment",
            "aws_region_name": "us-west-2",
            "_litellm_internal_model_credentials": snapshot,
            "model": "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        },
    )

    assert captured["file_id"] == "s3://configured-bucket/litellm-batch-outputs/job-1/out.jsonl.out"
    assert captured["custom_llm_provider"] == "bedrock"
    assert captured["aws_access_key_id"] == "AKIA-deployment"
    assert captured["aws_secret_access_key"] == "secret-deployment"
    assert captured["aws_session_token"] == "token-deployment"
    assert captured["aws_region_name"] == "us-west-2"
    assert captured["_litellm_internal_model_credentials"] is snapshot
    assert "model" not in captured


# =========================================================================== #
# _handle_completed_batch threads the deployment's model identity + pricing
# =========================================================================== #


def _bedrock_row(model: str, input_tokens: int, output_tokens: int) -> dict[str, object]:
    return {
        "modelInput": {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]},
        "modelOutput": {
            "model": model,
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        },
        "recordId": "r",
    }


@pytest.mark.asyncio
async def test_handle_completed_bedrock_batch_prices_from_deployment_model(monkeypatch) -> None:
    """A bedrock batch must price from the deployment model, not the response model."""
    rows = [_bedrock_row("claude-sonnet-4-6", 18, 10)] * 100

    async def fake_fetch(batch: object, custom_llm_provider: str, litellm_params: dict | None = None) -> bytes:
        return _vertex_jsonl(rows)

    monkeypatch.setattr(bu, "_fetch_batch_output_file_content", fake_fetch)

    cost, usage, _ = await bu._handle_completed_batch(
        _batch("of"),
        custom_llm_provider="bedrock",
        model_name="bedrock/global.anthropic.claude-sonnet-4-6",
    )

    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (1800, 1000, 2800)
    # 3e-06 / 1.5e-05 on-demand, halved for batch.
    assert cost == pytest.approx(1800 * 3e-06 / 2 + 1000 * 1.5e-05 / 2)

    # The response model alone cannot price a bedrock batch: this is the $0 bug.
    zero_cost, zero_usage, _ = await bu._handle_completed_batch(
        _batch("of"),
        custom_llm_provider="bedrock",
        model_name=None,
    )
    assert zero_cost == 0.0
    assert zero_usage.total_tokens == 2800


@pytest.mark.asyncio
async def test_handle_completed_batch_honors_deployment_pricing(monkeypatch) -> None:
    """A deployment's configured rates must win over the global cost map."""
    rows = [_success_row(model="gemini-2.5-flash", usage=_usage(60, 75))]

    async def fake_fetch(batch: object, custom_llm_provider: str, litellm_params: dict | None = None) -> bytes:
        return _vertex_jsonl(rows)

    monkeypatch.setattr(bu, "_fetch_batch_output_file_content", fake_fetch)

    free_cost, _, _ = await bu._handle_completed_batch(
        _batch("of"),
        custom_llm_provider="vertex_ai",
        model_name="vertex_ai/gemini-2.5-flash",
        model_info={
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "input_cost_per_token_batches": 0.0,
            "output_cost_per_token_batches": 0.0,
        },
    )
    assert free_cost == 0.0

    billed_cost, _, _ = await bu._handle_completed_batch(
        _batch("of"),
        custom_llm_provider="vertex_ai",
        model_name="vertex_ai/gemini-2.5-flash",
        model_info=None,
    )
    assert billed_cost > 0.0


# =========================================================================== #
# _get_batch_job_usage_from_response_body: bedrock usage shapes
# =========================================================================== #


def test_bedrock_converse_shaped_batch_usage_is_parsed():
    body = {"model": "us.amazon.nova-lite-v1:0", "usage": {"inputTokens": 2202, "outputTokens": 540, "totalTokens": 2742}}
    usage = bu._get_batch_job_usage_from_response_body(body, custom_llm_provider="bedrock")
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (2202, 540, 2742)


def test_bedrock_converse_batch_usage_totals_default_when_absent():
    body = {"model": "us.amazon.nova-lite-v1:0", "usage": {"inputTokens": 10, "outputTokens": 4}}
    usage = bu._get_batch_job_usage_from_response_body(body, custom_llm_provider="bedrock")
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (10, 4, 14)


def test_bedrock_converse_batch_usage_includes_cache_tokens():
    body = {
        "model": "us.amazon.nova-lite-v1:0",
        "usage": {
            "inputTokens": 100,
            "outputTokens": 20,
            "totalTokens": 120,
            "cacheReadInputTokens": 800,
            "cacheWriteInputTokens": 200,
        },
    }
    usage = bu._get_batch_job_usage_from_response_body(body, custom_llm_provider="bedrock")
    assert usage.prompt_tokens == 1100
    assert usage.completion_tokens == 20
    assert usage.prompt_tokens_details.cached_tokens == 800
    assert usage.prompt_tokens_details.cache_creation_tokens == 200


def test_bedrock_anthropic_shaped_batch_usage_still_parsed():
    """Anthropic-shaped bedrock output (what an Anthropic model's batch emits) must not regress."""
    body = {"model": "claude-sonnet-4-6", "usage": {"input_tokens": 18, "output_tokens": 10}}
    usage = bu._get_batch_job_usage_from_response_body(body, custom_llm_provider="bedrock")
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (18, 10, 28)


def test_unparsable_bedrock_batch_usage_warns(caplog):
    """An unrecognized usage shape must be visible, not a silent $0."""
    body = {"model": "amazon.titan-text-lite-v1", "usage": {"inputTextTokenCount": 42}}
    with caplog.at_level(logging.WARNING):
        usage = bu._get_batch_job_usage_from_response_body(body, custom_llm_provider="bedrock")
    assert usage.total_tokens == 0
    assert "does not understand" in caplog.text
    assert "inputTextTokenCount" in caplog.text
