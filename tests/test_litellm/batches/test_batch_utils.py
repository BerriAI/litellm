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

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

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


def test_parse_jsonl_malformed_raises():
    with pytest.raises(Exception):
        bu._get_file_content_as_dictionary(b"not valid json")


# =========================================================================== #
# _iter_batch_input_lines / _iter_batch_input_entries  (JSONL parsing)
# =========================================================================== #


def test_iter_input_lines_skips_blank_and_strips():
    content = b'{"a":1}\n\n  \n{"b":2}\n'
    assert list(bu._iter_batch_input_lines(content)) == [b'{"a":1}', b'{"b":2}']


def test_iter_input_lines_handles_missing_trailing_newline():
    assert list(bu._iter_batch_input_lines(b'{"a":1}')) == [b'{"a":1}']


def test_iter_input_lines_empty():
    assert list(bu._iter_batch_input_lines(b"")) == []


def test_iter_input_entries_parses_each_row():
    content = b'{"body": {"model": "gpt-4o"}}\n{"body": {"model": "claude-3"}}\n'
    assert list(bu._iter_batch_input_entries(content)) == [
        {"body": {"model": "gpt-4o"}},
        {"body": {"model": "claude-3"}},
    ]


def test_iter_input_entries_raises_on_malformed_line():
    # _iter_batch_input_entries raises on a bad row; callers that must survive
    # bad rows iterate _iter_batch_input_lines and parse per-row instead.
    with pytest.raises(Exception):
        list(bu._iter_batch_input_entries(b'{"ok":1}\nnot-json\n'))


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
# _get_batch_models_from_file_content  (output file)
# =========================================================================== #


def test_output_models_uses_model_name_override():
    # model_name short-circuits: content is ignored entirely.
    assert bu._get_batch_models_from_file_content([_success_row(model="ignored")], model_name="forced-model") == [
        "forced-model"
    ]


def test_output_models_collects_from_successful_only():
    rows = [
        _success_row(model="gpt-4o"),
        _failed_row(model="should-be-skipped"),
        _success_row(model="claude-3"),
    ]
    assert bu._get_batch_models_from_file_content(rows) == ["gpt-4o", "claude-3"]


def test_output_models_skips_successful_without_model():
    rows = [{"response": {"status_code": 200, "body": {}}}]
    assert bu._get_batch_models_from_file_content(rows) == []


# =========================================================================== #
# _extract_file_access_credentials
# =========================================================================== #


def test_extract_credentials_only_known_keys():
    params = {
        "api_key": "sk-1",
        "api_base": "https://b",
        "vertex_project": "proj",
        "model": "gpt-4o",  # not a credential key
        "unrelated": "x",
    }
    assert bu._extract_file_access_credentials(params) == {
        "api_key": "sk-1",
        "api_base": "https://b",
        "vertex_project": "proj",
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
# _get_batch_job_total_usage_from_file_content  (output usage aggregation)
# =========================================================================== #


def test_total_usage_sums_successful_only():
    rows = [
        _success_row(usage=_usage(10, 5)),  # 15
        _failed_row(),  # excluded
        _success_row(usage=_usage(20, 10)),  # 30
    ]
    usage = bu._get_batch_job_total_usage_from_file_content(rows)
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (
        30,
        15,
        45,
    )


def test_total_usage_empty_is_zero():
    usage = bu._get_batch_job_total_usage_from_file_content([])
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (
        0,
        0,
        0,
    )


# =========================================================================== #
# _get_batch_job_cost_from_file_content  (cost maps mocked)
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

    total = bu._get_batch_job_cost_from_file_content(rows, custom_llm_provider="openai")

    assert total == 1.0  # 2 successful * 0.5
    assert len(calls) == 2  # failed row not costed


def test_cost_from_content_model_info_path(monkeypatch):
    # model_info set -> batch_cost_calculator(prompt_cost, completion_cost).
    import litellm.cost_calculator as cc

    monkeypatch.setattr(cc, "batch_cost_calculator", lambda **kw: (0.1, 0.2))
    rows = [
        _success_row(usage=_usage(10, 5)),
        _success_row(usage=_usage(20, 10)),
    ]

    total = bu._get_batch_job_cost_from_file_content(
        rows,
        custom_llm_provider="openai",
        model_info={"input_cost_per_token": 0.0},  # type: ignore[arg-type]  # truthy -> model_info path
    )

    assert total == pytest.approx(0.6)  # 2 * (0.1 + 0.2)


# =========================================================================== #
# _batch_cost_calculator  (dispatch: vertex-disable-transform vs generic)
# =========================================================================== #


def test_batch_cost_calculator_generic_path(monkeypatch):
    monkeypatch.setattr(bu, "_get_batch_job_cost_from_file_content", lambda **kw: 4.2)
    assert bu._batch_cost_calculator([], custom_llm_provider="openai", model_name="gpt-4o") == 4.2


def test_batch_cost_calculator_vertex_disable_transform_path(monkeypatch):
    monkeypatch.setattr(litellm, "disable_vertex_batch_output_transformation", True, raising=False)
    monkeypatch.setattr(
        bu,
        "calculate_vertex_ai_batch_cost_and_usage",
        lambda content, model: (9.9, Usage()),
    )
    # generic path must NOT be taken
    monkeypatch.setattr(
        bu,
        "_get_batch_job_cost_from_file_content",
        lambda **kw: pytest.fail("generic path should not run"),
    )

    cost = bu._batch_cost_calculator([], custom_llm_provider="vertex_ai", model_name="gemini-2.0-flash-001")
    assert cost == 9.9


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
    monkeypatch.setattr(bu, "_batch_cost_calculator", lambda **kw: 2.5)
    monkeypatch.setattr(
        bu,
        "_get_batch_job_total_usage_from_file_content",
        lambda **kw: Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )

    cost, usage, models = await bu.calculate_batch_cost_and_usage(
        file_content_dictionary=rows, custom_llm_provider="openai"
    )

    assert cost == 2.5
    assert usage.total_tokens == 15
    assert models == ["gpt-4o"]  # real _get_batch_models_from_file_content


# =========================================================================== #
# _get_batch_output_file_content_as_dictionary  (file fetch + credential merge)
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


@pytest.mark.asyncio
async def test_output_file_content_vertex_raises():
    with pytest.raises(ValueError, match="Vertex AI does not support"):
        await bu._get_batch_output_file_content_as_dictionary(_batch("of"), custom_llm_provider="vertex_ai")


@pytest.mark.asyncio
async def test_output_file_content_no_output_file_id_raises():
    with pytest.raises(ValueError, match="Output file id is None"):
        await bu._get_batch_output_file_content_as_dictionary(_batch(None), custom_llm_provider="openai")


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

    result = await bu._get_batch_output_file_content_as_dictionary(
        _batch("file-out"),
        custom_llm_provider="azure",
        litellm_params={"api_key": "sk-az", "api_base": "https://az", "model": "x"},
    )

    assert result == [{"a": 1}, {"b": 2}]
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

    await bu._get_batch_output_file_content_as_dictionary(_batch("encoded-blob"), custom_llm_provider="openai")

    assert captured["file_id"] == "real-file-99"


# =========================================================================== #
# _handle_completed_batch  (async orchestrator: fetch -> cost/usage/models)
# =========================================================================== #


@pytest.mark.asyncio
async def test_handle_completed_batch_orchestration(monkeypatch):
    rows = [_success_row(model="gpt-4o", usage=_usage(10, 5))]

    async def fake_get_content(batch, custom_llm_provider, litellm_params=None):
        return rows

    monkeypatch.setattr(bu, "_get_batch_output_file_content_as_dictionary", fake_get_content)
    monkeypatch.setattr(bu, "_batch_cost_calculator", lambda **kw: 3.3)
    monkeypatch.setattr(
        bu,
        "_get_batch_job_total_usage_from_file_content",
        lambda **kw: Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )

    cost, usage, models = await bu._handle_completed_batch(_batch("of"), custom_llm_provider="openai")

    assert cost == 3.3
    assert usage.total_tokens == 15
    assert models == ["gpt-4o"]


@pytest.mark.asyncio
async def test_handle_completed_batch_all_error_batch_no_output_file(monkeypatch):
    """All-error batches complete with output_file_id=None; logging must not crash."""
    called = False

    async def fake_get_content(*args, **kwargs):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(bu, "_get_batch_output_file_content_as_dictionary", fake_get_content)

    cost, usage, models = await bu._handle_completed_batch(_batch(None), custom_llm_provider="openai")

    assert cost == 0.0
    assert usage.total_tokens == 0
    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert models == []
    assert called is False


# =========================================================================== #
# Remaining branch: vertex usage disable-transform path.
#
# NOTE: the error path of _get_batch_job_cost_from_file_content (its `raise e`)
# is intentionally NOT tested: the preceding line logs via
# `verbose_logger.error("...", e)`, which passes the exception as a logging
# format-arg with no placeholder and itself raises TypeError under
# logging.raiseExceptions, masking the original error. Asserting that masked
# behavior would lock a source bug; left uncovered on purpose.
# =========================================================================== #


def test_total_usage_vertex_disable_transform_path(monkeypatch):
    monkeypatch.setattr(litellm, "disable_vertex_batch_output_transformation", True, raising=False)
    monkeypatch.setattr(
        bu,
        "calculate_vertex_ai_batch_cost_and_usage",
        lambda content, model: (
            0.0,
            Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        ),
    )

    usage = bu._get_batch_job_total_usage_from_file_content([], custom_llm_provider="vertex_ai", model_name="gemini-x")
    assert usage.total_tokens == 3


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
    body = {
        "model": "claude-sonnet-4-5-20250929",
        "usage": _anthropic_usage(1000, 200, cache_creation=2000, cache_read=8000),
    }
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
    assert (
        bu._get_response_from_batch_job_output_file(row, custom_llm_provider="bedrock")["model"] == "claude-sonnet-4-6"
    )


def test_bedrock_cost_uses_deployment_model_name():
    row = {
        "recordId": "1",
        "modelOutput": {"model": "claude-sonnet-4-6", "usage": {"input_tokens": 13, "output_tokens": 5}},
    }
    cost = bu._get_batch_job_cost_from_file_content(
        file_content_dictionary=[row],
        custom_llm_provider="bedrock",
        model_name="us.anthropic.claude-sonnet-4-6",
        model_info={},
    )
    assert cost > 0


def test_anthropic_total_usage_sums_succeeded_only():
    rows = [
        _anthropic_succeeded_row(usage=_anthropic_usage(10, 5)),
        _anthropic_errored_row(),
        _anthropic_succeeded_row(usage=_anthropic_usage(20, 10, cache_read=100)),
    ]
    usage = bu._get_batch_job_total_usage_from_file_content(rows, custom_llm_provider="anthropic")
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (130, 15, 145)


def test_anthropic_total_usage_aggregates_cache_token_details():
    rows = [
        _anthropic_succeeded_row(usage=_anthropic_usage(1000, 200, cache_creation=2000, cache_read=8000)),
        _anthropic_errored_row(),
        _anthropic_succeeded_row(usage=_anthropic_usage(50, 20, cache_creation=300, cache_read=700)),
    ]
    usage = bu._get_batch_job_total_usage_from_file_content(rows, custom_llm_provider="anthropic")
    assert usage.prompt_tokens_details.cached_tokens == 8700
    assert usage.prompt_tokens_details.cache_creation_tokens == 2300
    assert usage.cache_read_input_tokens == 8700
    assert usage.cache_creation_input_tokens == 2300


def test_total_usage_without_cache_tokens_has_no_prompt_details():
    rows = [
        {
            "custom_id": "req-1",
            "response": {
                "status_code": 200,
                "body": {
                    "model": "gpt-5.2",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                },
            },
        }
    ]
    usage = bu._get_batch_job_total_usage_from_file_content(rows, custom_llm_provider="openai")
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (10, 5, 15)
    assert usage.prompt_tokens_details is None


def test_anthropic_cost_applies_batch_discount_and_cache_pricing():
    """Anthropic batches bill at 50% of the regular rate for base input,
    cache reads, cache writes, and output tokens alike."""
    rows = [
        _anthropic_succeeded_row(usage=_anthropic_usage(1000, 200, cache_creation=2000, cache_read=8000)),
        _anthropic_errored_row(),
    ]

    total = bu._get_batch_job_cost_from_file_content(
        rows,
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

    total = bu._get_batch_job_cost_from_file_content([_anthropic_succeeded_row()], custom_llm_provider="anthropic")

    assert total == pytest.approx(0.3)
    assert seen[0]["model"] == "claude-sonnet-4-5-20250929"
    assert seen[0]["custom_llm_provider"] == "anthropic"
    assert seen[0]["usage"].prompt_tokens == 10


def test_anthropic_batch_models_collected_from_succeeded_rows():
    rows = [
        _anthropic_succeeded_row(model="claude-sonnet-4-5-20250929"),
        _anthropic_errored_row(),
    ]
    assert bu._get_batch_models_from_file_content(rows, None, "anthropic") == ["claude-sonnet-4-5-20250929"]


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
