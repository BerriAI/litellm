"""Tests for input-token counting shared across the reservation path's models."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import wave
from types import MappingProxyType
from typing import Final

import pytest

import litellm
from litellm.proxy.spend_tracking.input_tokens import (
    TOKENIZE_OFF_EVENT_LOOP_MIN_CHARS,
    count_input_tokens,
    count_input_tokens_for_model,
    models_counted_for,
)
from litellm.router import Router

CL100K_MODEL: Final = "gpt-4"


@pytest.mark.asyncio
async def test_large_input_is_still_counted() -> None:
    request_body: Final = {
        "model": CL100K_MODEL,
        "messages": [{"role": "user", "content": "x" * (TOKENIZE_OFF_EVENT_LOOP_MIN_CHARS + 1)}],
    }

    counts: Final = await count_input_tokens(request_body=request_body, raw_body=None, models=(CL100K_MODEL,))

    assert counts[CL100K_MODEL] == count_input_tokens_for_model(request_body=request_body, model=CL100K_MODEL)
    assert isinstance(counts, MappingProxyType)


def _one_page_pdf_base64() -> str:
    stream: Final = b"BT /F1 12 Tf 72 720 Td (Revenue grew eleven percent while churn fell.) Tj ET"
    bodies: Final = (
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Count 1 /Kids [ 4 0 R ] >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> "
        b"/Contents 5 0 R >>",
        f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream",
    )
    objects: Final = tuple(f"{n} 0 obj\n".encode() + body + b"\nendobj\n" for n, body in enumerate(bodies, start=1))
    header: Final = b"%PDF-1.4\n"
    offsets: Final = tuple(len(header) + sum(len(obj) for obj in objects[:index]) for index in range(len(objects)))
    xref: Final = f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode() + b"".join(
        f"{offset:010d} 00000 n \n".encode() for offset in offsets
    )
    trailer: Final = (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{offsets[-1] + len(objects[-1])}\n%%EOF\n"
    ).encode()
    return base64.b64encode(header + b"".join(objects) + xref + trailer).decode()


def _one_page_pdf_message() -> list[dict[str, object]]:
    pdf: Final = _one_page_pdf_base64()
    return [
        {
            "role": "user",
            "content": [
                {"type": "file", "file": {"filename": "report.pdf", "file_data": f"data:application/pdf;base64,{pdf}"}},
                {"type": "text", "text": "Summarize the report."},
            ],
        }
    ]


def _router(*deployments: tuple[str, str]) -> Router:
    return Router(
        model_list=[
            {"model_name": name, "litellm_params": {"model": model, "api_key": "sk-test", "api_base": "https://x"}}
            for name, model in deployments
        ]
    )


@pytest.mark.parametrize(
    ("group", "deployment_model", "counted_with"),
    [
        ("gpt-4o", "anthropic/claude-haiku-5-5", "claude-haiku-5-5"),
        ("pdf-reports", "azure/gpt-5.6-sol", "azure/gpt-5.6-sol"),
        ("pdf-reports", "azure/responses/gpt-5.6-sol", "azure/responses/gpt-5.6-sol"),
        ("pdf-reports", "openrouter/openai/gpt-5.6-sol", "openrouter/openai/gpt-5.6-sol"),
        ("pdf-reports", "openai/gpt-5.6-sol", "openai/gpt-5.6-sol"),
        ("pdf-reports", "gpt-5.6-sol", "openai/gpt-5.6-sol"),
    ],
    ids=[
        "gpt-named-group-served-by-claude",
        "custom-group-served-by-azure-gpt",
        "custom-group-served-by-azure-responses-gpt",
        "custom-group-served-by-openrouter-gpt",
        "custom-group-served-by-openai-gpt",
        "custom-group-served-by-unprefixed-openai-gpt",
    ],
)
def test_a_model_group_counts_like_the_deployment_that_serves_it(
    group: str, deployment_model: str, counted_with: str
) -> None:
    request_body: Final = {"model": group, "messages": _one_page_pdf_message()}

    assert models_counted_for(group, _router((group, deployment_model)), None) == (counted_with,)
    assert count_input_tokens_for_model(request_body=request_body, model=counted_with) != count_input_tokens_for_model(
        request_body=request_body, model=group
    )


@pytest.mark.parametrize(
    ("group", "deployment_model"),
    [
        ("my-claude", "anthropic/claude-haiku-5-5"),
        ("my-llama", "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"),
        ("my-cohere", "cohere/command-r-plus"),
        ("gpt-4o", "cohere/command-r-plus"),
    ],
    ids=["claude-alias", "llama-alias", "cohere-alias", "gpt-named-group-served-by-a-model-that-reads-no-pdf"],
)
def test_a_group_whose_deployment_prices_pdf_pages_alike_or_reads_none_counts_as_the_group(
    group: str, deployment_model: str
) -> None:
    assert models_counted_for(group, _router((group, deployment_model)), None) == (group,)


def test_a_custom_named_deployment_counts_with_its_base_model():
    router: Final = Router(
        model_list=[
            {
                "model_name": "reports",
                "litellm_params": {"model": "azure/prod-deploy-1", "api_key": "x", "api_base": "https://x"},
                "model_info": {"base_model": "azure/gpt-5.6-sol"},
            },
            {"model_name": "drafts", "litellm_params": {"model": "azure/prod-deploy-2", "api_key": "x"}},
        ]
    )

    assert models_counted_for("reports", router, None) == ("azure/gpt-5.6-sol",)
    assert models_counted_for("drafts", router, None) == ("drafts",)


def test_a_group_of_differently_priced_deployments_counts_with_each_and_a_name_no_deployment_serves_counts_as_is():
    router: Final = _router(("mixed", "anthropic/claude-haiku-5-5"), ("mixed", "azure/gpt-5.6-sol"))

    assert models_counted_for("mixed", router, None) == ("mixed", "azure/gpt-5.6-sol")
    assert models_counted_for("not-deployed", router, None) == ("not-deployed",)
    assert models_counted_for("gpt-4o", None, None) == ("gpt-4o",)


def test_a_team_request_counts_with_the_deployment_its_team_is_served_by():
    router: Final = Router(
        model_list=[
            {"model_name": "reports", "litellm_params": {"model": "anthropic/claude-haiku-5-5", "api_key": "x"}},
            {
                "model_name": "reports-team-a",
                "litellm_params": {"model": "azure/gpt-5.6-sol", "api_key": "x"},
                "model_info": {"team_id": "team-a", "team_public_model_name": "reports"},
            },
        ]
    )

    assert models_counted_for("reports", router, "team-a") == ("azure/gpt-5.6-sol",)
    assert models_counted_for("reports", router, "team-b") == ("reports",)


@pytest.mark.parametrize("requested", ["reports", "latest-reports"], ids=["routing-group", "alias-of-routing-group"])
def test_a_team_request_to_a_routing_group_counts_with_the_member_deployment_its_team_is_served_by(requested: str):
    router: Final = Router(
        model_list=[
            {
                "model_name": "drafts",
                "litellm_params": {"model": "azure/gpt-5.6-sol", "api_key": "x", "api_base": "https://x"},
            },
            {
                "model_name": "drafts-team-a",
                "litellm_params": {"model": "anthropic/claude-haiku-5-5", "api_key": "x"},
                "model_info": {"team_id": "team-a", "team_public_model_name": "drafts"},
            },
        ],
        routing_groups=[{"group_name": "reports", "models": ["drafts"], "routing_strategy": "simple-shuffle"}],
        model_group_alias={"latest-reports": "reports"},
    )

    assert models_counted_for(requested, router, "team-a") == models_counted_for(
        requested, _router((requested, "anthropic/claude-haiku-5-5")), None
    )
    assert models_counted_for(requested, router, "team-b") == ("azure/gpt-5.6-sol",)


def test_a_team_with_its_own_wildcard_counts_with_it_and_never_with_a_proxy_wide_wildcard():
    router: Final = Router(
        model_list=[
            {"model_name": "*", "litellm_params": {"model": "anthropic/*", "api_key": "x"}},
            {
                "model_name": "team-a-gpt",
                "litellm_params": {"model": "azure/gpt-*", "api_key": "x", "api_base": "https://x"},
                "model_info": {"team_id": "team-a", "team_public_model_name": "gpt-*"},
            },
        ]
    )

    assert models_counted_for("gpt-5.6-sol", router, "team-a") == ("azure/gpt-5.6-sol",)
    assert models_counted_for("gpt-5.6-sol", router, "team-b") == models_counted_for("gpt-5.6-sol", router, None)
    assert "azure/gpt-5.6-sol" not in models_counted_for("gpt-5.6-sol", router, "team-b")


def test_count_input_tokens_for_model_counts_anthropic_auto_tool_choice() -> None:
    model: Final = "claude-sonnet-5-5"
    body_without_tool_choice: Final = {
        "model": model,
        "messages": [{"role": "user", "content": "weather in pune?"}],
        "tools": [
            {
                "name": "get_weather",
                "description": "Get the weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            }
        ],
    }
    body_with_tool_choice: Final = body_without_tool_choice | {"tool_choice": {"type": "auto"}}

    expected: Final = count_input_tokens_for_model(request_body=body_without_tool_choice, model=model)
    actual: Final = count_input_tokens_for_model(request_body=body_with_tool_choice, model=model)

    assert isinstance(expected, int)
    assert actual == expected


_COMPACTION_ITEM: Final = {
    "type": "compaction",
    "id": "cmp_1",
    "encrypted_content": json.dumps({"type": "compaction", "content": "Turn one planned Tokyo."}),
}
_TURN_AFTER_COMPACTION: Final = {"role": "user", "content": "And Osaka?"}


@pytest.mark.parametrize(
    "earlier_item",
    [
        pytest.param({"role": "user", "content": "Plan a Tokyo trip. " * 500}, id="user_turn"),
        pytest.param({"role": ["system"], "content": "Plan a Tokyo trip."}, id="list_role"),
        pytest.param({"role": {"name": "system"}, "content": "Plan a Tokyo trip."}, id="dict_role"),
    ],
)
@pytest.mark.parametrize("model", [CL100K_MODEL, "claude-sonnet-5-5"])
@pytest.mark.asyncio
async def test_history_before_a_compaction_item_is_not_counted(model: str, earlier_item: dict[str, object]) -> None:
    compacted_body: Final = {"model": model, "input": [_COMPACTION_ITEM, _TURN_AFTER_COMPACTION]}
    full_body: Final = {"model": model, "input": [earlier_item, _COMPACTION_ITEM, _TURN_AFTER_COMPACTION]}

    full_counts: Final = await count_input_tokens(
        request_body=full_body, raw_body=json.dumps(full_body).encode(), models=(model,)
    )
    compacted_counts: Final = await count_input_tokens(
        request_body=compacted_body, raw_body=json.dumps(compacted_body).encode(), models=(model,)
    )

    assert full_counts[model] == compacted_counts[model]
    assert count_input_tokens_for_model(request_body=full_body, model=model) == compacted_counts[model]


def _compaction_item_carrying(extra_block_fields: dict[str, str]) -> dict[str, str]:
    return {
        "type": "compaction",
        "id": "cmp_1",
        "encrypted_content": json.dumps(
            {"type": "compaction", "content": "Turn one planned Tokyo.", **extra_block_fields}
        ),
    }


@pytest.mark.parametrize("model", [CL100K_MODEL, "claude-sonnet-5-5"])
def test_a_compaction_item_counts_only_its_summary(model: str) -> None:
    plain_body: Final = {"model": model, "input": [_compaction_item_carrying({}), _TURN_AFTER_COMPACTION]}
    padded_body: Final = {
        "model": model,
        "input": [_compaction_item_carrying({"opaque_state": "x" * 4000}), _TURN_AFTER_COMPACTION],
    }

    assert count_input_tokens_for_model(request_body=padded_body, model=model) == count_input_tokens_for_model(
        request_body=plain_body, model=model
    )


GEMINI_MODEL: Final = "gemini-3.8-flash"


def _wav_base64(seconds: int) -> str:
    buffer: Final = io.BytesIO()
    with wave.open(buffer, "wb") as recording:
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(16_000)
        recording.writeframes(b"\x00\x00" * 16_000 * seconds)
    return base64.b64encode(buffer.getvalue()).decode()


def _gemini_tokens(*contents: dict[str, object], **fields: object) -> int | None:
    return count_input_tokens_for_model(request_body={"contents": list(contents), **fields}, model=GEMINI_MODEL)


def _user(*parts: object) -> dict[str, object]:
    return {"role": "user", "parts": list(parts)}


def test_a_gemini_text_body_counts_its_text_not_the_context_window() -> None:
    text: Final = "Summarize the quarterly report in three bullet points. " * 40

    assert _gemini_tokens(_user({"text": text})) == litellm.token_counter(model=GEMINI_MODEL, text=text)


@pytest.mark.parametrize(
    "contents",
    [
        pytest.param({"role": "user", "parts": {"text": "Name a prime."}}, id="single-content-single-part"),
        pytest.param([{"role": "user", "parts": "Name a prime."}], id="bare-string-parts"),
        pytest.param("Name a prime.", id="bare-string-contents"),
    ],
)
def test_a_gemini_body_in_any_shape_gemini_accepts_counts_its_text(contents: object) -> None:
    assert count_input_tokens_for_model(request_body={"contents": contents}, model=GEMINI_MODEL) == (
        litellm.token_counter(model=GEMINI_MODEL, text="Name a prime.")
    )


def test_every_gemini_turn_counts_whatever_its_role() -> None:
    question: Final = _user({"text": "What is the weather in Paris?"})
    tool_turns: Final = (
        {"role": "model", "parts": [{"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}}]},
        {
            "role": "function",
            "parts": [{"functionResponse": {"name": "get_weather", "response": {"forecast": "sunny " * 200}}}],
        },
    )

    with_tool_turns: Final = _gemini_tokens(question, *tool_turns)
    question_only: Final = _gemini_tokens(question)

    assert with_tool_turns is not None and question_only is not None
    assert with_tool_turns - question_only >= 200


@pytest.mark.parametrize("field", ["thoughtSignature", "thought_signature"])
def test_a_gemini_thought_signature_counts_its_decoded_size_because_gemini_bills_the_thinking_it_carries(
    field: str,
) -> None:
    call: Final = {"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}}
    signed: Final = {"role": "model", "parts": [{**call, field: base64.b64encode(b"s" * 3000).decode()}]}

    unsigned_tokens: Final = _gemini_tokens(_user({"text": "Weather?"}), {"role": "model", "parts": [call]})
    signed_tokens: Final = _gemini_tokens(_user({"text": "Weather?"}), signed)

    assert unsigned_tokens is not None and signed_tokens is not None
    assert signed_tokens - unsigned_tokens == 3000


@pytest.mark.parametrize("field", ["systemInstruction", "system_instruction"])
def test_a_gemini_system_instruction_is_counted(field: str) -> None:
    instruction: Final = "Answer like a pirate and never mention the weather. " * 10
    question: Final = _user({"text": "Hi"})

    with_instruction: Final = _gemini_tokens(question, **{field: {"parts": [{"text": instruction}]}})
    without_instruction: Final = _gemini_tokens(question)

    assert with_instruction is not None and without_instruction is not None
    assert with_instruction - without_instruction == litellm.token_counter(model=GEMINI_MODEL, text=instruction)


def test_gemini_function_declarations_are_counted() -> None:
    declarations: Final = [
        {
            "functionDeclarations": [
                {"name": f"lookup_{index}", "description": "Look up a customer record by id. " * 5}
                for index in range(10)
            ]
        }
    ]
    question: Final = _user({"text": "Find customer 7."})

    with_tools: Final = _gemini_tokens(question, tools=declarations)
    without_tools: Final = _gemini_tokens(question)

    assert with_tools is not None and without_tools is not None
    assert with_tools - without_tools >= 10 * litellm.token_counter(
        model=GEMINI_MODEL, text="Look up a customer record by id. " * 5
    )


@pytest.mark.parametrize(
    ("config_field", "schema_field"),
    [
        ("generationConfig", "responseSchema"),
        ("generation_config", "response_json_schema"),
        ("config", "response_schema"),
    ],
)
def test_a_gemini_response_schema_is_counted(config_field: str, schema_field: str) -> None:
    schema: Final = {
        "type": "OBJECT",
        "properties": {f"field_{index}": {"type": "STRING", "description": "A customer note. " * 5} for index in range(10)},
    }
    question: Final = _user({"text": "Fill the form."})

    with_schema: Final = _gemini_tokens(question, **{config_field: {"maxOutputTokens": 64, schema_field: schema}})
    without_schema: Final = _gemini_tokens(question, **{config_field: {"maxOutputTokens": 64}})

    assert with_schema is not None and without_schema is not None
    assert with_schema - without_schema == litellm.token_counter(model=GEMINI_MODEL, text=json.dumps(schema))


@pytest.mark.parametrize(("first", "later"), [("config", "generationConfig"), ("generationConfig", "generation_config")])
def test_a_gemini_response_schema_under_a_later_config_name_is_counted(first: str, later: str) -> None:
    schema: Final = {
        "type": "OBJECT",
        "properties": {f"field_{index}": {"type": "STRING", "description": "A customer note. " * 5} for index in range(10)},
    }
    question: Final = _user({"text": "Fill the form."})
    capped: Final = {"maxOutputTokens": 64}
    with_schema: Final = capped | {"responseSchema": schema}

    without_schema: Final = _gemini_tokens(question, **{first: capped, later: capped})
    schema_under_later: Final = _gemini_tokens(question, **{first: capped, later: with_schema})
    schema_under_both: Final = _gemini_tokens(question, **{first: with_schema, later: with_schema})

    assert without_schema is not None and schema_under_later is not None
    assert schema_under_later - without_schema == litellm.token_counter(model=GEMINI_MODEL, text=json.dumps(schema))
    assert schema_under_both == schema_under_later


@pytest.mark.asyncio
async def test_a_large_gemini_response_schema_is_counted_without_blocking_the_event_loop() -> None:
    schema: Final = {
        "type": "OBJECT",
        "properties": {
            f"field_{index}": {"type": "STRING", "description": "A customer note. " * 5}
            for index in range(TOKENIZE_OFF_EVENT_LOOP_MIN_CHARS // 50)
        },
    }
    request_body: Final = {"contents": [_user({"text": "Fill the form."})], "generationConfig": {"responseSchema": schema}}

    counting: Final = asyncio.create_task(
        count_input_tokens(request_body=request_body, raw_body=None, models=(GEMINI_MODEL,))
    )
    await asyncio.sleep(0)

    assert not counting.done()
    assert (await counting)[GEMINI_MODEL] == count_input_tokens_for_model(request_body=request_body, model=GEMINI_MODEL)


@pytest.mark.parametrize("model", ["openai/gpt-5.6-sol", "azure/gpt-5.6-sol"])
def test_a_gemini_body_for_an_openai_deployment_counts_the_chat_messages_it_is_sent_as(model: str) -> None:
    turns: Final = [{"role": role, "parts": [{"text": "ok"}]} for role in ("user", "model") * 100]
    messages: Final = [{"role": "user" if turn["role"] == "user" else "assistant", "content": "ok"} for turn in turns]

    assert count_input_tokens_for_model(request_body={"contents": turns}, model=model) == litellm.token_counter(
        model=model, messages=messages
    )


@pytest.mark.parametrize(
    ("served_models", "counted_as_chat"),
    [
        (("anthropic/claude-sonnet-5-5",), True),
        (("anthropic/claude-sonnet-5-5", "azure/my-gpt-deployment"), True),
        (("vertex_ai/gemini-3.8-flash",), False),
    ],
)
def test_a_gemini_body_for_an_alias_counts_as_chat_only_when_every_deployment_serves_it_through_the_adapter(
    served_models: tuple[str, ...], counted_as_chat: bool
) -> None:
    turns: Final = [{"role": role, "parts": [{"text": "ok"}]} for role in ("user", "model") * 100]
    messages: Final = [{"role": "user" if turn["role"] == "user" else "assistant", "content": "ok"} for turn in turns]

    counted: Final = count_input_tokens_for_model(
        request_body={"contents": turns}, model="team-assistant", served_models=served_models
    )

    assert counted == (
        litellm.token_counter(model="team-assistant", messages=messages)
        if counted_as_chat
        else litellm.token_counter(model="team-assistant", text="ok") * len(turns)
    )


_LONG_SECOND_SYSTEM_PART: Final = {
    "contents": [_user({"text": "Can I expense a taxi?"})],
    "systemInstruction": {"parts": [{"text": "Be brief."}, {"text": "Answer only from the travel policy. " * 200}]},
}
_MANY_SHORT_TURNS: Final = {"contents": [{"role": role, "parts": [{"text": "ok"}]} for role in ("user", "model") * 100]}


def test_a_gemini_body_counts_as_gemini_when_its_deployments_serve_it_natively_whatever_name_it_is_counted_as() -> None:
    counted: Final = count_input_tokens_for_model(
        request_body=_LONG_SECOND_SYSTEM_PART, model="gpt-5.6-sol", served_models=("vertex_ai/gemini-3.8-flash",)
    )

    assert counted is not None
    assert counted >= litellm.token_counter(model="gpt-5.6-sol", text="Answer only from the travel policy. " * 200)


@pytest.mark.parametrize(
    "request_body", [_LONG_SECOND_SYSTEM_PART, _MANY_SHORT_TURNS], ids=["native-count-larger", "chat-count-larger"]
)
def test_a_gemini_body_for_an_alias_mixing_native_and_adapter_deployments_holds_the_larger_count(
    request_body: dict[str, object],
) -> None:
    native: Final = count_input_tokens_for_model(
        request_body=request_body, model="team-assistant", served_models=("gemini/gemini-3.8-flash",)
    )
    adapted: Final = count_input_tokens_for_model(
        request_body=request_body, model="team-assistant", served_models=("anthropic/claude-sonnet-5-5",)
    )
    mixed: Final = count_input_tokens_for_model(
        request_body=request_body,
        model="team-assistant",
        served_models=("anthropic/claude-sonnet-5-5", "gemini/gemini-3.8-flash"),
    )

    assert native is not None and adapted is not None
    assert native != adapted
    assert mixed == max(native, adapted)


@pytest.mark.parametrize(
    "part",
    [
        pytest.param({"fileData": {"mimeType": "application/pdf", "fileUri": "https://files.example.invalid/a.pdf"}}, id="remote-pdf"),
        pytest.param({"fileData": {"mimeType": "image/png", "fileUri": "https://files.example.invalid/cat.png"}}, id="remote-image"),
        pytest.param({"fileData": {"fileUri": "gs://b/report.pdf"}}, id="remote-file-without-mime"),
        pytest.param({"inlineData": {"mimeType": "audio/wav", "data": _wav_base64(60)}}, id="inline-audio"),
        pytest.param({"inlineData": {"mimeType": "video/mp4", "data": "AAAA"}}, id="inline-video"),
    ],
)
def test_a_gemini_body_for_an_openai_deployment_with_media_the_chat_counter_cannot_read_is_unsized(
    part: dict[str, object],
) -> None:
    request_body: Final = {"contents": [_user({"text": "Describe it."}, part)]}

    assert count_input_tokens_for_model(request_body=request_body, model="openai/gpt-5.6-sol") is None


def test_a_gemini_body_for_an_openai_deployment_counts_an_inline_image_like_the_chat_request_it_becomes() -> None:
    contents: Final = [_user({"text": "Describe it."}, {"inlineData": {"mimeType": "image/png", "data": _PNG_PIXEL}})]
    image: Final = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_PNG_PIXEL}"}}
    messages: Final = [{"role": "user", "content": [{"type": "text", "text": "Describe it."}, image]}]

    assert count_input_tokens_for_model(
        request_body={"contents": contents}, model="openai/gpt-5.6-sol"
    ) == litellm.token_counter(model="openai/gpt-5.6-sol", messages=messages)


def test_a_gemini_body_an_openai_deployment_cannot_be_sent_is_left_unsized() -> None:
    assert count_input_tokens_for_model(request_body={"contents": "Fill the form."}, model="openai/gpt-5.6-sol") is None


def test_a_gemini_response_schema_for_an_openai_deployment_is_counted() -> None:
    schema: Final = {
        "type": "OBJECT",
        "properties": {f"field_{index}": {"type": "STRING", "description": "A customer note. " * 5} for index in range(10)},
    }
    contents: Final = [_user({"text": "Fill the form."})]
    capped: Final = {"maxOutputTokens": 64, "responseMimeType": "application/json"}

    with_schema: Final = count_input_tokens_for_model(
        request_body={"contents": contents, "generationConfig": capped | {"responseSchema": schema}},
        model="openai/gpt-5.6-sol",
    )
    without_schema: Final = count_input_tokens_for_model(
        request_body={"contents": contents, "generationConfig": capped}, model="openai/gpt-5.6-sol"
    )

    assert with_schema is not None and without_schema is not None
    assert with_schema - without_schema >= litellm.token_counter(model="openai/gpt-5.6-sol", text=json.dumps(schema))


def test_inline_gemini_audio_counts_by_its_duration() -> None:
    question: Final = {"text": "Transcribe this."}
    text_only: Final = _gemini_tokens(_user(question))
    two_seconds: Final = _gemini_tokens(_user(question, {"inlineData": {"mimeType": "audio/wav", "data": _wav_base64(2)}}))
    four_seconds: Final = _gemini_tokens(
        _user(question, {"inline_data": {"mime_type": "audio/wav", "data": _wav_base64(4)}})
    )

    assert text_only is not None and two_seconds is not None and four_seconds is not None
    assert two_seconds > text_only
    assert four_seconds - text_only == 2 * (two_seconds - text_only)


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param(
            {"text": "Check the weather first. " * 20, "functionCall": {"name": "get_weather", "args": {"city": "Paris"}}},
            id="text-beside-a-function-call",
        ),
        pytest.param(
            {"text": "", "functionResponse": {"name": "get_weather", "response": {"forecast": "sunny " * 200}}},
            id="empty-text-beside-a-function-response",
        ),
        pytest.param(
            {"text": "Transcribe this. " * 20, "inlineData": {"mimeType": "audio/wav", "data": _wav_base64(2)}},
            id="text-beside-inline-audio",
        ),
    ],
)
def test_a_gemini_part_carrying_several_fields_counts_each_like_its_own_part(fields: dict[str, object]) -> None:
    merged: Final = _gemini_tokens(_user(fields))

    assert merged is not None
    assert merged == _gemini_tokens(_user(*({key: value} for key, value in fields.items())))


def test_media_nested_in_a_gemini_function_response_counts_like_a_top_level_part() -> None:
    audio: Final = {"inlineData": {"mimeType": "audio/wav", "data": _wav_base64(2)}}
    response: Final = {"name": "record", "response": {"status": "ok"}}

    nested: Final = _gemini_tokens(_user({"functionResponse": {**response, "parts": [audio]}}))
    top_level: Final = _gemini_tokens(_user({"functionResponse": {**response, "parts": []}}, audio))

    assert nested is not None
    assert nested == top_level


_PNG_PIXEL: Final = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
        "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082"
    )
).decode()


@pytest.mark.parametrize(
    ("part", "fields"),
    [
        pytest.param({"inlineData": {"mimeType": "image/png", "data": _PNG_PIXEL}}, {}, id="inline-image"),
        pytest.param(
            {"inline_data": {"mime_type": "application/pdf", "data": base64.b64encode(b"%PDF-1.4").decode()}},
            {},
            id="inline-pdf",
        ),
        pytest.param({"inlineData": {"mimeType": "video/mp4", "data": "AAAA"}}, {}, id="inline-video"),
        pytest.param({"inlineData": {"mimeType": "text/plain", "data": "aGVsbG8="}}, {}, id="inline-text-file"),
        pytest.param({"inlineData": "not-an-object"}, {}, id="malformed-inline-data"),
        pytest.param(
            {"inlineData": {"mimeType": "audio/aac", "data": base64.b64encode(b"\xff\xf1" * 4000).decode()}},
            {},
            id="audio-without-a-readable-duration",
        ),
        pytest.param({"fileData": {"mimeType": "image/png", "fileUri": "gs://b/cat.png"}}, {}, id="file-image"),
        pytest.param({"file_data": {"mime_type": "video/mp4", "file_uri": "gs://b/clip.mp4"}}, {}, id="file-video"),
        pytest.param({"fileData": {"fileUri": "gs://b/report.pdf"}}, {}, id="file-without-mime"),
        pytest.param(
            {"functionResponse": {"name": "snap", "response": {}, "parts": [{"inlineData": {"mimeType": "image/png", "data": _PNG_PIXEL}}]}},
            {},
            id="image-in-function-response",
        ),
        pytest.param({"text": "Search it."}, {"tools": [{"googleSearch": {}}]}, id="google-search-tool"),
        pytest.param({"text": "Read it."}, {"tools": [{"urlContext": {}}]}, id="url-context-tool"),
        pytest.param(
            {"text": "Run it."},
            {"tools": [{"functionDeclarations": [{"name": "f"}], "codeExecution": {}}]},
            id="code-execution-beside-functions",
        ),
    ],
)
def test_a_gemini_body_with_input_google_sizes_on_its_side_is_unsized(
    part: dict[str, object], fields: dict[str, object]
) -> None:
    assert _gemini_tokens(_user({"text": "Describe it."}, part), **fields) is None


@pytest.mark.asyncio
async def test_a_gemini_body_with_a_raw_body_is_counted_in_python_when_rust_declines_it() -> None:
    request_body: Final = {"contents": [_user({"text": "Explain photosynthesis to a ten year old. " * 30})]}

    counts: Final = await count_input_tokens(
        request_body=request_body, raw_body=json.dumps(request_body).encode(), models=(GEMINI_MODEL,)
    )

    assert counts == {GEMINI_MODEL: count_input_tokens_for_model(request_body=request_body, model=GEMINI_MODEL)}
    assert counts[GEMINI_MODEL] > 0
