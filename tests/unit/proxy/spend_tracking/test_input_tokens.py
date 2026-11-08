"""Tests for input-token counting shared across the reservation path's models."""

from __future__ import annotations

import base64
import json
from types import MappingProxyType
from typing import Final

import pytest

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


def _one_page_pdf_message() -> list[dict[str, object]]:
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
    pdf: Final = base64.b64encode(header + b"".join(objects) + xref + trailer).decode()
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


def test_count_input_tokens_for_model_gemini_contents() -> None:
    body: Final = {
        "contents": [{"role": "user", "parts": [{"text": "Hello from Gemini Vertex AI passthrough"}]}],
        "systemInstruction": {"parts": [{"text": "You are a helpful assistant"}]},
    }
    tokens: Final = count_input_tokens_for_model(request_body=body, model="gpt-4o")
    assert tokens is not None
    assert tokens > 0


def test_count_input_tokens_for_model_gemini_function_call() -> None:
    body: Final = {
        "contents": [
            {
                "role": "model",
                "parts": [{"functionCall": {"name": "get_weather", "args": {"location": "San Francisco"}}}],
            },
            {
                "role": "user",
                "parts": [{"functionResponse": {"name": "get_weather", "response": {"temperature": "70F"}}}],
            },
        ],
    }
    tokens: Final = count_input_tokens_for_model(request_body=body, model="gpt-4o")
    assert tokens is not None
    assert tokens > 0


def test_count_input_tokens_for_model_gemini_inline_data_returns_none() -> None:
    body: Final = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": "What is in this image?"},
                    {"inlineData": {"mimeType": "image/png", "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"}},
                ],
            }
        ],
    }
    assert count_input_tokens_for_model(request_body=body, model="gpt-4o") is None


def test_count_input_tokens_for_model_gemini_executable_code() -> None:
    body: Final = {
        "contents": [
            {
                "role": "model",
                "parts": [{"executableCode": {"language": "PYTHON", "code": "print('hello world')"}}],
            },
            {
                "role": "user",
                "parts": [{"codeExecutionResult": {"outcome": "OUTCOME_OK", "output": "hello world\n"}}],
            },
        ],
    }
    tokens: Final = count_input_tokens_for_model(request_body=body, model="gpt-4o")
    assert tokens is not None
    assert tokens > 0


def test_count_input_tokens_for_model_gemini_unrecognized_part_returns_none() -> None:
    body: Final = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": "Calculate something"},
                    {"unrecognizedFutureBillablePart": {"foo": "bar"}},
                ],
            }
        ],
    }
    assert count_input_tokens_for_model(request_body=body, model="gpt-4o") is None


def test_count_input_tokens_for_model_gemini_cached_content_returns_none() -> None:
    body: Final = {
        "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
        "cachedContent": "cachedContents/12345678",
    }
    assert count_input_tokens_for_model(request_body=body, model="gpt-4o") is None


def test_count_input_tokens_for_model_gemini_cached_content_extra_body_returns_none() -> None:
    body: Final = {
        "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
        "extra_body": {"cached_content": "cachedContents/12345678"},
    }
    assert count_input_tokens_for_model(request_body=body, model="gpt-4o") is None


def test_approximate_input_size_skips_gemini_media_data() -> None:
    from litellm.proxy.spend_tracking.input_tokens import _approximate_input_size

    large_dummy_media: Final = "A" * 100_000
    body: Final = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": "Short prompt"},
                    {"inlineData": {"mimeType": "image/png", "data": large_dummy_media}},
                ],
            }
        ],
    }
    assert _approximate_input_size(body) < 1_000, (
        "Sizing should only account for Short prompt, not 100k chars of inline media data"
    )


def test_approximate_gemini_part_size_cheap_payload_sizing() -> None:
    from litellm.proxy.spend_tracking.input_tokens import _approximate_gemini_part_size

    part: Final = {
        "functionResponse": {
            "name": "large_output",
            "response": {"results": [{"id": i, "data": "item" * 10} for i in range(100)]},
        }
    }
    size = _approximate_gemini_part_size(part)
    assert size > 1_000, "Should size nested functionResponse payloads"


def test_count_input_tokens_for_model_gemini_extra_body_overrides_return_none() -> None:
    body_contents_override: Final = {
        "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
        "extra_body": {"contents": [{"role": "user", "parts": [{"text": "Overridden huge prompt"}]}]},
    }
    assert count_input_tokens_for_model(request_body=body_contents_override, model="gpt-4o") is None

    body_sys_override: Final = {
        "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
        "extra_body": {"systemInstruction": "Overridden system prompt"},
    }
    assert count_input_tokens_for_model(request_body=body_sys_override, model="gpt-4o") is None

    body_tools_override: Final = {
        "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
        "extra_body": {"tools": [{"functionDeclarations": []}]},
    }
    assert count_input_tokens_for_model(request_body=body_tools_override, model="gpt-4o") is None


@pytest.mark.parametrize(
    "retrieval_tool",
    (
        {"urlContext": {}},
        {"url_context": {}},
        {"googleSearch": {}},
        {"google_search": {}},
        {"googleSearchRetrieval": {}},
        {"vertexRetrieval": {}},
        {"enterpriseWebSearch": {}},
    ),
)
def test_count_input_tokens_for_model_gemini_builtin_retrieval_returns_none(retrieval_tool: dict) -> None:
    body: Final = {
        "contents": [{"role": "user", "parts": [{"text": "What is on this website?"}]}],
        "tools": [retrieval_tool],
    }
    assert (
        count_input_tokens_for_model(
            request_body=body, model="gpt-4o", route="/vertex_ai/models/gemini-pro:generateContent"
        )
        is None
    ), "Built-in retrieval tools fetch external uncounted input, so must return None for max fallback"


def test_count_input_tokens_for_model_gemini_function_declarations_counted() -> None:
    body: Final = {
        "contents": [{"role": "user", "parts": [{"text": "What is the weather?"}]}],
        "tools": [{"functionDeclarations": [{"name": "get_weather", "description": "Get current weather"}]}],
    }
    tokens = count_input_tokens_for_model(
        request_body=body, model="gpt-4o", route="/vertex_ai/models/gemini-pro:generateContent"
    )
    assert tokens is not None, "Function declarations without retrieval tools should be tokenized"
    assert tokens > 5, "Tokens should account for prompt and function declaration schema"


def test_count_input_tokens_for_model_responses_route_ignores_unrelated_contents() -> None:
    long_input: Final = "This is a long input prompt for Responses endpoint " * 10
    body: Final = {
        "input": long_input,
        "contents": "Hi",
    }
    tokens = count_input_tokens_for_model(request_body=body, model="gpt-4o", route="/v1/responses")
    assert tokens is not None, "Responses route should count input tokens"
    assert tokens > 20, "For /v1/responses route, the billable input is 'input', not the decoy 'contents'"

    tokens_no_route = count_input_tokens_for_model(request_body=body, model="gpt-4o")
    assert tokens_no_route == tokens, "Default route precedence should count input field over contents"


def test_count_input_tokens_for_model_native_gemini_route_counts_contents() -> None:
    body: Final = {
        "contents": [{"role": "user", "parts": [{"text": "Hello Gemini native"}]}],
        "input": "decoy",
    }
    tokens = count_input_tokens_for_model(
        request_body=body, model="gpt-4o", route="/vertex_ai/models/gemini-pro:generateContent"
    )
    assert tokens is not None, "Native Gemini route should count contents"
    assert tokens > 0, "On native Gemini route, contents is counted and decoy input is ignored"


def test_count_input_tokens_for_model_gemini_preserves_whitespace() -> None:
    whitespace_text: Final = ("\n\t \n " * 100) + "Hi"
    body: Final = {"contents": [{"role": "user", "parts": [{"text": whitespace_text}]}]}
    tokens = count_input_tokens_for_model(request_body=body, model="gpt-4o")
    assert tokens is not None, "Input token count should not be None"
    assert tokens > 10, "Whitespace tokens must be preserved, not stripped to 1 token"


def test_count_input_tokens_for_model_azure_deployment_route_ignores_gemini_name() -> None:
    body: Final = {
        "messages": [{"role": "user", "content": "A long message with lots of tokens " * 10}],
        "contents": "Hi",
    }
    route: Final = "/openai/deployments/gemini-2.0-flash/chat/completions"
    tokens = count_input_tokens_for_model(request_body=body, model="gpt-4o", route=route)
    assert tokens is not None, "Azure deployment route should count tokens"
    assert tokens > 20, "Must count the long message tokens, not the 1 token from contents"


def test_count_input_tokens_for_model_vertex_claude_raw_predict_counts_messages() -> None:
    body: Final = {
        "messages": [{"role": "user", "content": "A Vertex Claude prompt with several tokens " * 5}],
    }
    route: Final = (
        "/vertex_ai/v1/projects/my-project/locations/us-east5/publishers/anthropic/models/claude-3-5-sonnet:rawPredict"
    )
    tokens = count_input_tokens_for_model(request_body=body, model="gpt-4o", route=route)
    assert tokens is not None, "Vertex Claude rawPredict route should count tokens"
    assert tokens > 20, "Must count Anthropic messages on rawPredict endpoints"


def test_is_native_gemini_route_matches_endpoint_action_suffix() -> None:
    from litellm.proxy.spend_tracking.input_tokens import is_native_gemini_route

    route_with_responses_proj: Final = (
        "/vertex_ai/v1/projects/responses-prod/locations/us-central1/publishers/google"
        "/models/gemini-1.5-flash:generateContent"
    )
    assert is_native_gemini_route(route_with_responses_proj) is True, (
        "Vertex route with project id containing 'responses' must match native action suffix"
    )

    route_with_query: Final = (
        "/vertex_ai/v1/projects/my-proj/locations/us-central1/publishers/google"
        "/models/gemini-1.5-flash:streamGenerateContent?alt=sse"
    )
    assert is_native_gemini_route(route_with_query) is True, "Route with query params must match action suffix"

    assert is_native_gemini_route("/v1/chat/completions") is False
    assert is_native_gemini_route("/openai/deployments/gemini-2.0-flash/chat/completions") is False
    assert (
        is_native_gemini_route("/vertex_ai/v1/projects/proj/locations/us/publishers/anthropic/models/claude:rawPredict")
        is False
    )


def test_count_input_tokens_for_model_gemini_counts_response_schema() -> None:
    schema: Final = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "biography": {"type": "string", "description": "A long detailed life history " * 5},
        },
        "required": ["name", "age"],
    }
    body_prompt_only: Final = {"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]}
    tokens_prompt = count_input_tokens_for_model(request_body=body_prompt_only, model="gpt-4o")
    assert tokens_prompt is not None, "Prompt tokens should be counted"

    body_with_schema: Final = {
        "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
        "generationConfig": {"responseSchema": schema},
    }
    tokens_schema = count_input_tokens_for_model(request_body=body_with_schema, model="gpt-4o")
    assert tokens_schema is not None, "Request with responseSchema should be counted"
    assert tokens_schema > tokens_prompt, "Response schema must contribute billable input tokens to reservation"

    body_snake: Final = {
        "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
        "generation_config": {"response_schema": schema},
    }
    assert count_input_tokens_for_model(request_body=body_snake, model="gpt-4o") == tokens_schema, (
        "Snake case response_schema should yield identical token count"
    )

    body_extra_schema: Final = {
        "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
        "extra_body": {"responseSchema": schema},
    }
    assert count_input_tokens_for_model(request_body=body_extra_schema, model="gpt-4o") is None, (
        "extra_body schema override must return None to preserve fallback"
    )
