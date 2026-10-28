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
    ("group", "deployment_model"),
    [("gpt-4o", "anthropic/claude-haiku-5-5"), ("pdf-reports", "azure/gpt-5.6-sol")],
    ids=["gpt-named-group-served-by-claude", "custom-group-served-by-azure-gpt"],
)
def test_a_model_group_counts_like_the_deployment_that_serves_it(group: str, deployment_model: str) -> None:
    request_body: Final = {"model": group, "messages": _one_page_pdf_message()}
    counted_with: Final = deployment_model.partition("/")[2]

    assert models_counted_for(group, _router((group, deployment_model))) == (counted_with,)
    assert count_input_tokens_for_model(request_body=request_body, model=counted_with) != count_input_tokens_for_model(
        request_body=request_body, model=group
    )


def test_a_group_of_differently_priced_deployments_counts_with_each_and_a_name_no_deployment_serves_counts_as_is():
    router: Final = _router(("mixed", "anthropic/claude-haiku-5-5"), ("mixed", "azure/gpt-5.6-sol"))

    assert models_counted_for("mixed", router) == ("claude-haiku-5-5", "gpt-5.6-sol")
    assert models_counted_for("not-deployed", router) == ("not-deployed",)
    assert models_counted_for("gpt-4o", None) == ("gpt-4o",)


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
