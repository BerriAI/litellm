"""Contract: chat request features on `anthropic/` deployments that round 1 and 2 left out.

Three groups. A model-scoped file id that `/v1/files` handed out (`file-<base64 of "litellm:<raw id>;model,<model
group>">`, `encode_file_id_with_model`, `litellm/proxy/openai_files_endpoints/common_utils.py:327`) is decoded back
to the raw Anthropic id inside a chat `file` part by `update_messages_with_model_file_ids`
(`litellm/litellm_core_utils/prompt_templates/file_id_utils.py:50`, called from `litellm/main.py:6054`) before the
Anthropic translation builds its `file` source. A closed `response_format` JSON schema sent as native
`output_format` is rewritten by `filter_anthropic_output_schema` (`litellm/llms/anthropic/chat/transformation.py:999`)
the way the Anthropic SDK does: constraints the grammar cannot enforce move into the description, `oneOf` becomes
`anyOf`, every object gets `additionalProperties: false`, and `$defs` are inlined. `reasoning_effort` on an adaptive
model becomes `output_config.effort`, gated per model by `supports_<level>_reasoning_effort`
(`_validate_effort_for_model`, `transformation.py:802`).

Every upstream is a loopback peer.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway, object_value
from integration.providers.dec1_anthropic2_support import FILE_ID, FILES_BETA
from integration.providers.dec1_anthropic3_support import encoded_file_id
from integration.providers.dec1_anthropic_support import (
    ADAPTIVE_BACKEND,
    betas,
    chat,
    content_blocks,
    deployment,
    error_message,
    marker,
    peer,
    sent_message,
    served,
)
from pydantic import JsonValue


def _file_part_blocks(gateway: Gateway, file_format: str) -> tuple[list[dict[str, JsonValue]], frozenset[str]]:
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider)
        part: Final[dict[str, JsonValue]] = {
            "type": "file",
            "file": {"file_id": encoded_file_id(FILE_ID, model), "format": file_format},
        }
        messages: Final[list[JsonValue]] = [{"role": "user", "content": [part, {"type": "text", "text": marker()}]}]
        served(chat(gateway, model, messages=messages))
        call: Final = provider.only()
        assert b"litellm:" not in call.raw
        return content_blocks(sent_message(call, 0)), betas(call)


def _sent_schema(gateway: Gateway, schema: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Send `schema` as a `response_format` json_schema and return the `output_format.schema` the peer received."""
    response_format: Final[dict[str, JsonValue]] = {
        "type": "json_schema",
        "json_schema": {"name": "dec1_shape", "schema": schema},
    }
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), response_format=response_format))
        body: Final = provider.only().body
        assert "tools" not in body, body
        return object_value(object_value(body["output_format"])["schema"])


def _closed(properties: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def _property(schema: dict[str, JsonValue], name: str) -> dict[str, JsonValue]:
    return object_value(object_value(schema["properties"])[name])


def test_a_model_scoped_image_file_id_is_decoded_to_the_raw_anthropic_id_in_a_chat_message(gateway: Gateway) -> None:
    """The caller sends a `file` part whose `file_id` is the model-scoped id of `file_011dec1synthetic` for this
    deployment, with `format: image/png`.

    The upstream user turn must open with `{"type": "image", "source": {"type": "file", "file_id":
    "file_011dec1synthetic"}}`, the raw body must not contain the `litellm:` envelope, and the files beta must be
    sent. Without the decode, Anthropic answers 404 for an id it never issued.
    """
    blocks, sent_betas = _file_part_blocks(gateway, "image/png")
    assert blocks[0] == {"type": "image", "source": {"type": "file", "file_id": FILE_ID}}
    assert FILES_BETA in sent_betas


def test_a_model_scoped_text_file_id_is_decoded_into_a_document_file_source(gateway: Gateway) -> None:
    """Same model-scoped id with `format: text/plain`.

    The upstream user turn must open with a `document` block whose `source` is `{"type": "file", "file_id":
    "file_011dec1synthetic"}` (`get_original_file_id`, `common_utils.py:414`, then
    `select_anthropic_content_block_type_for_file`).
    """
    blocks, sent_betas = _file_part_blocks(gateway, "text/plain")
    assert blocks[0]["type"] == "document"
    assert object_value(blocks[0]["source"]) == {"type": "file", "file_id": FILE_ID}
    assert FILES_BETA in sent_betas


def test_numeric_constraints_move_into_the_property_description(gateway: Gateway) -> None:
    """The closed schema has `count: {"type": "integer", "minimum": 1, "maximum": 10}`.

    Upstream, `count` must be `{"type": "integer", "description": "Note: minimum value: 1, maximum value: 10."}`:
    Anthropic rejects `minimum` and `maximum` in `output_format`, so they are removed and stated in the description
    in the fixed label order (`filter_anthropic_output_schema`, `transformation.py:1078`).
    """
    schema: Final = _sent_schema(gateway, _closed({"count": {"type": "integer", "minimum": 1, "maximum": 10}}))
    assert _property(schema, "count") == {
        "type": "integer",
        "description": "Note: minimum value: 1, maximum value: 10.",
    }


def test_a_removed_constraint_is_appended_to_an_existing_description(gateway: Gateway) -> None:
    """The closed schema has `name: {"type": "string", "description": "Display name", "maxLength": 20}`.

    Upstream, `name.description` must be `Display name Note: maximum length: 20.` (`_described_with_note`,
    `transformation.py:491`) and `maxLength` must be gone.
    """
    schema: Final = _sent_schema(
        gateway, _closed({"name": {"type": "string", "description": "Display name", "maxLength": 20}})
    )
    assert _property(schema, "name") == {"type": "string", "description": "Display name Note: maximum length: 20."}


def test_array_item_counts_above_one_and_unique_items_are_removed_but_min_items_one_is_kept(gateway: Gateway) -> None:
    """The closed schema has `tags: {"type": "array", "items": {"type": "string"}, "minItems": 1, "uniqueItems":
    false}` and `codes: {"type": "array", "items": {"type": "string"}, "minItems": 2}`.

    Upstream, `tags` must keep `minItems: 1` and lose `uniqueItems` with no note (a false boolean imposes nothing),
    and `codes` must lose `minItems` and get the description `Note: minimum number of items: 2.`
    (`transformation.py:1068`).
    """
    schema: Final = _sent_schema(
        gateway,
        _closed(
            {
                "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1, "uniqueItems": False},
                "codes": {"type": "array", "items": {"type": "string"}, "minItems": 2},
            }
        ),
    )
    assert _property(schema, "tags") == {"type": "array", "items": {"type": "string"}, "minItems": 1}
    assert _property(schema, "codes") == {
        "type": "array",
        "items": {"type": "string"},
        "description": "Note: minimum number of items: 2.",
    }


def test_one_of_is_rewritten_to_any_of(gateway: Gateway) -> None:
    """The closed schema has `value: {"oneOf": [{"type": "string"}, {"type": "integer"}]}`.

    Upstream, `value` must carry `anyOf` with the same two branches and no `oneOf`, since Anthropic rejects
    `oneOf` (`transformation.py:1121`).
    """
    schema: Final = _sent_schema(gateway, _closed({"value": {"oneOf": [{"type": "string"}, {"type": "integer"}]}}))
    value: Final = _property(schema, "value")
    assert "oneOf" not in value
    assert value["anyOf"] == [{"type": "string"}, {"type": "integer"}]


def test_a_nested_object_gets_additional_properties_false(gateway: Gateway) -> None:
    """The closed schema has a nested object `address: {"type": "object", "properties": {"city": {"type":
    "string"}}, "required": ["city"]}` with no `additionalProperties`.

    Upstream, `address.additionalProperties` must be `false`, which Anthropic requires on every object schema
    (`transformation.py:1131`).
    """
    nested: Final[dict[str, JsonValue]] = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    schema: Final = _sent_schema(gateway, _closed({"address": nested}))
    assert _property(schema, "address").get("additionalProperties") is False


def test_defs_references_are_inlined_before_the_schema_is_sent(gateway: Gateway) -> None:
    """The closed schema keeps its item shape in `$defs.Item` and references it as `item: {"$ref":
    "#/$defs/Item"}`.

    Upstream, the schema must have no `$defs` and `item` must carry the `Item` properties inline
    (`_defs_unpacked`, `transformation.py:1898`). UNVERIFIED: whether the inlined copy keeps a `$ref` key; only the
    absent `$defs` and the inlined properties are pinned.
    """
    item: Final[dict[str, JsonValue]] = _closed({"sku": {"type": "string"}})
    schema: Final = _sent_schema(gateway, {**_closed({"item": {"$ref": "#/$defs/Item"}}), "$defs": {"Item": item}})
    assert "$defs" not in schema
    assert object_value(_property(schema, "item")["properties"]) == {"sku": {"type": "string"}}


def test_reasoning_effort_max_becomes_output_config_max_on_an_adaptive_model(gateway: Gateway) -> None:
    """On `claude-opus-4-6` (`supports_max_reasoning_effort: true`) the caller sends `reasoning_effort: max`.

    The upstream body must carry `thinking: {"type": "adaptive", "display": "summarized"}` and `output_config:
    {"effort": "max"}` (`REASONING_EFFORT_TO_OUTPUT_CONFIG_EFFORT`, `transformation.py:579`). Round 1 pins only
    minimal to high.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, f"anthropic/{ADAPTIVE_BACKEND}", priced=False)
        served(chat(gateway, model, reasoning_effort="max"))
        body: Final = provider.only().body
        assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert body["output_config"] == {"effort": "max"}


def test_reasoning_effort_xhigh_is_refused_on_a_model_without_xhigh_support(gateway: Gateway) -> None:
    """On `claude-opus-4-6`, whose cost map row has no `supports_xhigh_reasoning_effort`, the caller sends
    `reasoning_effort: xhigh`.

    The caller must get 400 whose message says `effort='xhigh' is not supported by this model`, and the peer must
    not be called (`_validate_effort_for_model`, `transformation.py:809`).
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, f"anthropic/{ADAPTIVE_BACKEND}", priced=False)
        response: Final = chat(gateway, model, reasoning_effort="xhigh")
        assert response.status_code == 400, response.text
        assert "effort='xhigh' is not supported by this model" in error_message(response)
        assert provider.posts() == ()


def test_reasoning_effort_xhigh_is_sent_on_a_model_with_xhigh_support(gateway: Gateway) -> None:
    """On `claude-opus-4-7` (`supports_xhigh_reasoning_effort: true` in the cost map) the caller sends
    `reasoning_effort: xhigh`.

    The upstream body must carry `output_config: {"effort": "xhigh"}` and adaptive thinking.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "anthropic/claude-opus-4-7", priced=False)
        served(chat(gateway, model, reasoning_effort="xhigh"))
        body: Final = provider.only().body
        assert body["output_config"] == {"effort": "xhigh"}
        assert object_value(body["thinking"])["type"] == "adaptive"


def test_the_interleaved_thinking_beta_from_the_caller_is_forwarded_with_thinking(gateway: Gateway) -> None:
    """The caller enables thinking with `reasoning_effort: low` and sends `anthropic-beta:
    interleaved-thinking-2025-05-14`.

    The upstream request must carry that beta and the `thinking` budget: the anthropic column of
    `litellm/anthropic_beta_headers_config.json` maps it to itself, so it is allowed through, while litellm never
    derives it on its own.
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(
            chat(
                gateway,
                deployment(scenario, provider),
                reasoning_effort="low",
                headers={"anthropic-beta": "interleaved-thinking-2025-05-14"},
            )
        )
        call: Final = provider.only()
        assert "interleaved-thinking-2025-05-14" in betas(call)
        assert call.body["thinking"] == {"type": "enabled", "budget_tokens": 1024}


@pytest.mark.parametrize("effort", ["low", "high"])
def test_no_interleaved_thinking_beta_is_derived_from_thinking_alone(gateway: Gateway, effort: str) -> None:
    """The caller enables thinking with `reasoning_effort` and sends no `anthropic-beta`.

    The upstream request must not carry `interleaved-thinking-2025-05-14`: no chat code path adds it
    (`update_headers_with_optional_anthropic_beta`, `transformation.py:2455`), so tool calls between thinking blocks
    stay opt-in for the caller.
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), reasoning_effort=effort))
        assert "interleaved-thinking-2025-05-14" not in betas(provider.only())
