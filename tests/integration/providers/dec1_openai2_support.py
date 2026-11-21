"""Shared rig for the `test_dec1_openai2_*` files: native OpenAI round 2 (hosted tools, resource routes, passthrough).

The files reuse the recording provider peer of `dec1_openai_support.py` (`peer`, `deployment`, `Call`) and add:

- `resource_respond`, a canned answer for the OpenAI resource routes the round 2 tests touch: containers,
  vector stores and vector store files, conversations and conversation items, stored chat completions and
  fine-tuning jobs. Every answer echoes the id from the path so a test can tell which raw id reached the peer.
- `with_output`, a Responses API object whose `output` list is chosen by the test (hosted tool call items).
- `managed_container_id` and `decoded_managed_id`, the `cntr_` plus base64 format that
  `ResponsesAPIRequestUtils.build_container_id` (`litellm/responses/utils.py:934`) gives a provider container id.
- `owned_config`, a proxy config built on `tests/integration/proxy_config.yaml` whose `finetune_settings` and
  `files_settings` point at the peer, for the owned proxies that need process-level settings.

No helper ever names a real OpenAI host: every `api_base`, `OPENAI_API_BASE` and `OPENAI_BASE_URL` points at
a local peer.
"""

from __future__ import annotations

import base64
import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final

import yaml
from integration._support.client import JSON_OBJECT, Gateway, string_value
from integration._support.wire import Reply
from integration.providers.dec1_openai_support import (
    PROVIDER_KEY,
    Call,
    default_respond,
    json_reply,
    openai_error,
    responses_object,
)
from pydantic import JsonValue

ENV_OPENAI_KEY: Final = "sk-dec1-openai2-env-key"
FINETUNE_KEY: Final = "sk-dec1-openai2-finetune-key"
FINETUNE_ORG: Final = "org-dec1-openai2-finetune"
FILE_SEARCH_CALL_COST: Final = 0.0025
CONTAINER_SESSION_COST: Final = 0.03

_load_yaml: Final[Callable[[str], object]] = yaml.safe_load


def managed_container_id(provider: str, model_id: str, raw: str) -> str:
    """`cntr_` + base64 of `litellm:custom_llm_provider:<provider>;model_id:<model_id>;container_id:<raw>`."""
    assembled: Final = f"litellm:custom_llm_provider:{provider};model_id:{model_id};container_id:{raw}"
    return f"cntr_{base64.b64encode(assembled.encode()).decode()}"


def decoded_managed_id(managed: str, prefix: str) -> str:
    """The plain text inside a managed `resp_` or `cntr_` id."""
    assert managed.startswith(prefix), managed
    return base64.b64decode(managed.removeprefix(prefix)).decode()


def deployment_id(gateway: Gateway, model: str) -> str:
    """The `model_id` the proxy stored for the deployment a scenario created under the public name `model`."""
    rows: Final = gateway.rows(
        'SELECT model_id FROM "LiteLLM_ProxyModelTable" WHERE model_name = %s '
        "OR model_info->>'team_public_model_name' = %s",
        (model, model),
    )
    assert len(rows) == 1, rows
    return string_value(rows[0]["model_id"])


def with_output(body: Mapping[str, JsonValue], output: Sequence[JsonValue], **extra: JsonValue) -> Reply:
    """A completed Responses object (usage 11 + 7 tokens) whose `output` is exactly `output`."""
    return json_reply({**responses_object(body, **extra), "output": list(output)})


def container_object(identity: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {
        "id": identity,
        "object": "container",
        "created_at": 1,
        "status": "running",
        "name": "dec1-openai2",
        "expires_after": {"anchor": "last_active_at", "minutes": 20},
        **extra,
    }


def vector_store_object(identity: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {
        "id": identity,
        "object": "vector_store",
        "created_at": 1,
        "name": "dec1-openai2-store",
        "usage_bytes": 0,
        "file_counts": {"in_progress": 0, "completed": 1, "failed": 0, "cancelled": 0, "total": 1},
        "status": "completed",
        "expires_after": None,
        "expires_at": None,
        "last_active_at": 1,
        "metadata": {},
        **extra,
    }


def vector_store_file_object(store: str, identity: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {
        "id": identity,
        "object": "vector_store.file",
        "usage_bytes": 12,
        "created_at": 1,
        "vector_store_id": store,
        "status": "completed",
        "last_error": None,
        "chunking_strategy": {"type": "static", "static": {"max_chunk_size_tokens": 800, "chunk_overlap_tokens": 400}},
        "attributes": {},
        **extra,
    }


def fine_tuning_job(identity: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {
        "id": identity,
        "object": "fine_tuning.job",
        "created_at": 1,
        "error": None,
        "fine_tuned_model": None,
        "finished_at": None,
        "hyperparameters": {"n_epochs": "auto", "batch_size": "auto", "learning_rate_multiplier": "auto"},
        "model": "gpt-4.1-mini-2025-04-14",
        "organization_id": "org-dec1-openai2",
        "result_files": [],
        "seed": 7,
        "status": "validating_files",
        "trained_tokens": None,
        "training_file": "file-dec1-openai2-train",
        "validation_file": None,
        "estimated_finish": None,
        "integrations": [],
        **extra,
    }


def _segments(path: str) -> tuple[str, ...]:
    return tuple(part for part in path.removeprefix("/v1").split("/") if part)


def _container_reply(call: Call, parts: tuple[str, ...]) -> Reply:
    if len(parts) == 1:
        if call.method == "POST":
            return json_reply(container_object("cntr_dec1openai2raw", name=call.body.get("name") or "unnamed"))
        return json_reply(
            {
                "object": "list",
                "data": [container_object("cntr_dec1openai2listed")],
                "first_id": "cntr_dec1openai2listed",
                "last_id": "cntr_dec1openai2listed",
                "has_more": False,
            }
        )
    if call.method == "DELETE":
        return json_reply({"id": parts[1], "object": "container.deleted", "deleted": True})
    return json_reply(container_object(parts[1]))


def _vector_store_reply(call: Call, parts: tuple[str, ...]) -> Reply:
    if len(parts) == 1:
        if call.method == "POST":
            return json_reply(vector_store_object("vs_dec1openai2created"))
        return json_reply({"object": "list", "data": [vector_store_object("vs_dec1openai2listed")], "has_more": False})
    store: Final = parts[1]
    if len(parts) == 2:
        if call.method == "DELETE":
            return json_reply({"id": store, "object": "vector_store.deleted", "deleted": True})
        return json_reply(vector_store_object(store, **({"name": call.body["name"]} if "name" in call.body else {})))
    if len(parts) == 3:
        if call.method == "POST":
            return json_reply(
                vector_store_file_object(store, string_value(call.body.get("file_id") or "file-dec1-openai2"))
            )
        return json_reply(
            {
                "object": "list",
                "data": [vector_store_file_object(store, "file-dec1-openai2-listed")],
                "first_id": "file-dec1-openai2-listed",
                "last_id": "file-dec1-openai2-listed",
                "has_more": False,
            }
        )
    file_id: Final = parts[3]
    if len(parts) == 5 and parts[4] == "content":
        return json_reply(
            {
                "file_id": file_id,
                "filename": "notes.txt",
                "attributes": {},
                "content": [{"type": "text", "text": "dec1 openai2 file text"}],
            }
        )
    if call.method == "DELETE":
        return json_reply({"id": file_id, "object": "vector_store.file.deleted", "deleted": True})
    attributes: Final = call.body.get("attributes")
    return json_reply(vector_store_file_object(store, file_id, attributes=attributes if attributes is not None else {}))


def _conversation_reply(call: Call, parts: tuple[str, ...]) -> Reply:
    if len(parts) == 1:
        return json_reply(
            {"id": "conv_dec1openai2", "object": "conversation", "created_at": 1, "metadata": call.body.get("metadata")}
        )
    conversation: Final = parts[1]
    if len(parts) == 2:
        if call.method == "DELETE":
            return json_reply({"id": conversation, "object": "conversation.deleted", "deleted": True})
        return json_reply(
            {"id": conversation, "object": "conversation", "created_at": 1, "metadata": call.body.get("metadata", {})}
        )
    if len(parts) == 3:
        return json_reply(
            {
                "object": "list",
                "data": [
                    {
                        "id": "msg_dec1openai2",
                        "type": "message",
                        "status": "completed",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "dec1 openai2 item"}],
                    }
                ],
                "first_id": "msg_dec1openai2",
                "last_id": "msg_dec1openai2",
                "has_more": False,
            }
        )
    if call.method == "DELETE":
        return json_reply({"id": conversation, "object": "conversation", "created_at": 1, "metadata": {}})
    return json_reply({"id": parts[3], "type": "message", "status": "completed", "role": "user", "content": []})


def _stored_chat_reply(call: Call, parts: tuple[str, ...]) -> Reply:
    if len(parts) == 2:
        return json_reply(
            {"object": "list", "data": [{"id": "chatcmpl-dec1stored", "object": "chat.completion"}], "has_more": False}
        )
    completion: Final = parts[2]
    if len(parts) == 4 and parts[3] == "messages":
        return json_reply(
            {
                "object": "list",
                "data": [{"id": f"{completion}-0", "role": "user", "content": "dec1 stored prompt"}],
                "has_more": False,
            }
        )
    if call.method == "DELETE":
        return json_reply({"id": completion, "object": "chat.completion.deleted", "deleted": True})
    return json_reply(
        {"id": completion, "object": "chat.completion", "metadata": call.body.get("metadata", {}), "choices": []}
    )


def _fine_tuning_reply(call: Call, parts: tuple[str, ...]) -> Reply:
    if len(parts) == 2:
        if call.method == "POST":
            return json_reply(
                fine_tuning_job(
                    "ftjob-dec1openai2",
                    model=call.body.get("model"),
                    training_file=call.body.get("training_file"),
                )
            )
        return json_reply({"object": "list", "data": [fine_tuning_job("ftjob-dec1openai2-listed")], "has_more": False})
    if len(parts) == 4 and parts[3] == "cancel":
        return json_reply(fine_tuning_job(parts[2], status="cancelled"))
    return json_reply(fine_tuning_job(parts[2]))


def resource_respond(call: Call) -> Reply:
    """Canned OpenAI resource answers keyed on the path; everything else falls back to `default_respond`."""
    parts: Final = _segments(call.path)
    if not parts:
        return openai_error(404, f"dec1 openai2 peer has no route {call.path}")
    if parts[0] == "containers":
        return _container_reply(call, parts)
    if parts[0] == "vector_stores":
        return _vector_store_reply(call, parts)
    if parts[0] == "conversations":
        return _conversation_reply(call, parts)
    if parts[:2] == ("chat", "completions") and (len(parts) > 2 or call.method == "GET"):
        return _stored_chat_reply(call, parts)
    if parts[:2] == ("fine_tuning", "jobs"):
        return _fine_tuning_reply(call, parts)
    if parts == ("files",) and call.method == "POST":
        return json_reply(
            {
                "id": "file-dec1-openai2-train",
                "object": "file",
                "bytes": 10,
                "created_at": 1,
                "filename": "train.jsonl",
                "purpose": "fine-tune",
                "status": "processed",
            }
        )
    return default_respond(call)


def calls_to(calls: Sequence[Call], prefix: str) -> tuple[Call, ...]:
    """The calls whose path (without a leading `/v1`) starts with `prefix`; model discovery GETs drop out."""
    return tuple(call for call in calls if call.path.removeprefix("/v1").startswith(prefix))


def owned_config(directory: Path, base_url: str, *, organization: str | None = None) -> Path:
    """The shared proxy config plus `finetune_settings` and `files_settings` for OpenAI at `base_url`.

    `organization`, when given, is set on the fine-tuning settings so a test can read the
    `openai-organization` header the OpenAI SDK derives from it.
    """
    base: Final = JSON_OBJECT.validate_python(_load_yaml(Path("tests/integration/proxy_config.yaml").read_text()))
    finetune: Final[dict[str, JsonValue]] = {
        "custom_llm_provider": "openai",
        "api_base": base_url,
        "api_key": FINETUNE_KEY,
        **({"organization": organization} if organization else {}),
    }
    configuration: Final = {
        **base,
        "finetune_settings": [finetune],
        "files_settings": [{"custom_llm_provider": "openai", "api_base": base_url, "api_key": FINETUNE_KEY}],
        "router_settings": {"disable_cooldowns": True, "num_retries": 0},
    }
    path: Final = directory / "dec1-openai2.yaml"
    path.write_text(yaml.safe_dump(json.loads(json.dumps(configuration))))
    return path


def unique(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


__all__ = [
    "CONTAINER_SESSION_COST",
    "ENV_OPENAI_KEY",
    "FILE_SEARCH_CALL_COST",
    "FINETUNE_KEY",
    "FINETUNE_ORG",
    "PROVIDER_KEY",
    "calls_to",
    "container_object",
    "decoded_managed_id",
    "deployment_id",
    "fine_tuning_job",
    "managed_container_id",
    "owned_config",
    "resource_respond",
    "unique",
    "vector_store_file_object",
    "vector_store_object",
    "with_output",
]
