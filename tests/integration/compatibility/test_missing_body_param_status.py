from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from email.parser import BytesParser
from email.policy import HTTP
from functools import partial
from pathlib import Path
from typing import Final

import httpx
import psutil
import pytest
from integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy_process
from integration._support.upstream import ScenarioHandle, delete_scenario, register_scenario
from integration._support.wire import Reply, Request, Wire, wire_server
from openai import AsyncOpenAI, BadRequestError, OpenAI
from openai.types import Video
from pydantic import JsonValue

from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_character_id_with_provider,
    encode_video_id_with_provider,
)
from tests.integration.cost_calculation.cost_tracking_case import (
    BinaryResponse,
    JsonResponse,
    RoutedResponse,
    SseResponse,
)

_Route = tuple[str, str, tuple[str, ...], dict[str, JsonValue], str]
_ROUTES: Final[dict[str, _Route]] = {
    "acompletion": (
        "/v1/chat/completions",
        "/chat/completions",
        ("messages",),
        {"messages": [{"role": "user", "content": "chat"}]},
        "openai/gpt-4o-mini",
    ),
    "aembedding": (
        "/v1/embeddings",
        "/embeddings",
        ("input",),
        {"input": ["embedding"]},
        "openai/text-embedding-3-small",
    ),
    "aresponses": ("/v1/responses", "/responses", ("input",), {"input": "response"}, "openai/gpt-4o-mini"),
    "acreate_batch": (
        "/v1/batches",
        "/batches",
        ("input_file_id", "endpoint", "completion_window"),
        {"input_file_id": "file-audit", "endpoint": "/v1/chat/completions", "completion_window": "24h"},
        "openai/gpt-4o-mini",
    ),
    "aspeech": (
        "/v1/audio/speech",
        "/audio/speech",
        ("input",),
        {"input": "speech", "voice": "alloy"},
        "openai/gpt-4o-mini-tts",
    ),
    "amoderation": ("/v1/moderations", "/moderations", ("input",), {"input": "moderate"}, "openai/gpt-4o-mini"),
    "aimage_generation": (
        "/v1/images/generations",
        "/image/generations",
        ("prompt",),
        {"prompt": "image"},
        "openai/gpt-image-1",
    ),
    "asearch": ("/v1/search/{tool}", "/search", ("query",), {"query": "search"}, "openai/gpt-4o-mini"),
    "atext_completion": (
        "/v1/completions",
        "/completions",
        ("prompt",),
        {"prompt": "complete"},
        "openai/gpt-3.5-turbo-instruct",
    ),
    "atranscription": (
        "/v1/audio/transcriptions",
        "/audio/transcriptions",
        ("file",),
        {},
        "openai/gpt-4o-mini-transcribe",
    ),
    "arerank": (
        "/v1/rerank",
        "/rerank",
        ("query", "documents"),
        {"query": "rank", "documents": ["first"]},
        "cohere/rerank-v4.0",
    ),
    "acompact_responses": (
        "/v1/responses/compact",
        "/responses/compact",
        ("input",),
        {"input": "response"},
        "openai/gpt-4o-mini",
    ),
    "anthropic_messages": (
        "/v1/messages",
        "anthropic_messages",
        ("messages", "max_tokens"),
        {"messages": [{"role": "user", "content": "message"}], "max_tokens": 8},
        "anthropic/claude-haiku-4-5",
    ),
    "agenerate_content": (
        "/v1beta/models/{model}:generateContent",
        "agenerate_content",
        ("contents",),
        {"contents": [{"parts": [{"text": "Gemini"}]}]},
        "gemini/gemini-2.5-flash",
    ),
    "aocr": ("/v1/ocr", "/ocr", ("document",), {}, "mistral/mistral-ocr-latest"),
    "acreate_fine_tuning_job": (
        "/v1/fine_tuning/jobs",
        "/fine_tuning/jobs",
        ("training_file",),
        {"training_file": "file-audit", "model": "gpt-4o-mini"},
        "openai/gpt-4o-mini",
    ),
    "avector_store_search": (
        "/v1/vector_stores/{vector_store_id}/search",
        "avector_store_search",
        ("query",),
        {"query": "vector query"},
        "openai/text-embedding-3-small",
    ),
    "avector_store_file_create": (
        "/v1/vector_stores/{vector_store_id}/files",
        "avector_store_file_create",
        ("file_id",),
        {"file_id": "file-audit"},
        "openai/text-embedding-3-small",
    ),
    "avector_store_file_update": (
        "/v1/vector_stores/{vector_store_id}/files/{file_id}",
        "avector_store_file_update",
        ("attributes",),
        {"attributes": {"source": "audit"}},
        "openai/text-embedding-3-small",
    ),
    "avideo_generation": ("/v1/videos", "/videos", ("prompt",), {"prompt": "video"}, "openai/sora-2"),
    "avideo_remix": (
        "/v1/videos/{video_id}/remix",
        "/videos/{video_id}/remix",
        ("prompt",),
        {"prompt": "remix"},
        "openai/sora-2",
    ),
    "avideo_edit": (
        "/v1/videos/edits",
        "/videos/edits",
        ("prompt",),
        {"prompt": "edit", "video": {"id": "video-audit"}},
        "openai/sora-2",
    ),
    "avideo_extension": (
        "/v1/videos/extensions",
        "/videos/extensions",
        ("prompt", "seconds"),
        {"prompt": "extend", "seconds": 5, "video_id": "video-audit"},
        "openai/sora-2",
    ),
    "avideo_create_character": (
        "/v1/videos/characters",
        "/videos/characters",
        ("name", "video"),
        {"name": "character"},
        "openai/sora-2",
    ),
    "acreate_container": ("/v1/containers", "/containers", ("name",), {"name": "container"}, "openai/gpt-4o-mini"),
    "aupload_container_file": (
        "/v1/containers/container-audit/files",
        "/containers/{container_id}/files",
        ("file",),
        {},
        "openai/gpt-4o-mini",
    ),
    "acreate_agent": (
        "/v1beta/agents",
        "/v1beta/agents",
        ("name",),
        {"name": "agent", "base_agent": "waverunner", "instructions": "You are a helpful assistant."},
        "gemini/gemini-2.5-flash",
    ),
    "acreate_interaction": (
        "/interactions",
        "/interactions",
        ("input", "model"),
        {"input": "interaction"},
        "gemini/gemini-2.5-flash",
    ),
    "acreate_eval": (
        "/v1/evals",
        "/evals",
        ("data_source_config", "testing_criteria"),
        {"data_source_config": {"type": "custom"}, "testing_criteria": [{"type": "string_check"}]},
        "openai/gpt-4o-mini",
    ),
    "acreate_run": (
        "/v1/evals/eval-audit/runs",
        "/evals/{eval_id}/runs",
        ("data_source",),
        {"data_source": {"type": "custom"}},
        "openai/gpt-4o-mini",
    ),
}
_MISSING: Final = tuple(
    route
    for route in _ROUTES
    if route not in {"acreate_fine_tuning_job", "atranscription", "avideo_create_character", "aupload_container_file"}
)
_SKIP_VALID: Final = frozenset(
    {
        "aspeech",
        "asearch",
        "atranscription",
        "aocr",
        "avideo_create_character",
        "aupload_container_file",
        "acreate_fine_tuning_job",
        "acreate_agent",
    }
)
_NO_MODEL_BODY: Final = frozenset(
    {
        "asearch",
        "agenerate_content",
        "avector_store_search",
        "avector_store_file_create",
        "avector_store_file_update",
        "acreate_agent",
    }
)
_DOCUMENTED_GAPS: Final = (
    pytest.param("/v1/audio/transcriptions", {}, 422, ("body", "file"), None, False, id="atranscription-gap"),
    pytest.param(
        "/v1/videos/characters",
        {"name": "character"},
        422,
        ("body", "video"),
        None,
        True,
        id="avideo_create_character-gap",
    ),
    pytest.param(
        "/v1/containers/container-audit/files",
        {},
        400,
        None,
        {"detail": "Missing required 'file' field"},
        False,
        id="aupload_container_file-gap",
    ),
)
_BODIES: Final[dict[str, dict[str, JsonValue]]] = {
    "acompletion": {
        "id": "chatcmpl-$UNIQUE_ID",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "scripted"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    },
    "aembedding": {
        "object": "list",
        "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    },
    "aresponses": {
        "id": "resp_$UNIQUE_ID",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-4o-mini",
        "output": [],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    },
    "acreate_batch": {
        "id": "batch_$UNIQUE_ID",
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "input_file_id": "file-audit",
        "completion_window": "24h",
        "created_at": 1,
        "status": "validating",
    },
    "amoderation": {
        "id": "modr-$UNIQUE_ID",
        "model": "omni-moderation-latest",
        "results": [{"flagged": False, "categories": {}, "category_scores": {}}],
    },
    "aimage_generation": {"created": 1, "data": [{"url": "https://images.invalid/audit.png"}]},
    "arerank": {"id": "rerank-$UNIQUE_ID", "results": [{"index": 0, "relevance_score": 0.5}], "meta": {}},
    "asearch": {"object": "search", "results": []},
    "atext_completion": {
        "id": "cmpl-$UNIQUE_ID",
        "object": "text_completion",
        "created": 1,
        "model": "gpt-3.5-turbo-instruct",
        "choices": [{"text": "scripted", "index": 0, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    },
    "anthropic_messages": {
        "id": "msg-$UNIQUE_ID",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-4-5",
        "content": [{"type": "text", "text": "scripted"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    },
    "agenerate_content": {
        "candidates": [{"content": {"parts": [{"text": "scripted"}], "role": "model"}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
    },
    "acompact_responses": {
        "id": "resp_$UNIQUE_ID",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-4o-mini",
        "output": [],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    },
    "acreate_fine_tuning_job": {
        "id": "ftjob-$UNIQUE_ID",
        "object": "fine_tuning.job",
        "created_at": 1,
        "error": None,
        "fine_tuned_model": None,
        "finished_at": None,
        "hyperparameters": {"n_epochs": "auto"},
        "model": "gpt-4o-mini",
        "organization_id": "org-audit",
        "result_files": [],
        "seed": 1,
        "status": "validating_files",
        "trained_tokens": None,
        "training_file": "file-audit",
        "validation_file": None,
    },
    "avector_store_search": {"object": "vector_store.search_results.page", "search_query": "vector query", "data": []},
    "avector_store_file_create": {
        "id": "file-audit",
        "object": "vector_store.file",
        "created_at": 1,
        "usage_bytes": 0,
        "vector_store_id": "vs-audit",
        "status": "completed",
        "last_error": None,
        "attributes": {},
    },
    "avector_store_file_update": {
        "id": "file-audit",
        "object": "vector_store.file",
        "created_at": 1,
        "usage_bytes": 0,
        "vector_store_id": "vs-audit",
        "status": "completed",
        "last_error": None,
        "attributes": {"source": "audit"},
    },
    "avideo_generation": {
        "id": "video-audit-generation",
        "object": "video",
        "created_at": 1,
        "status": "queued",
        "model": "sora-2",
    },
    "avideo_remix": {
        "id": "video-audit-remix",
        "object": "video",
        "created_at": 1,
        "status": "queued",
        "model": "sora-2",
        "remixed_from_video_id": "video-audit",
    },
    "avideo_extension": {
        "id": "video-audit-extension",
        "object": "video",
        "created_at": 1,
        "status": "queued",
        "model": "sora-2",
        "seconds": "5",
    },
    "acreate_container": {
        "id": "container-audit",
        "object": "container",
        "created_at": 1,
        "status": "running",
        "name": "container",
    },
    "acreate_agent": {"id": "agent-$UNIQUE_ID", "name": "agent"},
    "acreate_interaction": {
        "id": "interaction-$UNIQUE_ID",
        "object": "interaction",
        "status": "completed",
        "model": "gemini-2.5-flash",
    },
    "acreate_eval": {
        "id": "eval-$UNIQUE_ID",
        "object": "eval",
        "created_at": 1,
        "data_source_config": {"type": "custom"},
        "testing_criteria": [{"type": "string_check"}],
    },
    "acreate_run": {
        "id": "evalrun-$UNIQUE_ID",
        "object": "eval.run",
        "created_at": 1,
        "status": "queued",
        "data_source": {"type": "custom"},
        "eval_id": "eval-audit",
    },
    "avideo_edit": {
        "id": "video-audit-edit",
        "object": "video",
        "created_at": 1,
        "status": "queued",
        "model": "sora-2",
    },
    "avideo_create_character": {
        "id": "character-audit",
        "object": "character",
        "created_at": 1,
        "name": "character",
    },
    "aupload_container_file": {
        "id": "container-file-audit",
        "object": "container.file",
        "container_id": "container-audit",
        "created_at": 1,
        "path": "notes.txt",
        "source": "user",
    },
}
_STREAM_RESPONSES: Final[dict[str, SseResponse]] = {
    "acompletion": SseResponse(
        content_type="text/event-stream",
        frames=(
            (
                'data: {"id":"chatcmpl-$UNIQUE_ID","object":"chat.completion.chunk","created":1,'
                '"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-$UNIQUE_ID","object":"chat.completion.chunk","created":1,'
                '"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"streamed "},"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-$UNIQUE_ID","object":"chat.completion.chunk","created":1,'
                '"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"response"},"finish_reason":null}]}'
            ),
            (
                'data: {"id":"chatcmpl-$UNIQUE_ID","object":"chat.completion.chunk","created":1,'
                '"model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
            ),
            "data: [DONE]",
        ),
    ),
    "aresponses": SseResponse(
        content_type="text/event-stream",
        frames=(
            (
                "event: response.created\n"
                'data: {"type":"response.created","response":{"id":"resp_$REQUEST_ID","object":"response",'
                '"created_at":1,"status":"in_progress","model":"gpt-4o-mini","output":[],"usage":null}}'
            ),
            (
                "event: response.output_item.added\n"
                'data: {"type":"response.output_item.added","output_index":0,'
                '"item":{"type":"message","id":"msg_$REQUEST_ID","status":"in_progress",'
                '"role":"assistant","content":[]}}'
            ),
            (
                "event: response.content_part.added\n"
                'data: {"type":"response.content_part.added","item_id":"msg_$REQUEST_ID",'
                '"output_index":0,"content_index":0,'
                '"part":{"type":"output_text","text":"","annotations":[]}}'
            ),
            (
                "event: response.output_text.delta\n"
                'data: {"type":"response.output_text.delta","item_id":"msg_$REQUEST_ID",'
                '"output_index":0,"content_index":0,"delta":"streamed "}'
            ),
            (
                "event: response.output_text.delta\n"
                'data: {"type":"response.output_text.delta","item_id":"msg_$REQUEST_ID",'
                '"output_index":0,"content_index":0,"delta":"response"}'
            ),
            (
                "event: response.output_text.done\n"
                'data: {"type":"response.output_text.done","item_id":"msg_$REQUEST_ID",'
                '"output_index":0,"content_index":0,"text":"streamed response"}'
            ),
            (
                "event: response.content_part.done\n"
                'data: {"type":"response.content_part.done","item_id":"msg_$REQUEST_ID",'
                '"output_index":0,"content_index":0,'
                '"part":{"type":"output_text","text":"streamed response","annotations":[]}}'
            ),
            (
                "event: response.output_item.done\n"
                'data: {"type":"response.output_item.done","output_index":0,'
                '"item":{"type":"message","id":"msg_$REQUEST_ID","status":"completed",'
                '"role":"assistant","content":[{"type":"output_text","text":"streamed response",'
                '"annotations":[]}]}}'
            ),
            (
                "event: response.completed\n"
                'data: {"type":"response.completed","response":{"id":"resp_$REQUEST_ID",'
                '"object":"response","created_at":1,"status":"completed","model":"gpt-4o-mini",'
                '"output":[{"id":"msg_$REQUEST_ID","type":"message","status":"completed",'
                '"role":"assistant","content":[{"type":"output_text","text":"streamed response",'
                '"annotations":[]}]}],"usage":{"input_tokens":1,"output_tokens":2}}}'
            ),
        ),
    ),
    "anthropic_messages": SseResponse(
        content_type="text/event-stream",
        frames=(
            (
                "event: message_start\n"
                'data: {"type":"message_start","message":{"id":"msg_$REQUEST_ID","type":"message",'
                '"role":"assistant","model":"claude-haiku-4-5","content":[],"stop_reason":null,'
                '"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":0}}}'
            ),
            (
                "event: content_block_start\n"
                'data: {"type":"content_block_start","index":0,'
                '"content_block":{"type":"text","text":""}}'
            ),
            (
                "event: content_block_delta\n"
                'data: {"type":"content_block_delta","index":0,'
                '"delta":{"type":"text_delta","text":"streamed "}}'
            ),
            (
                "event: content_block_delta\n"
                'data: {"type":"content_block_delta","index":0,'
                '"delta":{"type":"text_delta","text":"response"}}'
            ),
            ('event: content_block_stop\ndata: {"type":"content_block_stop","index":0}'),
            (
                "event: message_delta\n"
                'data: {"type":"message_delta","delta":{"stop_reason":"end_turn",'
                '"stop_sequence":null},"usage":{"output_tokens":2}}'
            ),
            'event: message_stop\ndata: {"type":"message_stop"}',
        ),
    ),
}


class _Observations:
    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")
        self.items: tuple[dict[str, JsonValue], ...] = ()

    def read(self) -> tuple[dict[str, JsonValue], ...]:
        with httpx.Client(timeout=10, trust_env=False) as client:
            payload: Final = JSON_OBJECT.validate_python(
                client.get(f"{self.url}/__observations?include_method=true").json()
            )
        requests: Final = payload.get("requests")
        assert isinstance(requests, list)
        self.items = (*self.items, *(object_value(item) for item in requests if isinstance(item, dict)))
        return self.items

    def for_scenario(self, identity: str) -> tuple[dict[str, JsonValue], ...]:
        return tuple(item for item in self.items if f"/{identity}/" in str(item.get("path")))

    def provider_calls(self, identity: str) -> tuple[dict[str, JsonValue], ...]:
        calls: Final = tuple(
            item
            for item in self.for_scenario(identity)
            if not (item.get("method") == "GET" and str(item.get("path", "")).endswith(("/v1/models", "/models")))
        )
        return calls


def _response(
    route: str,
    *,
    streaming: bool = False,
) -> BinaryResponse | JsonResponse | RoutedResponse | SseResponse:
    if route == "aspeech":
        return BinaryResponse(content_type="audio/mpeg", length=16)
    if streaming:
        return _STREAM_RESPONSES[route]
    if route == "acreate_fine_tuning_job":
        return RoutedResponse(
            content_type="application/x-routed",
            routes={
                "POST /files": JsonResponse(
                    content_type="application/json",
                    body={
                        "id": "file-training",
                        "object": "file",
                        "purpose": "fine-tune",
                        "filename": "training.jsonl",
                        "bytes": 90,
                        "created_at": 1,
                        "status": "processed",
                    },
                ),
                "POST /fine_tuning/jobs": JsonResponse(
                    content_type="application/json",
                    body=_BODIES[route],
                ),
            },
        )
    if route == "aupload_container_file":
        return RoutedResponse(
            content_type="application/x-routed",
            routes={
                "POST /containers": JsonResponse(
                    content_type="application/json",
                    body=_BODIES["acreate_container"],
                ),
                "POST /containers/container-audit/files": JsonResponse(
                    content_type="application/json",
                    body=_BODIES[route],
                ),
            },
        )
    return JsonResponse(
        content_type="application/json", body=_BODIES.get(route, {"id": "audit-$UNIQUE_ID", "object": "audit_response"})
    )


def _assert_scripted_response(route: str, caller: dict[str, JsonValue]) -> None:
    scripted: Final = _BODIES[route]
    if route == "acompletion":
        expected_choices: Final = scripted["choices"]
        actual_choices: Final = caller["choices"]
        assert isinstance(expected_choices, list) and isinstance(actual_choices, list)
        expected_choice: Final = object_value(expected_choices[0])
        actual_choice: Final = object_value(actual_choices[0])
        assert object_value(expected_choice["message"])["content"] == object_value(actual_choice["message"])["content"]
    elif route == "atext_completion":
        expected_choices = scripted["choices"]
        actual_choices = caller["choices"]
        assert isinstance(expected_choices, list) and isinstance(actual_choices, list)
        expected_choice = object_value(expected_choices[0])
        actual_choice = object_value(actual_choices[0])
        assert actual_choice["text"] == expected_choice["text"]
    elif route == "aembedding":
        expected_data: Final = scripted["data"]
        actual_data: Final = caller["data"]
        assert isinstance(expected_data, list) and isinstance(actual_data, list)
        expected_item: Final = object_value(expected_data[0])
        actual_item: Final = object_value(actual_data[0])
        assert actual_item["embedding"] == expected_item["embedding"]
    elif route == "amoderation":
        expected_results: Final = scripted["results"]
        actual_results: Final = caller["results"]
        assert isinstance(expected_results, list) and isinstance(actual_results, list)
        expected_result: Final = object_value(expected_results[0])
        actual_result: Final = object_value(actual_results[0])
        assert actual_result["flagged"] is expected_result["flagged"]
    elif route == "aimage_generation":
        expected_data = scripted["data"]
        actual_data = caller["data"]
        assert isinstance(expected_data, list) and isinstance(actual_data, list)
        expected_item = object_value(expected_data[0])
        actual_item = object_value(actual_data[0])
        assert actual_item["url"] == expected_item["url"]
    elif route == "arerank":
        expected_results = scripted["results"]
        actual_results = caller["results"]
        assert isinstance(expected_results, list) and isinstance(actual_results, list)
        expected_result = object_value(expected_results[0])
        actual_result = object_value(actual_results[0])
        assert actual_result["index"] == expected_result["index"]
        assert actual_result["relevance_score"] == expected_result["relevance_score"]
    elif route == "anthropic_messages":
        expected_content_list: Final = scripted["content"]
        actual_content_list: Final = caller["content"]
        assert isinstance(expected_content_list, list) and isinstance(actual_content_list, list)
        expected_content: Final = object_value(expected_content_list[0])
        actual_content: Final = object_value(actual_content_list[0])
        assert actual_content["text"] == expected_content["text"]
    elif route == "agenerate_content":
        expected_candidates: Final = scripted["candidates"]
        actual_candidates: Final = caller["candidates"]
        assert isinstance(expected_candidates, list) and isinstance(actual_candidates, list)
        expected_candidate: Final = object_value(expected_candidates[0])
        actual_candidate: Final = object_value(actual_candidates[0])
        expected_parts: Final = object_value(expected_candidate["content"])["parts"]
        actual_parts: Final = object_value(actual_candidate["content"])["parts"]
        assert isinstance(expected_parts, list) and isinstance(actual_parts, list)
        expected_part: Final = object_value(expected_parts[0])
        actual_part: Final = object_value(actual_parts[0])
        assert actual_part["text"] == expected_part["text"]
    elif route in {
        "aresponses",
        "acreate_batch",
        "acompact_responses",
        "avector_store_search",
        "avector_store_file_create",
        "avector_store_file_update",
        "avideo_generation",
        "avideo_remix",
        "avideo_extension",
        "avideo_edit",
        "avideo_create_character",
        "aupload_container_file",
        "acreate_container",
        "acreate_agent",
        "acreate_interaction",
        "acreate_eval",
        "acreate_run",
    }:
        for field in ("status", "object"):
            if field in scripted:
                assert caller.get(field) == scripted[field]
        if "id" in scripted:
            actual_id: Final = caller.get("id")
            expected_id: Final = str(scripted["id"])
            assert isinstance(actual_id, str) and actual_id
            if route in {"avideo_generation", "avideo_remix", "avideo_extension", "avideo_edit"}:
                assert decode_video_id_with_provider(actual_id)["video_id"] == expected_id
            elif route == "acreate_container":
                assert ResponsesAPIRequestUtils.decode_container_id_to_original(actual_id) == expected_id
            else:
                expected_prefix: Final = expected_id.split("$UNIQUE_ID", maxsplit=1)[0]
                assert actual_id.startswith(expected_prefix)
        if route in {"avideo_create_character", "acreate_agent"}:
            assert caller.get("name") == scripted["name"]
        if route == "aupload_container_file":
            assert caller.get("container_id") == scripted["container_id"]


def _register(
    scenario: Scenario,
    route: str,
    *,
    streaming: bool = False,
    deployment_params: dict[str, JsonValue] | None = None,
    provider_model: str | None = None,
    response_route: str | None = None,
) -> tuple[str, str, ScenarioHandle]:
    identity: Final = f"audit-{route}-{uuid.uuid4().hex}"
    handle: Final = register_scenario(identity, _response(response_route or route, streaming=streaming))
    scenario.cleanups.callback(delete_scenario, handle)
    model: Final = (
        ""
        if route == "asearch"
        else scenario.model(
            model=provider_model or _ROUTES[route][4],
            api_base=handle.api_base(),
            api_key=identity,
            **(deployment_params or {}),
        )
    )
    return model, identity, handle


def _path(template: str, model: str, tool: str, store: str) -> str:
    video: Final = (
        encode_video_id_with_provider("video-audit", "openai", model_id=model) if "{video_id}" in template else ""
    )
    return (
        template.replace("{model}", model)
        .replace("{tool}", tool)
        .replace("{vector_store_id}", store)
        .replace("{file_id}", "file-audit")
        .replace("{video_id}", video)
    )


def _store(gateway: Gateway, scenario: Scenario, model: str, identity: str, handle: ScenarioHandle, store: str) -> None:
    response: Final = gateway.request(
        "POST",
        "/vector_store/new",
        {
            "vector_store_id": store,
            "custom_llm_provider": "openai",
            "litellm_params": {"model": model, "api_base": handle.api_base(), "api_key": identity},
        },
    )
    assert response.status_code == 200, response.text
    scenario.cleanups.callback(gateway.post, "/vector_store/delete", {"vector_store_id": store})


def _search_tool(gateway: Gateway, scenario: Scenario, identity: str, handle: ScenarioHandle) -> str:
    name: Final = f"audit-search-{uuid.uuid4().hex}"
    created: Final = gateway.post(
        "/search_tools",
        {
            "search_tool": {
                "search_tool_name": name,
                "litellm_params": {"search_provider": "exa_ai", "api_key": identity, "api_base": handle.api_base()},
            }
        },
    )
    scenario.cleanups.callback(
        lambda tool_id: gateway.request("DELETE", f"/search_tools/{tool_id}"), str(created["search_tool_id"])
    )
    return name


def _error(route: str, parameter: str) -> dict[str, JsonValue]:
    message: Final = f"{route}: Missing required parameter: '{parameter}'."
    return (
        {"type": "error", "error": {"type": "invalid_request_error", "message": message}}
        if route == "anthropic_messages"
        else {"error": {"message": message, "type": "invalid_request_error", "param": parameter, "code": "400"}}
    )


def _missing_response(
    gateway: Gateway,
    path: str,
    body: dict[str, JsonValue],
    expected: dict[str, JsonValue],
) -> httpx.Response:
    response: Final = _post(gateway, path, body)
    assert response.status_code == 400, response.text
    assert response.json() == expected, response.text
    return response


def _post(gateway: Gateway, path: str, body: dict[str, JsonValue]) -> httpx.Response:
    return gateway.client.post(path, json=body, headers={"Authorization": f"Bearer {gateway.key}"})


def _observed(
    buffer: _Observations,
    identity: str,
    expected_count: int = 1,
) -> tuple[dict[str, JsonValue], ...]:
    eventually(buffer.read, lambda _items: len(buffer.for_scenario(identity)) == expected_count, seconds=20)
    return buffer.for_scenario(identity)


def _stream_event_payloads(lines: tuple[str, ...]) -> tuple[dict[str, JsonValue], ...]:
    return tuple(JSON_OBJECT.validate_json(line.removeprefix("data: ")) for line in lines if line.startswith("data: {"))


def _stream_event_names(lines: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(line.removeprefix("event: ") for line in lines if line.startswith("event: "))


def _stream_event_text(route: str, event: dict[str, JsonValue]) -> str:
    if route == "acompletion":
        choices: Final = event.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        delta: Final = object_value(object_value(choices[0]).get("delta"))
        content: Final = delta.get("content")
        return content if isinstance(content, str) else ""
    if route == "aresponses" and event.get("type") == "response.output_text.delta":
        delta: Final = event.get("delta")
        return delta if isinstance(delta, str) else ""
    if route == "anthropic_messages" and event.get("type") == "content_block_delta":
        delta: Final = object_value(event.get("delta"))
        if delta.get("type") != "text_delta":
            return ""
        text: Final = delta.get("text")
        return text if isinstance(text, str) else ""
    return ""


def _assembled_stream_text(route: str, lines: tuple[str, ...]) -> str:
    return "".join(_stream_event_text(route, event) for event in _stream_event_payloads(lines))


@pytest.mark.parametrize("route", _MISSING, ids=_MISSING)
def test_added_required_fields_return_exact_400(gateway: Gateway, route: str) -> None:
    template, error_route, fields, valid_body, _provider_model = _ROUTES[route]
    with gateway.scenario() as scenario:
        model, identity, handle = _register(scenario, route)
        tool: Final = _search_tool(gateway, scenario, identity, handle) if route == "asearch" else ""
        store: Final = f"vs-{uuid.uuid4().hex}"
        if route.startswith("avector_store_"):
            _store(gateway, scenario, model, identity, handle, store)
        path: Final = _path(template, model, tool, store)
        observations: Final = _Observations(gateway.upstream_url)
        for field in fields:
            missing_body: Final = {
                key: value
                for key, value in {**valid_body, **({"model": model} if route not in _NO_MODEL_BODY else {})}.items()
                if key != field
            }
            body: Final = {
                **missing_body,
                **(
                    {
                        "video": {
                            "id": encode_video_id_with_provider("video-audit", "openai", model_id=model),
                        }
                    }
                    if route == "avideo_edit"
                    else {}
                ),
            }
            expected: Final = _error(error_route, field)
            response: Final = _missing_response(gateway, path, body, expected)
            assert response.status_code == 400, f"{route}.{field}: {response.text}"
            assert response.json() == expected, response.text
            observations.read()
            assert observations.provider_calls(identity) == ()


@pytest.mark.parametrize(
    ("path", "body", "expected_status", "expected_loc", "expected_body", "as_form"), _DOCUMENTED_GAPS
)
def test_unchanged_from_base_documented_gaps(
    gateway: Gateway,
    path: str,
    body: dict[str, JsonValue],
    expected_status: int,
    expected_loc: tuple[str, str] | None,
    expected_body: dict[str, JsonValue] | None,
    as_form: bool,
) -> None:
    response: Final = (
        gateway.client.post(path, data=body, headers={"Authorization": f"Bearer {gateway.key}"})
        if as_form
        else _post(gateway, path, body)
    )
    assert response.status_code == expected_status, response.text
    if expected_body is not None:
        assert response.json() == expected_body, response.text
    else:
        assert expected_loc is not None
        detail: Final = JSON_OBJECT.validate_python(response.json()).get("detail")
        assert isinstance(detail, list) and detail, response.text
        assert object_value(detail[0]).get("loc") == list(expected_loc), response.text


def test_fine_tuning_missing_training_file_returns_422_without_upstream_call(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model, identity, _handle = _register(scenario, "acreate_fine_tuning_job")
        response: Final = _post(gateway, "/v1/fine_tuning/jobs", {"model": model})
        assert response.status_code == 422, response.text
        detail: Final = JSON_OBJECT.validate_python(response.json()).get("detail")
        assert isinstance(detail, list) and detail, response.text
        assert object_value(detail[0]).get("loc") == ["body", "training_file"], response.text
        observations: Final = _Observations(gateway.upstream_url)
        observations.read()
        assert observations.provider_calls(identity) == ()


@pytest.mark.parametrize(
    "route",
    tuple(name for name in _ROUTES if name not in _SKIP_VALID),
    ids=tuple(name for name in _ROUTES if name not in _SKIP_VALID),
)
def test_valid_required_fields_reach_upstream(gateway: Gateway, route: str) -> None:
    template, _error_route, fields, body, provider_model = _ROUTES[route]
    with gateway.scenario() as scenario:
        model, identity, handle = _register(scenario, route)
        store: Final = f"vs-{uuid.uuid4().hex}"
        if route.startswith("avector_store_"):
            _store(gateway, scenario, model, identity, handle, store)
        path: Final = _path(template, model, "", store)
        request_body: Final = {
            **body,
            **(
                {
                    "video": {
                        "id": encode_video_id_with_provider("video-audit", "openai", model_id=model),
                    }
                }
                if route == "avideo_edit"
                else {}
            ),
            **({"model": model} if route not in _NO_MODEL_BODY else {}),
            **({"input": f"response-{identity}"} if route == "aresponses" else {}),
        }
        response: Final = eventually(
            lambda: _post(gateway, path, request_body),
            lambda result: not (result.status_code == 400 and "Invalid model name" in result.text),
            seconds=30,
        )
        assert response.status_code == 200, response.text
        caller: Final = JSON_OBJECT.validate_python(response.json())
        _assert_scripted_response(route, caller)
        matches: Final = _observed(_Observations(gateway.upstream_url), identity)
        outbound: Final = object_value(matches[0]["body"])
        expected_model: Final = provider_model.removeprefix("gemini/") if route == "acreate_interaction" else model
        assert all(
            outbound.get(field) == (expected_model if field == "model" else request_body[field]) for field in fields
        ), matches
        if route == "avideo_edit":
            assert outbound.get("video") == {"id": "video-audit"}, matches


def test_anthropic_messages_uses_deployment_max_tokens_default(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model, identity, _handle = _register(
            scenario,
            "anthropic_messages",
            deployment_params={"max_tokens": 32},
        )
        response: Final = _post(
            gateway,
            "/v1/messages",
            {"model": model, "messages": [{"role": "user", "content": "default max tokens"}]},
        )
        assert response.status_code == 200, response.text
        _assert_scripted_response("anthropic_messages", JSON_OBJECT.validate_python(response.json()))
        observations: Final = _Observations(gateway.upstream_url)
        captured: Final = eventually(
            observations.read,
            lambda _items: any(item.get("method") == "POST" for item in observations.for_scenario(identity)),
            seconds=20,
        )
        provider_requests: Final = tuple(
            item for item in observations.for_scenario(identity) if item.get("method") == "POST"
        )
        assert len(provider_requests) == 1, captured
        outbound: Final = object_value(provider_requests[0]["body"])
        assert outbound.get("max_tokens") == 32, provider_requests


def test_anthropic_messages_uses_router_wide_max_tokens_default(gateway: Gateway, tmp_path: Path) -> None:
    with gateway.scenario() as scenario:
        identity: Final = f"router-default-{uuid.uuid4().hex}"
        handle: Final = register_scenario(identity, _response("anthropic_messages"))
        scenario.cleanups.callback(delete_scenario, handle)
        config: Final = tmp_path / "router-default.yaml"
        config.write_text(
            json.dumps(
                {
                    "model_list": [
                        {
                            "model_name": "router-default-anthropic",
                            "litellm_params": {
                                "model": "anthropic/claude-haiku-4-5",
                                "api_base": handle.api_base(),
                                "api_key": identity,
                            },
                        }
                    ],
                    "router_settings": {"default_litellm_params": {"max_tokens": 32}},
                }
            ),
            encoding="utf-8",
        )
        with owned_proxy_process(gateway, tmp_path, {}, config=config) as owned:
            candidate: Final = Gateway(owned.gateway.client, owned.gateway.key, gateway.upstream_url)
            response: Final = _post(
                candidate,
                "/v1/messages",
                {"model": "router-default-anthropic", "messages": [{"role": "user", "content": "router default"}]},
            )
            assert response.status_code == 200, response.text
            _assert_scripted_response("anthropic_messages", JSON_OBJECT.validate_python(response.json()))
            observations: Final = _Observations(gateway.upstream_url)
            captured: Final = eventually(
                observations.read,
                lambda _items: any(item.get("method") == "POST" for item in observations.for_scenario(identity)),
                seconds=20,
            )
            provider_requests: Final = tuple(
                item for item in observations.for_scenario(identity) if item.get("method") == "POST"
            )
            assert len(provider_requests) == 1, captured
            outbound: Final = object_value(provider_requests[0]["body"])
            assert outbound.get("max_tokens") == 32, provider_requests


def test_rerank_uses_router_wide_documents_default(gateway: Gateway, tmp_path: Path) -> None:
    with gateway.scenario() as scenario:
        identity: Final = f"router-default-rerank-{uuid.uuid4().hex}"
        handle: Final = register_scenario(identity, _response("arerank"))
        scenario.cleanups.callback(delete_scenario, handle)
        config: Final = tmp_path / "router-default-rerank.yaml"
        config.write_text(
            json.dumps(
                {
                    "model_list": [
                        {
                            "model_name": "router-default-rerank",
                            "litellm_params": {
                                "model": "cohere/rerank-v4.0",
                                "api_base": handle.api_base(),
                                "api_key": identity,
                            },
                        }
                    ],
                    "router_settings": {"default_litellm_params": {"documents": ["router default document"]}},
                }
            ),
            encoding="utf-8",
        )
        with owned_proxy_process(gateway, tmp_path, {}, config=config) as owned:
            candidate: Final = Gateway(owned.gateway.client, owned.gateway.key, gateway.upstream_url)
            response: Final = _post(
                candidate,
                "/rerank",
                {"model": "router-default-rerank", "query": "which document?"},
            )
            assert response.status_code == 200, response.text
            observations: Final = _Observations(gateway.upstream_url)
            captured: Final = eventually(
                observations.read,
                lambda _items: any(item.get("method") == "POST" for item in observations.for_scenario(identity)),
                seconds=20,
            )
            provider_requests: Final = tuple(
                item for item in observations.for_scenario(identity) if item.get("method") == "POST"
            )
            assert len(provider_requests) == 1, captured
            outbound: Final = object_value(provider_requests[0]["body"])
            assert outbound.get("documents") == ["router default document"], provider_requests


def test_anthropic_messages_explicit_null_reaches_upstream(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model, identity, _handle = _register(
            scenario,
            "anthropic_messages",
            deployment_params={"max_tokens": 32},
            provider_model="openai/gpt-4o-mini",
            response_route="acompletion",
        )
        response: Final = _post(
            gateway,
            "/v1/messages",
            {"model": model, "messages": [{"role": "user", "content": "null max tokens"}], "max_tokens": None},
        )
        assert response.status_code == 200, response.text
        observations: Final = _Observations(gateway.upstream_url)
        captured: Final = eventually(
            observations.read,
            lambda _items: any(item.get("method") == "POST" for item in observations.for_scenario(identity)),
            seconds=20,
        )
        provider_requests: Final = tuple(
            item for item in observations.for_scenario(identity) if item.get("method") == "POST"
        )
        assert len(provider_requests) == 1, captured


def test_image_generation_null_prompt_reaches_upstream(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model, identity, _handle = _register(scenario, "aimage_generation")
        response: Final = _post(gateway, "/v1/images/generations", {"model": model, "prompt": None})
        assert response.status_code == 200, response.text
        _assert_scripted_response("aimage_generation", JSON_OBJECT.validate_python(response.json()))
        observations: Final = _Observations(gateway.upstream_url)
        captured: Final = eventually(
            observations.read,
            lambda _items: any(item.get("method") == "POST" for item in observations.for_scenario(identity)),
            seconds=20,
        )
        provider_requests: Final = tuple(
            item for item in observations.for_scenario(identity) if item.get("method") == "POST"
        )
        assert len(provider_requests) == 1, captured
        outbound: Final = object_value(provider_requests[0]["body"])
        assert "prompt" in outbound and outbound["prompt"] is None, provider_requests


def test_valid_agent_creation_reaches_upstream(gateway: Gateway) -> None:
    body: Final = _ROUTES["acreate_agent"][3]
    with gateway.scenario() as scenario:
        identity: Final = f"audit-acreate_agent-{uuid.uuid4().hex}"
        handle: Final = register_scenario(identity, _response("acreate_agent"))
        scenario.cleanups.callback(delete_scenario, handle)
        request_body: Final = {
            **body,
            "litellm_params_template": {"api_base": handle.api_base(), "api_key": identity},
        }
        response: Final = gateway.request("POST", "/v1beta/agents", request_body)
        assert response.status_code == 200, response.text
        caller: Final = JSON_OBJECT.validate_python(response.json())
        _assert_scripted_response("acreate_agent", caller)
        matches: Final = _observed(_Observations(gateway.upstream_url), identity)
        outbound: Final = object_value(matches[0]["body"])
        assert outbound == {
            "name": "agent",
            "base_agent": "waverunner",
            "instructions": "You are a helpful assistant.",
        }, matches


def test_valid_speech_returns_binary_audio_and_reaches_upstream(gateway: Gateway) -> None:
    template, _error_route, fields, body, _provider_model = _ROUTES["aspeech"]
    with gateway.scenario() as scenario:
        model, identity, _handle = _register(scenario, "aspeech")
        request_body: Final = {**body, "model": model}
        response: Final = _post(gateway, template, request_body)
        assert response.status_code == 200, response.text
        assert response.headers.get("content-type") == "audio/mpeg"
        assert response.content == b"\x00" * 16
        matches: Final = _observed(_Observations(gateway.upstream_url), identity)
        outbound: Final = object_value(matches[0]["body"])
        assert all(outbound.get(field) == request_body[field] for field in fields), matches
        assert outbound.get("voice") == request_body["voice"], matches


def test_valid_video_character_request_reaches_upstream(gateway: Gateway) -> None:
    template, _error_route, _fields, _body, _provider_model = _ROUTES["avideo_create_character"]
    with gateway.scenario() as scenario:
        model, identity, _handle = _register(scenario, "avideo_create_character")
        path: Final = _path(template, model, "", "")
        response: Final = gateway.request_multipart(
            path,
            {"name": "character", "model": model},
            {"video": ("character.mp4", b"scripted-video", "video/mp4")},
        )
        assert response.status_code == 200, response.text
        _assert_scripted_response("avideo_create_character", JSON_OBJECT.validate_python(response.json()))
        matches: Final = _observed(_Observations(gateway.upstream_url), identity)
        outbound: Final = object_value(matches[0]["body"])
        assert outbound.get("video") == {
            "filename": "character.mp4",
            "content_type": "video/mp4",
        }, matches
        assert outbound.get("name") == "character", matches


def test_valid_container_file_upload_reaches_upstream(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model, identity, _handle = _register(scenario, "aupload_container_file")
        container_response: Final = gateway.request("POST", "/v1/containers", {"model": model, "name": "container"})
        assert container_response.status_code == 200, container_response.text
        container: Final = object_value(JSON_OBJECT.validate_python(container_response.json()))
        assert container.get("object") == "container", container
        container_id: Final = string_value(container["id"])
        response: Final = gateway.request_multipart(
            f"/v1/containers/{container_id}/files",
            {},
            {"file": ("notes.txt", b"container file contents", "text/plain")},
        )
        assert response.status_code == 200, response.text
        _assert_scripted_response("aupload_container_file", JSON_OBJECT.validate_python(response.json()))
        matches: Final = _observed(_Observations(gateway.upstream_url), identity, expected_count=2)
        create_request: Final = object_value(matches[0]["body"])
        upload_request: Final = object_value(matches[1]["body"])
        assert str(matches[0]["path"]).endswith("/containers"), matches
        assert create_request == {"name": "container"}, matches
        assert str(matches[1]["path"]).endswith("/containers/container-audit/files"), matches
        assert upload_request == {
            "file": {
                "filename": "notes.txt",
                "content_type": "text/plain",
            }
        }, matches


@pytest.mark.parametrize(
    ("route", "expected_text"),
    (
        pytest.param("acompletion", "streamed response", id="chat-completions"),
        pytest.param("anthropic_messages", "streamed response", id="anthropic-messages"),
        pytest.param("aresponses", "streamed response", id="responses"),
    ),
)
def test_valid_streaming_required_fields_reach_upstream(
    gateway: Gateway,
    route: str,
    expected_text: str,
) -> None:
    template, _error_route, fields, body, _provider_model = _ROUTES[route]
    with gateway.scenario() as scenario:
        model, identity, _handle = _register(scenario, route, streaming=True)
        request_body: Final = {
            **body,
            **(
                {"messages": [{"role": "user", "content": f"stream-{identity}"}]}
                if route in {"acompletion", "anthropic_messages"}
                else {}
            ),
            **({"input": f"response-{identity}"} if route == "aresponses" else {}),
            "model": model,
            "stream": True,
        }
        headers: Final = {"Authorization": f"Bearer {gateway.key}"}
        with gateway.client.stream("POST", template, json=request_body, headers=headers) as response:
            assert response.status_code == 200, response.read().decode()
            lines: Final = tuple(response.iter_lines())
        assert _assembled_stream_text(route, lines) == expected_text, lines
        if route == "anthropic_messages":
            events: Final = _stream_event_names(lines)
            payloads: Final = _stream_event_payloads(lines)
            assert events[-1:] == ("message_stop",), lines
            assert payloads and payloads[-1].get("type") == "message_stop", lines
        matches: Final = _observed(_Observations(gateway.upstream_url), identity)
        outbound: Final = object_value(matches[0]["body"])
        assert outbound.get("stream") is True, matches
        assert all(outbound.get(field) == request_body[field] for field in fields), matches


@pytest.mark.parametrize("client_kind", ("sync", "async"), ids=("sync", "async"))
def test_openai_sdk_missing_moderations_input_returns_bad_request(gateway: Gateway, client_kind: str) -> None:
    with gateway.scenario() as scenario:
        model, identity, _handle = _register(scenario, "amoderation")
        _missing_response(gateway, "/v1/moderations", {"model": model}, _error("/moderations", "input"))
        if client_kind == "sync":
            with OpenAI(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0) as client:
                with pytest.raises(BadRequestError) as raised:
                    client.post("/v1/moderations", body={"model": model}, cast_to=httpx.Response)
        else:

            async def request() -> None:
                async with AsyncOpenAI(
                    base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0
                ) as client:
                    await client.post("/v1/moderations", body={"model": model}, cast_to=httpx.Response)

            with pytest.raises(BadRequestError) as raised:
                asyncio.run(request())
        assert raised.value.status_code == 400
        assert raised.value.response.json() == _error("/moderations", "input")
        observations: Final = _Observations(gateway.upstream_url)
        observations.read()
        assert observations.provider_calls(identity) == ()


def test_default_search_model_uses_query_without_model(gateway: Gateway, tmp_path: Path) -> None:
    with gateway.scenario() as scenario:
        identity: Final = f"default-search-{uuid.uuid4().hex}"
        handle: Final = register_scenario(identity, _response("asearch"))
        scenario.cleanups.callback(delete_scenario, handle)
        config: Final = tmp_path / "search.yaml"
        config.write_text(
            json.dumps(
                {
                    "model_list": [],
                    "general_settings": {"completion_model": "exa-search"},
                    "search_tools": [
                        {
                            "search_tool_name": "exa-search",
                            "litellm_params": {
                                "search_provider": "exa_ai",
                                "api_key": identity,
                                "api_base": handle.api_base(),
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
            candidate: Final = Gateway(owned.gateway.client, owned.gateway.key, gateway.upstream_url)
            query: Final = f"query-{uuid.uuid4().hex}"
            response: Final = eventually(
                lambda: _post(candidate, "/v1/search", {"query": query}),
                lambda result: not (result.status_code == 400 and "Invalid model name" in result.text),
                seconds=30,
            )
            assert (
                response.status_code == 200 and JSON_OBJECT.validate_python(response.json()).get("object") == "search"
            ), response.text
            outbound: Final = object_value(_observed(_Observations(gateway.upstream_url), identity)[0]["body"])
            assert outbound.get("query") == query, outbound


def test_interaction_without_model_uses_completion_model(gateway: Gateway, tmp_path: Path) -> None:
    with gateway.scenario() as scenario:
        identity: Final = f"default-interaction-{uuid.uuid4().hex}"
        handle: Final = register_scenario(identity, _response("acreate_interaction"))
        scenario.cleanups.callback(delete_scenario, handle)
        config: Final = tmp_path / "interaction.yaml"
        config.write_text(
            json.dumps(
                {
                    "model_list": [
                        {
                            "model_name": "interaction-default",
                            "litellm_params": {
                                "model": "gemini/gemini-2.5-flash",
                                "api_base": handle.api_base(),
                                "api_key": identity,
                            },
                        }
                    ],
                    "general_settings": {"completion_model": "interaction-default"},
                }
            ),
            encoding="utf-8",
        )
        with owned_proxy_process(gateway, tmp_path, {}, config=config) as owned:
            candidate: Final = Gateway(owned.gateway.client, owned.gateway.key, gateway.upstream_url)
            request_input: Final = f"interaction-{identity}"
            response: Final = _post(candidate, "/interactions", {"input": request_input})
            assert response.status_code == 200, response.text
            _assert_scripted_response("acreate_interaction", JSON_OBJECT.validate_python(response.json()))
            observations: Final = _Observations(gateway.upstream_url)
            eventually(
                observations.read,
                lambda _items: any(item.get("method") == "POST" for item in observations.for_scenario(identity)),
                seconds=20,
            )
            upstream_calls: Final = tuple(
                item for item in observations.for_scenario(identity) if item.get("method") == "POST"
            )
            assert len(upstream_calls) == 1, upstream_calls
            outbound: Final = object_value(upstream_calls[0]["body"])
            assert outbound.get("input") == request_input, outbound
            assert outbound.get("model") == "gemini-2.5-flash", outbound


def test_promptless_image_edit_reaches_upstream(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        identity: Final = f"image-edit-{uuid.uuid4().hex}"
        handle: Final = register_scenario(identity, _response("aimage_generation"))
        scenario.cleanups.callback(delete_scenario, handle)
        model: Final = scenario.model(model="openai/gpt-image-1", api_base=handle.api_base(), api_key=identity)
        response: Final = eventually(
            lambda: gateway.client.post(
                "/v1/images/edits",
                data={"model": model},
                files={"image": ("audit.png", b"png", "image/png")},
                headers={"Authorization": f"Bearer {gateway.key}"},
            ),
            lambda result: not (result.status_code == 400 and "Invalid model name" in result.text),
            seconds=30,
        )
        assert response.status_code == 200, response.text
        outbound: Final = object_value(_observed(_Observations(gateway.upstream_url), identity)[0]["body"])
        assert "prompt" not in outbound and any(field in outbound for field in ("image", "image[]")), outbound


def _healthy(url: str) -> int:
    try:
        return httpx.get(f"{url}/health", timeout=2, trust_env=False).status_code
    except httpx.TransportError:
        return 0


@contextmanager
def _upstream(directory: Path) -> Iterator[tuple[subprocess.Popen[bytes], str]]:
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        port: Final = int(reserve.getsockname()[1])
    root: Final = Path(__file__).resolve().parents[2]
    url: Final = f"http://127.0.0.1:{port}"
    with (directory / "upstream.log").open("w") as log:
        process: Final = subprocess.Popen(
            [sys.executable, "-P", "-m", "integration._support.upstream", "--port", str(port)],
            cwd=root,
            env={**os.environ, "PYTHONPATH": str(root)},
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        eventually(lambda: _healthy(url), lambda status: status == 200, seconds=30)
        yield process, url
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGCONT)
            process.terminate()
            process.wait(timeout=10)


def _register_owned(url: str, identity: str, scripted: JsonResponse) -> ScenarioHandle:
    result: Final = httpx.post(
        f"{url}/__scenarios",
        json={"scenario_id": identity, "response": scripted.model_dump(mode="json")},
        timeout=10,
        trust_env=False,
    )
    result.raise_for_status()
    return ScenarioHandle(identity, url)


def _delete_owned(handle: ScenarioHandle) -> None:
    httpx.delete(
        f"{handle.control_url}/__scenarios/{handle.scenario_id}", timeout=10, trust_env=False
    ).raise_for_status()


def _workers(process: subprocess.Popen[bytes]) -> tuple[psutil.Process, ...]:
    return tuple(psutil.Process(process.pid).children(recursive=True))


def _process_tree_line(process: psutil.Process) -> str:
    try:
        return f"{process.pid} {' '.join(process.cmdline())}"
    except psutil.Error:
        return f"{process.pid} <exited>"


def _worker_alive(worker: psutil.Process) -> bool:
    try:
        return worker.is_running() and worker.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _chat_body(model: str, marker: str, valid: bool) -> dict[str, JsonValue]:
    return {"model": model, "user": marker, **({"messages": [{"role": "user", "content": marker}]} if valid else {})}


def _spend_rows(request_id: str) -> list[dict[str, JsonValue]]:
    return read_rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (request_id,))


def _owned_model(
    scenario: Scenario,
    url: str,
    route: str,
    script: JsonResponse,
) -> tuple[str, str, ScenarioHandle]:
    identity: Final = f"chaos-{route}-{uuid.uuid4().hex}"
    handle: Final = _register_owned(url, identity, script)
    scenario.cleanups.callback(_delete_owned, handle)
    model: Final = scenario.model(model=_ROUTES[route][4], api_base=handle.api_base(), api_key=identity)
    return model, identity, handle


def _missing_call(
    route: str,
    parameter: str,
    model: str,
    identity: str,
) -> tuple[str, dict[str, JsonValue], dict[str, JsonValue], str]:
    template, error_route, _fields, valid_body, _provider_model = _ROUTES[route]
    body: Final = {key: value for key, value in {**valid_body, "model": model}.items() if key != parameter}
    return _path(template, model, "", ""), body, _error(error_route, parameter), identity


def test_upstream_pause_and_worker_kill_preserve_required_body_status(
    gateway: Gateway,
    tmp_path: Path,
    record_property: pytest.RecordProperty,
) -> None:
    with (
        _upstream(tmp_path) as (upstream, url),
        owned_proxy_process(
            gateway,
            tmp_path,
            {"INTEGRATION_UPSTREAM_URL": url},
            workers=2,
        ) as owned,
    ):
        candidate: Final = Gateway(owned.gateway.client, owned.gateway.key, url)
        with candidate.scenario() as scenario:
            script: Final = JsonResponse(content_type="application/json", body=_BODIES["acompletion"])
            chat_model, chat_identity, _chat_handle = _owned_model(scenario, url, "acompletion", script)
            probe: Final = eventually(
                lambda: _post(candidate, "/v1/chat/completions", _chat_body(chat_model, "probe", True)),
                lambda response: response.status_code == 200,
                seconds=30,
            )
            assert probe.status_code == 200, probe.text
            process_root: Final = psutil.Process(owned.process.pid)
            processes: Final = (process_root, *process_root.children(recursive=True))
            process_tree: Final = "\n".join(_process_tree_line(process) for process in processes)
            record_property("owned_proxy_process_tree", process_tree)
            workers: Final = eventually(
                lambda: _workers(owned.process),
                lambda children: len(children) >= 2,
                seconds=30,
            )
            observations: Final = _Observations(url)
            missing_routes: Final = (
                ("aspeech", "input"),
                ("aspeech", "input"),
                ("amoderation", "input"),
                ("amoderation", "input"),
                ("aimage_generation", "prompt"),
                ("aimage_generation", "prompt"),
                ("atext_completion", "prompt"),
                ("atext_completion", "prompt"),
                ("arerank", "query"),
                ("arerank", "documents"),
            )
            missing_models: Final = tuple(
                _owned_model(scenario, url, route, script) for route, _parameter in missing_routes
            )
            missing_calls: Final = tuple(
                _missing_call(route, parameter, model, identity)
                for (route, parameter), (model, identity, _handle) in zip(missing_routes, missing_models)
            )
            markers: Final = tuple(f"burst-{uuid.uuid4().hex}" for _ in range(20))
            valid_calls: Final = tuple(
                ("/v1/chat/completions", _chat_body(chat_model, marker, True), marker) for marker in markers
            )
            upstream.send_signal(signal.SIGSTOP)
            try:
                with ThreadPoolExecutor(max_workers=10) as pool:
                    missing_futures: Final = tuple(
                        pool.submit(_post, candidate, path, body) for path, body, _expected, _identity in missing_calls
                    )
                    missing: Final = tuple(
                        (call, future.result(timeout=15)) for call, future in zip(missing_calls, missing_futures)
                    )
                    assert all(
                        response.status_code == 400 and response.json() == expected
                        for (_path, _body, expected, _identity), response in missing
                    ), [response.text for _call, response in missing]
                    paused_statuses: Final = tuple(response.status_code for _call, response in missing)
                    record_property(
                        "chaos_paused_missing_status_counts",
                        str({status: paused_statuses.count(status) for status in sorted(set(paused_statuses))}),
                    )
            finally:
                upstream.send_signal(signal.SIGCONT)
            with ThreadPoolExecutor(max_workers=20) as pool:
                valid_futures: Final = tuple(
                    pool.submit(_post, candidate, path, body) for path, body, _marker in valid_calls
                )
                valid: Final = tuple(future.result(timeout=30) for future in valid_futures)
            assert all(response.status_code == 200 for response in valid), [response.text for response in valid]
            record_property("chaos_burst_size", len(missing_calls) + len(valid_calls))
            resumed_statuses: Final = tuple(response.status_code for response in valid)
            record_property(
                "chaos_resumed_valid_status_counts",
                str({status: resumed_statuses.count(status) for status in sorted(set(resumed_statuses))}),
            )
            eventually(
                observations.read,
                lambda _items: all(
                    sum(
                        object_value(item["body"]).get("user") == marker
                        for item in observations.for_scenario(chat_identity)
                    )
                    == 1
                    for _path, _body, marker in valid_calls
                ),
                seconds=30,
            )
            missing_observations: Final = {
                identity: len(observations.provider_calls(identity))
                for _path, _body, _expected, identity in missing_calls
            }
            assert all(count == 0 for count in missing_observations.values()), missing_observations
            record_property(
                "chaos_missing_split",
                str(tuple(f"{route}:{parameter}" for route, parameter in missing_routes)),
            )
            record_property("chaos_missing_upstream_provider_call_counts", str(missing_observations))
            request_ids: Final = tuple(str(JSON_OBJECT.validate_python(response.json())["id"]) for response in valid)
            spend_rows: Final = tuple(
                eventually(partial(_spend_rows, request_id), lambda values: len(values) == 1, seconds=60)
                for request_id in request_ids
            )
            assert all(rows[0]["request_id"] == request_id for rows, request_id in zip(spend_rows, request_ids)), (
                spend_rows
            )
            record_property("chaos_spend_query", 'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id = %s')
            record_property("chaos_spend_response_id_count", len(request_ids))
            record_property("chaos_spend_row_counts", str(tuple(len(rows) for rows in spend_rows)))
            workers[0].kill()
            eventually(lambda: _worker_alive(workers[0]), lambda alive: not alive, seconds=10)
            post_kill: Final = tuple(
                _post(candidate, path, body) for path, body, _expected, _identity in missing_calls[:5]
            )
            assert all(
                response.status_code == 400 and response.json() == expected
                for response, (_path, _body, expected, _identity) in zip(post_kill, missing_calls[:5])
            ), [response.text for response in post_kill]
            recovered: Final = _post(candidate, "/v1/chat/completions", _chat_body(chat_model, "recovered", True))
            assert recovered.status_code == 200, recovered.text


_VIDEO_UPSTREAM_ID: Final = "video_upstream"
_VIDEO_PROVIDER_KEY: Final = "video-provider-key"
_VIDEO_BODY: Final = {
    "id": _VIDEO_UPSTREAM_ID,
    "object": "video",
    "model": "sora-2",
    "status": "queued",
    "created_at": 1,
    "progress": 0,
    "seconds": "8",
    "size": "1280x720",
}
_PNG: Final = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + bytes(20)
_MP4: Final = b"\x00\x00\x00\x18ftypmp42" + bytes(20)

_Part = tuple[str, str | None, str, bytes]


def _multipart_parts(request: Request) -> tuple[_Part, ...]:
    content_type: Final = request.headers["content-type"]
    assert content_type.startswith("multipart/form-data; boundary="), content_type
    envelope: Final = f"content-type: {content_type}\r\n\r\n".encode() + request.body
    parsed: Final = BytesParser(policy=HTTP).parsebytes(envelope)
    assert parsed.is_multipart(), request.body
    return tuple(
        (
            str(part.get_param("name", header="content-disposition")),
            part.get_filename(),
            part.get_content_type(),
            bytes(part.get_payload(decode=True)),
        )
        for part in parsed.iter_parts()
    )


def _video_upstream(seen: list[Request]) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        seen.append(request)
        return Reply(body=json.dumps(_VIDEO_BODY).encode())

    return respond


def _video_model(scenario: Scenario, wire: Wire) -> str:
    return scenario.model(model="openai/sora-2", api_base=f"{wire.url}/v1", api_key=_VIDEO_PROVIDER_KEY)


def _routable_video_client(gateway: Gateway, model: str) -> OpenAI:
    client: Final = OpenAI(base_url=str(gateway.client.base_url.join("/v1")), api_key=gateway.key, max_retries=0)
    eventually(lambda: tuple(entry.id for entry in client.models.list()), lambda ids: model in ids)
    return client


def _routable_over_gateway(gateway: Gateway, model: str) -> None:
    eventually(
        lambda: tuple(str(object_value(entry)["id"]) for entry in gateway.get("/v1/models")["data"]),
        lambda ids: model in ids,
    )


def _video_target(request: Request) -> tuple[str, str, str]:
    return request.method, request.target, request.headers["authorization"]


def _client_video(video_id: str, model: str, *, created: bool) -> Video:
    usage: Final = {"duration_seconds": 8.0} if created else None
    return Video.model_validate({**_VIDEO_BODY, "id": video_id, "model": model, "usage": usage})


def _upstream_video_id(video: Video) -> tuple[str, str]:
    decoded: Final = decode_video_id_with_provider(video.id)
    return str(decoded["custom_llm_provider"]), str(decoded["video_id"])


def test_openai_sdk_multipart_video_create_forwards_input_reference_as_a_file(gateway: Gateway) -> None:
    seen: Final[list[Request]] = []
    with wire_server(_video_upstream(seen)) as wire, gateway.scenario() as scenario:
        model: Final = _video_model(scenario, wire)
        versioned: Final = _routable_video_client(gateway, model)
        unversioned: Final = versioned.with_options(base_url=str(gateway.client.base_url))
        videos: Final = tuple(
            client.videos.create(
                prompt=prompt,
                model=model,
                seconds="8",
                size="1280x720",
                input_reference=("r.png", _PNG, "image/png"),
                extra_body={"style_hint": "noir"},
            )
            for client, prompt in ((versioned, "p-v1"), (unversioned, "p-alias"))
        )
        for video in videos:
            assert video == _client_video(video.id, model, created=True), video
            assert _upstream_video_id(video) == ("openai", _VIDEO_UPSTREAM_ID), video.id
        assert [_video_target(request) for request in seen] == [
            ("POST", "/v1/videos", f"Bearer {_VIDEO_PROVIDER_KEY}"),
            ("POST", "/v1/videos", f"Bearer {_VIDEO_PROVIDER_KEY}"),
        ]
        assert [_multipart_parts(request) for request in seen] == [
            (
                ("model", None, "text/plain", b"sora-2"),
                ("prompt", None, "text/plain", prompt),
                ("seconds", None, "text/plain", b"8"),
                ("size", None, "text/plain", b"1280x720"),
                ("style_hint", None, "text/plain", b"noir"),
                ("input_reference", "input_reference.png", "image/png", _PNG),
            )
            for prompt in (b"p-v1", b"p-alias")
        ], [request.body for request in seen]
        assert len(wire.drain()) == 2


def test_openai_sdk_multipart_video_edit_forwards_the_uploaded_video_file(gateway: Gateway) -> None:
    seen: Final[list[Request]] = []
    with wire_server(_video_upstream(seen)) as wire, gateway.scenario() as scenario:
        model: Final = _video_model(scenario, wire)
        client: Final = _routable_video_client(gateway, model)
        video: Final = client.videos.edit(
            prompt="brighter", video=("v.mp4", _MP4, "video/mp4"), extra_body={"model": model}
        )
        assert video == _client_video(video.id, model, created=False), video
        assert _upstream_video_id(video) == ("openai", _VIDEO_UPSTREAM_ID), video.id
        assert [_video_target(request) for request in seen] == [
            ("POST", "/v1/videos/edits", f"Bearer {_VIDEO_PROVIDER_KEY}")
        ]
        assert _multipart_parts(seen[0]) == (
            ("prompt", None, "text/plain", b"brighter"),
            ("video", "v.mp4", "video/mp4", _MP4),
        ), seen[0].body
        assert len(wire.drain()) == 1


def test_openai_sdk_multipart_video_extension_forwards_the_uploaded_video_file(gateway: Gateway) -> None:
    pytest.skip(
        "BUG: client.videos.extend(video=<mp4 file>) returns 200 but the upstream receives JSON "
        '{"prompt":"extend","seconds":"4","video":{"id":""}} instead of a multipart video file part'
    )
    seen: Final[list[Request]] = []
    with wire_server(_video_upstream(seen)) as wire, gateway.scenario() as scenario:
        model: Final = _video_model(scenario, wire)
        client: Final = _routable_video_client(gateway, model)
        video: Final = client.videos.extend(
            prompt="extend", seconds="4", video=("v.mp4", _MP4, "video/mp4"), extra_body={"model": model}
        )
        assert video == _client_video(video.id, model, created=False), video
        assert _upstream_video_id(video) == ("openai", _VIDEO_UPSTREAM_ID), video.id
        assert [_video_target(request) for request in seen] == [
            ("POST", "/v1/videos/extensions", f"Bearer {_VIDEO_PROVIDER_KEY}")
        ]
        assert sorted(_multipart_parts(seen[0])) == sorted(
            (
                ("prompt", None, "text/plain", b"extend"),
                ("seconds", None, "text/plain", b"4"),
                ("video", "v.mp4", "video/mp4", _MP4),
            )
        ), seen[0].body
        assert len(wire.drain()) == 1


def test_form_encoded_video_references_reach_upstream_as_decoded_ids(gateway: Gateway) -> None:
    seen: Final[list[Request]] = []
    with wire_server(_video_upstream(seen)) as wire, gateway.scenario() as scenario:
        model: Final = _video_model(scenario, wire)
        _routable_over_gateway(gateway, model)
        encoded: Final = encode_video_id_with_provider("video-source", "openai", model_id=model)
        auth: Final = {"Authorization": f"Bearer {gateway.key}"}
        responses: Final = (
            gateway.client.post(
                "/v1/videos/edits", data={"prompt": "edit", "video": json.dumps({"id": encoded})}, headers=auth
            ),
            gateway.client.post(
                "/videos/edits",
                files={"prompt": (None, "edit"), "video": (None, json.dumps({"id": encoded}))},
                headers=auth,
            ),
            gateway.client.post(
                "/v1/videos/extensions", data={"prompt": "extend", "seconds": "4", "video": encoded}, headers=auth
            ),
            gateway.client.post(
                "/videos/extensions",
                files={
                    "prompt": (None, "extend"),
                    "seconds": (None, "4"),
                    "video": (None, json.dumps({"id": encoded})),
                },
                headers=auth,
            ),
            gateway.client.post("/v1/videos/edits", data={"prompt": "edit", "video": encoded}, headers=auth),
            gateway.client.post(
                "/videos/edits",
                files={"prompt": (None, "edit"), "video": (None, encoded)},
                headers=auth,
            ),
            gateway.client.post(
                "/v1/videos/extensions",
                files={
                    "prompt": (None, "extend"),
                    "seconds": (None, "4"),
                    "video": (None, encoded),
                },
                headers=auth,
            ),
        )
        provider_auth: Final = f"Bearer {_VIDEO_PROVIDER_KEY}"
        assert [_video_target(request) for request in seen] == [
            ("POST", "/v1/videos/edits", provider_auth),
            ("POST", "/v1/videos/edits", provider_auth),
            ("POST", "/v1/videos/extensions", provider_auth),
            ("POST", "/v1/videos/extensions", provider_auth),
            ("POST", "/v1/videos/edits", provider_auth),
            ("POST", "/v1/videos/edits", provider_auth),
            ("POST", "/v1/videos/extensions", provider_auth),
        ], [response.text for response in responses]
        for response in responses:
            assert response.status_code == 200, response.text
            video = Video.model_validate_json(response.text)
            assert video == _client_video(video.id, model, created=False), response.text
            assert _upstream_video_id(video) == ("openai", _VIDEO_UPSTREAM_ID), response.text
        assert [request.headers["content-type"] for request in seen] == ["application/json"] * 7
        assert [JSON_OBJECT.validate_json(request.body) for request in seen] == [
            {"prompt": "edit", "video": {"id": "video-source"}},
            {"prompt": "edit", "video": {"id": "video-source"}},
            {"prompt": "extend", "seconds": "4", "video": {"id": "video-source"}},
            {"prompt": "extend", "seconds": "4", "video": {"id": "video-source"}},
            {"prompt": "edit", "video": {"id": "video-source"}},
            {"prompt": "edit", "video": {"id": "video-source"}},
            {"prompt": "extend", "seconds": "4", "video": {"id": "video-source"}},
        ], [request.body for request in seen]
        assert len(wire.drain()) == 7


def test_documented_json_video_extension_routes_and_decodes_the_reference(gateway: Gateway) -> None:
    seen: Final[list[Request]] = []
    with wire_server(_video_upstream(seen)) as wire, gateway.scenario() as scenario:
        model: Final = _video_model(scenario, wire)
        _routable_over_gateway(gateway, model)
        encoded: Final = encode_video_id_with_provider("video-audit", "openai", model_id=model)
        response: Final = gateway.request(
            "POST", "/v1/videos/extensions", {"prompt": "extend", "seconds": "4", "video": {"id": encoded}}
        )
        assert [_video_target(request) for request in seen] == [
            ("POST", "/v1/videos/extensions", f"Bearer {_VIDEO_PROVIDER_KEY}")
        ], response.text
        assert response.status_code == 200, response.text
        video: Final = Video.model_validate_json(response.text)
        assert video == _client_video(video.id, model, created=False), response.text
        assert _upstream_video_id(video) == ("openai", _VIDEO_UPSTREAM_ID), response.text
        assert JSON_OBJECT.validate_json(seen[0].body) == {
            "prompt": "extend",
            "seconds": "4",
            "video": {"id": "video-audit"},
        }, seen[0].body
        assert len(wire.drain()) == 1


def test_documented_json_video_edit_routes_and_decodes_the_reference(gateway: Gateway) -> None:
    seen: Final[list[Request]] = []
    with wire_server(_video_upstream(seen)) as wire, gateway.scenario() as scenario:
        model: Final = _video_model(scenario, wire)
        _routable_over_gateway(gateway, model)
        encoded: Final = encode_video_id_with_provider("video-audit", "openai", model_id=model)
        response: Final = gateway.request("POST", "/v1/videos/edits", {"prompt": "edit", "video": {"id": encoded}})
        assert [_video_target(request) for request in seen] == [
            ("POST", "/v1/videos/edits", f"Bearer {_VIDEO_PROVIDER_KEY}")
        ], response.text
        assert response.status_code == 200, response.text
        video: Final = Video.model_validate_json(response.text)
        assert video == _client_video(video.id, model, created=False), response.text
        assert _upstream_video_id(video) == ("openai", _VIDEO_UPSTREAM_ID), response.text
        assert JSON_OBJECT.validate_json(seen[0].body) == {
            "prompt": "edit",
            "video": {"id": "video-audit"},
        }, seen[0].body
        assert len(wire.drain()) == 1


def test_multipart_bracketed_video_reference_reaches_upstream_as_a_decoded_id(gateway: Gateway) -> None:
    pytest.skip(
        "BUG: multipart video[id]=<encoded id> to /v1/videos/edits and /v1/videos/extensions returns 404 and "
        "never reaches the provider because the handler only reads a top-level video field"
    )
    seen: Final[list[Request]] = []
    with wire_server(_video_upstream(seen)) as wire, gateway.scenario() as scenario:
        model: Final = _video_model(scenario, wire)
        _routable_over_gateway(gateway, model)
        encoded: Final = encode_video_id_with_provider("video-source", "openai", model_id=model)
        auth: Final = {"Authorization": f"Bearer {gateway.key}"}
        edit_response: Final = gateway.client.post(
            "/v1/videos/edits",
            files={"prompt": (None, "edit"), "video[id]": (None, encoded)},
            headers=auth,
        )
        extension_response: Final = gateway.client.post(
            "/videos/extensions",
            files={
                "prompt": (None, "extend"),
                "seconds": (None, "4"),
                "video[id]": (None, encoded),
            },
            headers=auth,
        )
        responses: Final = (edit_response, extension_response)
        assert [_video_target(request) for request in seen] == [
            ("POST", "/v1/videos/edits", f"Bearer {_VIDEO_PROVIDER_KEY}"),
            ("POST", "/v1/videos/extensions", f"Bearer {_VIDEO_PROVIDER_KEY}"),
        ], [response.text for response in responses]
        assert edit_response.status_code == 200, edit_response.text
        assert extension_response.status_code == 200, extension_response.text
        edit_video: Final = Video.model_validate_json(edit_response.text)
        extension_video: Final = Video.model_validate_json(extension_response.text)
        assert edit_video == _client_video(edit_video.id, model, created=False), edit_response.text
        assert extension_video == _client_video(extension_video.id, model, created=False), extension_response.text
        assert _upstream_video_id(edit_video) == ("openai", _VIDEO_UPSTREAM_ID), edit_response.text
        assert _upstream_video_id(extension_video) == ("openai", _VIDEO_UPSTREAM_ID), extension_response.text
        assert [JSON_OBJECT.validate_json(request.body) for request in seen] == [
            {"prompt": "edit", "video": {"id": "video-source"}},
            {"prompt": "extend", "seconds": "4", "video": {"id": "video-source"}},
        ], [request.body for request in seen]
        assert len(wire.drain()) == 2


def test_multipart_object_input_reference_reaches_upstream_as_bracketed_text_parts(gateway: Gateway) -> None:
    seen: Final[list[Request]] = []
    with wire_server(_video_upstream(seen)) as wire, gateway.scenario() as scenario:
        model: Final = _video_model(scenario, wire)
        _routable_over_gateway(gateway, model)
        auth: Final = {"Authorization": f"Bearer {gateway.key}"}
        image_response: Final = gateway.client.post(
            "/v1/videos",
            files={
                "model": (None, model),
                "prompt": (None, "p"),
                "input_reference[image_url]": (None, "https://example.com/a.png"),
            },
            headers=auth,
        )
        file_id_response: Final = gateway.client.post(
            "/v1/videos",
            files={
                "model": (None, model),
                "prompt": (None, "p"),
                "input_reference[file_id]": (None, "file_abc"),
            },
            headers=auth,
        )
        responses: Final = (image_response, file_id_response)
        assert [_video_target(request) for request in seen] == [
            ("POST", "/v1/videos", f"Bearer {_VIDEO_PROVIDER_KEY}"),
            ("POST", "/v1/videos", f"Bearer {_VIDEO_PROVIDER_KEY}"),
        ], [response.text for response in responses]
        assert image_response.status_code == 200, image_response.text
        assert file_id_response.status_code == 200, file_id_response.text
        image_video: Final = Video.model_validate_json(image_response.text)
        file_id_video: Final = Video.model_validate_json(file_id_response.text)
        assert image_video == _client_video(image_video.id, model, created=True), image_response.text
        assert file_id_video == _client_video(file_id_video.id, model, created=True), file_id_response.text
        assert _upstream_video_id(image_video) == ("openai", _VIDEO_UPSTREAM_ID), image_response.text
        assert _upstream_video_id(file_id_video) == ("openai", _VIDEO_UPSTREAM_ID), file_id_response.text
        assert sorted(_multipart_parts(seen[0])) == sorted(
            (
                ("model", None, "text/plain", b"sora-2"),
                ("prompt", None, "text/plain", b"p"),
                ("input_reference[image_url]", None, "text/plain", b"https://example.com/a.png"),
            )
        ), seen[0].body
        assert sorted(_multipart_parts(seen[1])) == sorted(
            (
                ("model", None, "text/plain", b"sora-2"),
                ("prompt", None, "text/plain", b"p"),
                ("input_reference[file_id]", None, "text/plain", b"file_abc"),
            )
        ), seen[1].body
        assert len(wire.drain()) == 2


def test_json_object_input_reference_reaches_upstream_as_bracketed_text_parts(gateway: Gateway) -> None:
    pytest.skip(
        "BUG: JSON input_reference {image_url} or {file_id} to /v1/videos returns 400 'dict' object has no "
        "attribute 'read' and the provider gets an empty multipart body"
    )
    seen: Final[list[Request]] = []
    with wire_server(_video_upstream(seen)) as wire, gateway.scenario() as scenario:
        model: Final = _video_model(scenario, wire)
        _routable_over_gateway(gateway, model)
        image_response: Final = gateway.request(
            "POST",
            "/v1/videos",
            {"model": model, "prompt": "p", "input_reference": {"image_url": "https://example.com/a.png"}},
        )
        file_id_response: Final = gateway.request(
            "POST",
            "/v1/videos",
            {"model": model, "prompt": "p", "input_reference": {"file_id": "file_abc"}},
        )
        responses: Final = (image_response, file_id_response)
        assert [_video_target(request) for request in seen] == [
            ("POST", "/v1/videos", f"Bearer {_VIDEO_PROVIDER_KEY}"),
            ("POST", "/v1/videos", f"Bearer {_VIDEO_PROVIDER_KEY}"),
        ], [response.text for response in responses]
        assert image_response.status_code == 200, image_response.text
        assert file_id_response.status_code == 200, file_id_response.text
        image_video: Final = Video.model_validate_json(image_response.text)
        file_id_video: Final = Video.model_validate_json(file_id_response.text)
        assert image_video == _client_video(image_video.id, model, created=True), image_response.text
        assert file_id_video == _client_video(file_id_video.id, model, created=True), file_id_response.text
        assert _upstream_video_id(image_video) == ("openai", _VIDEO_UPSTREAM_ID), image_response.text
        assert _upstream_video_id(file_id_video) == ("openai", _VIDEO_UPSTREAM_ID), file_id_response.text
        assert sorted(_multipart_parts(seen[0])) == sorted(
            (
                ("model", None, "text/plain", b"sora-2"),
                ("prompt", None, "text/plain", b"p"),
                ("input_reference[image_url]", None, "text/plain", b"https://example.com/a.png"),
            )
        ), seen[0].body
        assert sorted(_multipart_parts(seen[1])) == sorted(
            (
                ("model", None, "text/plain", b"sora-2"),
                ("prompt", None, "text/plain", b"p"),
                ("input_reference[file_id]", None, "text/plain", b"file_abc"),
            )
        ), seen[1].body
        assert len(wire.drain()) == 2


def test_json_video_create_decodes_character_ids_into_text_parts(gateway: Gateway) -> None:
    seen: Final[list[Request]] = []
    with wire_server(_video_upstream(seen)) as wire, gateway.scenario() as scenario:
        model: Final = _video_model(scenario, wire)
        _routable_over_gateway(gateway, model)
        character: Final = encode_character_id_with_provider("char_source", "openai", None)
        responses: Final = tuple(
            gateway.request(
                "POST",
                path,
                {
                    "model": model,
                    "prompt": prompt,
                    "seconds": 8,
                    "size": "1280x720",
                    "characters": [{"id": character}],
                    "style_hint": "noir",
                },
            )
            for path, prompt in (("/v1/videos", "p-v1"), ("/videos", "p-alias"))
        )
        for response in responses:
            assert response.status_code == 200, response.text
            video = Video.model_validate_json(response.text)
            assert video == _client_video(video.id, model, created=True), response.text
            assert _upstream_video_id(video) == ("openai", _VIDEO_UPSTREAM_ID), response.text
        assert [_video_target(request) for request in seen] == [
            ("POST", "/v1/videos", f"Bearer {_VIDEO_PROVIDER_KEY}"),
            ("POST", "/v1/videos", f"Bearer {_VIDEO_PROVIDER_KEY}"),
        ], [response.text for response in responses]
        assert [_multipart_parts(request) for request in seen] == [
            (
                ("model", None, "text/plain", b"sora-2"),
                ("prompt", None, "text/plain", prompt),
                ("seconds", None, "text/plain", b"8"),
                ("size", None, "text/plain", b"1280x720"),
                ("characters[][id]", None, "text/plain", b"char_source"),
                ("style_hint", None, "text/plain", b"noir"),
            )
            for prompt in (b"p-v1", b"p-alias")
        ], [request.body for request in seen]
        assert len(wire.drain()) == 2
