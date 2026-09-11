"""
Transformation layer for the TinyFish Agent provider (goal-based web automation REST API).
Docs: https://docs.tinyfish.ai/agent-api
"""

import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final
from uuid import uuid4

from pydantic import TypeAdapter, ValidationError
from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm._logging import verbose_logger

TINYFISH_AGENT_DOCS_URL: Final = "https://docs.tinyfish.ai/agent-api"

# Fields that run with the TinyFish account's saved logins/vault; admin-opt-in behind a shared key.
AUTHENTICATED_RUN_FIELDS: Final = frozenset({"use_profile", "profile_id", "use_vault", "credential_item_ids"})

TERMINAL_RUN_STATUSES: Final = frozenset({"COMPLETED", "FAILED", "CANCELLED"})

_StrObjectDict: Final = TypeAdapter(Mapping[str, object])

EMPTY_MAPPING: Final[Mapping[str, object]] = MappingProxyType({})


class TinyfishRunError(TypedDict, total=False):
    """Fields read from a TinyFish run-level error object."""

    code: ReadOnly[str]
    message: ReadOnly[str]
    category: ReadOnly[str]
    retry_after: ReadOnly[float]
    help_url: ReadOnly[str]


class TinyfishRun(TypedDict, total=False):
    """Fields read from TinyFish run objects and run-async/SSE payloads; most are null until terminal."""

    run_id: ReadOnly[str | None]
    status: ReadOnly[str | None]
    num_of_steps: ReadOnly[int | None]
    result: ReadOnly[object]
    error: ReadOnly[TinyfishRunError | None]
    type: ReadOnly[str | None]
    purpose: ReadOnly[str | None]
    streaming_url: ReadOnly[str | None]


class A2AJsonRpcEvent(TypedDict):
    """JSON-RPC envelope for A2A responses/stream events, plus the reserved bridge cost key."""

    jsonrpc: ReadOnly[str]
    id: ReadOnly[str]
    result: ReadOnly[object]
    _litellm_response_cost: NotRequired[ReadOnly[float]]


class A2APart(TypedDict, total=False):
    kind: ReadOnly[str]
    text: ReadOnly[str]
    data: ReadOnly[object]


class A2AAgentMessage(TypedDict, total=False):
    contextId: ReadOnly[str]
    kind: ReadOnly[str]
    messageId: ReadOnly[str]
    parts: ReadOnly[tuple[A2APart, ...]]
    role: ReadOnly[str]
    taskId: ReadOnly[str]


def as_str_object_dict(value: object) -> Mapping[str, object] | None:
    """Validated read-only view of an untyped payload; the runtime object stays a plain dict."""
    if not isinstance(value, dict):
        return None
    try:
        return _StrObjectDict.validate_python(value)
    except ValidationError:
        return None


class TinyfishAgentTransformation:
    """Request/response transformation between A2A and the TinyFish Agent REST API."""

    @staticmethod
    def build_run_body(
        params: Mapping[str, object],
        default_request_params: Mapping[str, object],
        allow_authenticated_runs: bool,
    ) -> Mapping[str, object]:
        """Text parts form the goal, message.metadata carries url and other fields (forwarded
        permissively); a single text part that is a JSON object with a `goal` key is the body itself."""
        message: Final = as_str_object_dict(params.get("message")) or EMPTY_MAPPING

        text_parts: Final = _text_parts(message)
        goal_text: Final = " ".join(text_parts)
        body_from_text: Final = _parse_json_goal(text_parts)
        metadata: Final = as_str_object_dict(message.get("metadata")) or EMPTY_MAPPING

        caller_fields: Final = MappingProxyType({**metadata, **(body_from_text or EMPTY_MAPPING)})
        sanitized_caller_fields: Final = _strip_authenticated_fields(caller_fields, allow_authenticated_runs)
        goal_seed: Final = (
            MappingProxyType({"goal": goal_text}) if goal_text and body_from_text is None else EMPTY_MAPPING
        )

        merged: Final = MappingProxyType({**goal_seed, **default_request_params, **sanitized_caller_fields})

        goal: Final = merged.get("goal")
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError(
                "TinyFish Agent: request has no goal. Send the goal as the message's text part. "
                f"See {TINYFISH_AGENT_DOCS_URL} for details."
            )
        url: Final = merged.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ValueError(
                "TinyFish Agent: request has no target url. Set `url` in the message's `metadata` "
                f"(or in the agent's default_request_params). See {TINYFISH_AGENT_DOCS_URL} for details."
            )
        return merged

    @staticmethod
    def build_a2a_message_response(request_id: str, run: TinyfishRun) -> A2AJsonRpcEvent:
        """Build the A2A SendMessageResponse for a COMPLETED TinyFish run."""
        response: Final[A2AJsonRpcEvent] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "kind": "message",
                "role": "agent",
                "parts": _result_parts(run.get("result")),
                "messageId": str(uuid4()),
                "metadata": _run_correlation_metadata(run),
            },
        }
        return response

    @staticmethod
    def run_failure_message(run: TinyfishRun) -> str:
        """Human-readable, attributed message for a FAILED/CANCELLED run."""
        status: Final = run.get("status") or "FAILED"
        error: Final = run.get("error")
        if not isinstance(error, dict):
            return TinyfishAgentTransformation.wrap_error_message(f"run ended with status {status}")
        detail_pairs: Final = (
            ("category", error.get("category")),
            ("retry_after", error.get("retry_after")),
            ("help_url", error.get("help_url")),
        )
        details: Final = ", ".join(f"{name}={value}" for name, value in detail_pairs if value is not None)
        message: Final = str(error.get("message") or f"run ended with status {status}")
        suffix: Final = f" ({details})" if details else ""
        return TinyfishAgentTransformation.wrap_error_message(f"{message}{suffix}")

    @staticmethod
    def wrap_error_message(message: str) -> str:
        """Attribute an error to TinyFish Agent with a docs pointer."""
        inner: Final = _unwrap_error_envelope(message)
        return f"TinyFish Agent: {inner}. See {TINYFISH_AGENT_DOCS_URL} for details."

    @staticmethod
    def task_submitted_event(request_id: str, task_id: str, context_id: str) -> A2AJsonRpcEvent:
        event: Final[A2AJsonRpcEvent] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "contextId": context_id,
                "id": task_id,
                "kind": "task",
                "status": {"state": "submitted"},
            },
        }
        return event

    @staticmethod
    def working_status_event(
        request_id: str,
        task_id: str,
        context_id: str,
        message_text: str,
    ) -> A2AJsonRpcEvent:
        event: Final[A2AJsonRpcEvent] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "contextId": context_id,
                "final": False,
                "kind": "status-update",
                "status": {
                    "state": "working",
                    "message": _agent_message(task_id, context_id, message_text),
                },
                "taskId": task_id,
            },
        }
        return event

    @staticmethod
    def artifact_event(
        request_id: str,
        task_id: str,
        context_id: str,
        run: TinyfishRun,
    ) -> A2AJsonRpcEvent:
        return TinyfishAgentTransformation.artifact_event_from_parts(
            request_id=request_id,
            task_id=task_id,
            context_id=context_id,
            parts=_result_parts(run.get("result")),
            metadata=_run_correlation_metadata(run),
        )

    @staticmethod
    def artifact_event_from_parts(
        request_id: str,
        task_id: str,
        context_id: str,
        parts: object,
        metadata: object,
    ) -> A2AJsonRpcEvent:
        event: Final[A2AJsonRpcEvent] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "artifact": {
                    "artifactId": str(uuid4()),
                    "name": "tinyfish_run_result",
                    "parts": parts,
                    "metadata": metadata,
                },
                "contextId": context_id,
                "kind": "artifact-update",
                "taskId": task_id,
            },
        }
        return event

    @staticmethod
    def final_status_event(
        request_id: str,
        task_id: str,
        context_id: str,
        state: str,
        message_text: str | None = None,
    ) -> A2AJsonRpcEvent:
        event: Final[A2AJsonRpcEvent] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "contextId": context_id,
                "final": True,
                "kind": "status-update",
                "status": {
                    "state": state,
                    **({"message": _agent_message(task_id, context_id, message_text)} if message_text else {}),
                },
                "taskId": task_id,
            },
        }
        return event

    @staticmethod
    def with_response_cost(event: A2AJsonRpcEvent, response_cost: float | None) -> A2AJsonRpcEvent:
        """Attach the reserved bridge cost key; the bridge strips it before the wire."""
        if response_cost is None:
            return event
        augmented: Final[A2AJsonRpcEvent] = {**event, "_litellm_response_cost": response_cost}
        return augmented


def _agent_message(task_id: str, context_id: str, message_text: str) -> A2AAgentMessage:
    message: Final[A2AAgentMessage] = {
        "contextId": context_id,
        "kind": "message",
        "messageId": str(uuid4()),
        "parts": ({"kind": "text", "text": message_text},),
        "role": "agent",
        "taskId": task_id,
    }
    return message


def _result_parts(result_obj: object) -> tuple[A2APart, ...]:
    """A dict result ships as a lossless data part plus a text rendering; anything else as text."""
    if isinstance(result_obj, dict):
        data_part: Final[A2APart] = {"kind": "data", "data": result_obj}
        text_part: Final[A2APart] = {"kind": "text", "text": json.dumps(result_obj, separators=(",", ":"))}
        return (data_part, text_part)
    text: Final = result_obj if isinstance(result_obj, str) else json.dumps(result_obj, separators=(",", ":"))
    lone_part: Final[A2APart] = {"kind": "text", "text": text}
    return (lone_part,)


def _text_parts(message: Mapping[str, object]) -> tuple[str, ...]:
    parts_obj: Final = message.get("parts")
    parts: Final[tuple[object, ...]] = tuple(parts_obj) if isinstance(parts_obj, list) else ()
    part_dicts: Final = tuple(part for part in (as_str_object_dict(item) for item in parts) if part is not None)
    return tuple(
        str(part["text"]) for part in part_dicts if part.get("kind") in (None, "", "text") and part.get("text")
    )


def _parse_json_goal(text_parts: tuple[str, ...]) -> Mapping[str, object] | None:
    """A single text part that is a JSON object with a ``goal`` key is the run body."""
    if len(text_parts) != 1:
        return None
    try:
        parsed: Final[object] = json.loads(text_parts[0])  # any-ok: json.loads -> Any
    except json.JSONDecodeError:
        return None
    body: Final = as_str_object_dict(parsed)
    if body is not None and isinstance(body.get("goal"), str):
        return body
    return None


def _strip_authenticated_fields(
    caller_fields: Mapping[str, object],
    allow_authenticated_runs: bool,
) -> Mapping[str, object]:
    if allow_authenticated_runs:
        return caller_fields
    stripped: Final = tuple(key for key in caller_fields if key in AUTHENTICATED_RUN_FIELDS)
    if stripped:
        verbose_logger.warning(
            "TinyFish Agent: dropping caller-supplied credentialed-run fields %s; "
            "set `allow_authenticated_runs: true` in the agent's litellm_params to permit them.",
            stripped,
        )
    return MappingProxyType({key: value for key, value in caller_fields.items() if key not in AUTHENTICATED_RUN_FIELDS})


def _run_correlation_metadata(run: TinyfishRun) -> dict[str, object]:  # mutable-ok: embedded in the JSON response
    """run_id/status/num_of_steps let callers correlate with TinyFish server-side logs."""
    return {  # mutable-ok: embedded in the JSON response; must stay a plain serializable dict
        key: value
        for key, value in (
            ("tinyfish_run_id", run.get("run_id")),
            ("tinyfish_status", run.get("status")),
            ("tinyfish_num_of_steps", run.get("num_of_steps")),
        )
        if value is not None
    }


def _unwrap_error_envelope(message: str) -> str:
    """Best-effort unwrap of TinyFish's ``{"error": {code, message}}`` envelope."""
    try:
        parsed: Final[object] = json.loads(message)  # any-ok: json.loads -> Any
    except (json.JSONDecodeError, TypeError):
        return message
    body: Final = as_str_object_dict(parsed)
    error_obj: Final = as_str_object_dict(body.get("error")) if body is not None else None
    if error_obj is None:
        return message
    candidate: Final = error_obj.get("message")
    if isinstance(candidate, str) and candidate:
        return candidate
    return message
