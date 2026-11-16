import io
import json
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from os import PathLike
from typing import Final, Literal, TypeVar

from openai.types import Batch
from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.core_helpers import finite_cost
from litellm.litellm_core_utils.llm_cost_calc.guardrail_cost import guardrail_information_cost
from litellm.types.llms.openai import FileTypes, OpenAIFilesPurpose
from litellm.types.utils import CallTypes, LlmProviders
from litellm.types.videos.main import VideoObject
from litellm.types.videos.utils import POLLED_VIDEO_STATUS_KWARG, canonical_video_id, is_video_status_call_type

_DECODED_JSONL_ROW: Final = TypeAdapter(object)
_DEPLOYMENT_FIELD: Final = TypeAdapter(Mapping[str, object])
_PAYLOAD_FIELDS: Final = TypeAdapter(Mapping[str, object])
_ENDPOINT_PATHS: Final = TypeAdapter(tuple[str, ...])

_COMPLETED_BATCH_STATUSES: Final = frozenset({"completed", "complete"})
_TERMINAL_BATCH_STATUSES: Final = _COMPLETED_BATCH_STATUSES | frozenset({"failed", "cancelled", "expired"})
BATCH_ENDPOINT: Final = "/v1/batch"
POLLED_BATCH_ID_KWARG: Final = "polled_batch_id"
POLLED_BATCH_COST_IS_FINAL_KWARG: Final = "polled_batch_cost_is_final"
BATCH_COST_REQUEST_ID_SUFFIX: Final = "_batch_cost"
_AZURE_PROVIDERS: Final = frozenset({LlmProviders.AZURE.value, LlmProviders.AZURE_AI.value})

_batch_create_routing: Final[ContextVar[bool]] = ContextVar("batch_create_routing", default=False)
_DeploymentT: Final = TypeVar("_DeploymentT", bound=Mapping[str, object])


class InMemoryFile(io.BytesIO):
    def __init__(self, content: bytes, name: str, content_type: str = "application/jsonl"):
        super().__init__(content)
        self.name = name
        self.content_type = content_type


def parse_jsonl_with_embedded_newlines(content: str) -> list[dict]:
    """
    Parse JSONL content that may contain JSON objects with embedded newlines in string values.

    Unlike splitlines(), this function properly handles cases where JSON string values
    contain literal newline characters, which would otherwise break simple line-based parsing.

    Args:
        content: The JSONL file content as a string

    Returns:
        List of parsed JSON objects

    Example:
        >>> content = '{"id":1,"msg":"Line 1\\nLine 2"}\\n{"id":2,"msg":"test"}'
        >>> parse_jsonl_with_embedded_newlines(content)
        [{"id":1,"msg":"Line 1\\nLine 2"}, {"id":2,"msg":"test"}]
    """
    json_objects: Final = []
    buffer = ""

    for char in content:
        buffer += char
        if char == "\n":
            # Try to parse what we have so far
            try:
                json_object = _DECODED_JSONL_ROW.validate_python(json.loads(buffer.strip()))
                json_objects.append(json_object)
                buffer = ""  # Reset buffer for next JSON object
            except json.JSONDecodeError:
                # Not a complete JSON object yet, keep accumulating
                continue

    # Handle any remaining content in buffer
    if buffer.strip():
        try:
            json_object = _DECODED_JSONL_ROW.validate_python(json.loads(buffer.strip()))
            json_objects.append(json_object)
        except json.JSONDecodeError as e:
            verbose_logger.error("error parsing final buffer: %s..., error: %s", buffer[:100], e)
            raise e

    return json_objects


@contextmanager
def batch_create_routing() -> Generator[None, None, None]:
    token: Final = _batch_create_routing.set(True)
    try:
        yield
    finally:
        _batch_create_routing.reset(token)


def is_batch_create_routing() -> bool:
    return _batch_create_routing.get()


def _deployment_field(deployment: Mapping[str, object], field: str) -> Mapping[str, object]:
    return _DEPLOYMENT_FIELD.validate_python(deployment.get(field) or {})


def is_batch_capable(deployment: Mapping[str, object]) -> bool:
    return _deployment_field(deployment, "model_info").get("mode") == "batch"


def _deployment_provider(deployment: Mapping[str, object]) -> str | None:
    litellm_params: Final = _deployment_field(deployment, "litellm_params")
    configured_provider: Final = litellm_params.get("custom_llm_provider")
    if isinstance(configured_provider, str):
        return configured_provider
    model: Final = litellm_params.get("model")
    return model.split("/", 1)[0] if isinstance(model, str) and "/" in model else None


def _cannot_run_batches(deployment: Mapping[str, object], provider: str | None) -> bool:
    return provider in _AZURE_PROVIDERS and not is_batch_capable(deployment)


def without_deployments_that_cannot_batch(
    deployments: Sequence[_DeploymentT], caller_deployments: Callable[[], Iterable[Mapping[str, object]]]
) -> Sequence[_DeploymentT]:
    if not is_batch_create_routing() or not any(is_batch_capable(deployment) for deployment in caller_deployments()):
        return deployments
    return [
        deployment
        for deployment in deployments
        if not _cannot_run_batches(deployment, _deployment_provider(deployment))
    ]


def _endpoint_paths(configured: object) -> tuple[str, ...]:
    try:
        return _ENDPOINT_PATHS.validate_python(configured)
    except ValidationError:
        return ()


def deployment_supported_endpoints(
    resolved_endpoints: object, deployment: Mapping[str, object], provider: str | None
) -> tuple[str, ...]:
    endpoints: Final = _endpoint_paths(resolved_endpoints)
    if is_batch_capable(deployment) and provider in _AZURE_PROVIDERS:
        return (BATCH_ENDPOINT,)
    if is_batch_capable(deployment):
        return endpoints if BATCH_ENDPOINT in endpoints else (*endpoints, BATCH_ENDPOINT)
    if _cannot_run_batches(deployment, provider):
        return tuple(endpoint for endpoint in endpoints if endpoint != BATCH_ENDPOINT)
    return endpoints


def should_replace_model_in_jsonl(
    purpose: OpenAIFilesPurpose,
    passthrough: bool = False,
) -> bool:
    """
    Check if the model name should be replaced in the JSONL file for the deployment model name.

    Azure raises an error on create batch if the model name for deployment is not in the .jsonl.
    A passthrough upload keeps the caller's bytes untouched, so its rows are never rewritten.
    """
    return purpose == "batch" and not passthrough


def replace_model_in_jsonl(file_content: FileTypes, new_model_name: str, keep_line_numbers: bool = False) -> FileTypes:
    try:
        ## if pathlike, return the original file content
        if isinstance(file_content, PathLike):
            return file_content

        # Iterate the source line-by-line WITHOUT reading it all into memory. A
        # spooled upload handle (managed batches stream from it) is read straight
        # off its backing; bytes/str are wrapped so they iterate line-by-line.
        source: Final = file_content[1] if isinstance(file_content, tuple) else file_content
        if isinstance(source, (bytes, bytearray)):
            line_iter: Iterable[bytes] | Iterable[str] = io.BytesIO(bytes(source))
        elif isinstance(source, str):
            line_iter = io.StringIO(source)
        elif hasattr(source, "read") and not isinstance(source, PathLike):
            if hasattr(source, "seek"):
                try:
                    source.seek(0)
                except (OSError, ValueError):
                    pass
            line_iter = source
        else:
            return file_content

        # Rewrite one row at a time, writing straight into the output buffer
        # instead of holding every parsed row in a list. Peak memory stays at
        # ~one row plus the output rather than several full copies of the file,
        # which the managed-files path depends on (it re-runs this rewrite once
        # per target model). Lines are accumulated so JSON objects that span
        # multiple physical lines still parse. Streaming the handle also means
        # the model rewrite is actually applied to tuple-wrapped upload handles;
        # otherwise a restricted body.model would survive and bypass the batch
        # model allowlist (which validates the upload target alias). With
        # keep_line_numbers each row is written on the line it ends on, blank
        # lines included, so a row error naming a line still points into the
        # caller's file; otherwise rows go back to back, since OpenAI rejects a
        # batch file whose first line is empty.
        output: Final = InMemoryFile(b"", name="modified_file.jsonl", content_type="application/jsonl")
        last_written_line = 0
        buffer = ""
        for line_number, raw_line in enumerate(line_iter, start=1):
            buffer += (
                raw_line.decode("utf-8-sig")
                if isinstance(raw_line, (bytes, bytearray))
                else raw_line.removeprefix("\ufeff")
            )
            stripped = buffer.strip()
            if not stripped:
                buffer = ""
                continue
            try:
                json_object = _DECODED_JSONL_ROW.validate_python(json.loads(stripped))
            except json.JSONDecodeError:
                continue  # object not complete yet; keep accumulating
            if isinstance(json_object, dict) and isinstance(json_object.get("body"), dict):
                json_object["body"]["model"] = new_model_name
            output.write(
                (
                    "\n" * (line_number - max(last_written_line, 1) if keep_line_numbers else min(last_written_line, 1))
                    + json.dumps(json_object)
                ).encode("utf-8")
            )
            last_written_line = line_number
            buffer = ""

        if buffer.strip():
            # A row never parsed (truncated/malformed, or it swallowed the rows
            # that followed it). Returning the partial `output` would silently
            # drop those rows; return the unchanged original so the provider
            # rejects the batch loudly instead of accepting a truncated one.
            verbose_logger.error("error parsing trailing batch content: %s...", buffer[:100])
            if hasattr(source, "seek"):
                try:
                    source.seek(0)
                except (OSError, ValueError):
                    pass
            return file_content

        # If no valid JSON objects were found, return the original content
        if not last_written_line:
            return file_content

        output.seek(0)
        return output

    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        # return the original file content if there is an error replacing the model name
        return file_content


def get_router_metadata_variable_name(function_name: str | None) -> str:
    """
    Helper to return what the "metadata" field should be called in the request data

    For all /thread or /assistant endpoints we need to call this "litellm_metadata"

    For ALL other endpoints we call this "metadata
    """
    ROUTER_METHODS_USING_LITELLM_METADATA: Final = set(
        [
            "batch",
            "generic_api_call",
            "_acreate_batch",
            "file",
            "_ageneric_api_call_with_fallbacks",
        ]
    )
    if function_name and any(method in function_name for method in ROUTER_METHODS_USING_LITELLM_METADATA):
        return "litellm_metadata"
    else:
        return "metadata"


_get_router_metadata_variable_name = get_router_metadata_variable_name


BATCH_RETRIEVE_CALL_TYPES: Final = frozenset(
    {
        CallTypes.aretrieve_batch.value,
        CallTypes.retrieve_batch.value,
    }
)


def is_batch_retrieve_call_type(call_type: object) -> bool:
    """
    A batch retrieve reports the whole job's token usage, which the provider spent
    asynchronously over the life of the batch, and reports it again on every poll of the
    finished batch. The counters that measure live traffic, per-minute rate limits and the
    routing strategies' own state, must not be fed from it.
    """
    return isinstance(call_type, str) and call_type in BATCH_RETRIEVE_CALL_TYPES


@dataclass(frozen=True, slots=True)
class ChargeOnceJob:
    kind: Literal["batch", "video"]
    job_id: str
    repeat_poll_cost: float = 0.0


def job_charged_once_marker_key(spend_key: str, job: ChargeOnceJob) -> str:
    return f"{spend_key}:{job.kind}:{job.job_id}"


def _guardrail_cost_of_poll(standard_logging_payload: object) -> float:
    guardrail_information: Final = (
        _PAYLOAD_FIELDS.validate_python(standard_logging_payload).get("guardrail_information")
        if isinstance(standard_logging_payload, Mapping)
        else None
    )
    return guardrail_information_cost(guardrail_information)


def charged_poll_cost(kwargs: Mapping[str, object]) -> object:
    """The poll's whole cost as the cost callback charges it, guardrails included, which the logging kwargs' own
    response_cost leaves out."""
    standard_logging_payload: Final = kwargs.get("standard_logging_object")
    logged_cost: Final = (
        _PAYLOAD_FIELDS.validate_python(standard_logging_payload).get("response_cost")
        if isinstance(standard_logging_payload, Mapping)
        else None
    )
    return kwargs.get("response_cost") if logged_cost is None else logged_cost


def job_cost_of_poll(response_cost: object, standard_logging_payload: object) -> float:
    """What the polled job itself cost, since a guardrail on the poll charges every poll, running or finished."""
    return (finite_cost(response_cost) or 0.0) - _guardrail_cost_of_poll(standard_logging_payload)


def repeat_poll_cost(call_type: object, response_cost: object, standard_logging_payload: object) -> float:
    """What a poll still costs once an earlier poll charged its job: a video poll's guardrails, which every video
    poll has always paid, and nothing for a batch poll, whose repeats have always been free."""
    if not is_video_status_call_type(call_type):
        return 0.0
    return max(0.0, min(finite_cost(response_cost) or 0.0, _guardrail_cost_of_poll(standard_logging_payload)))


def batch_cost_is_final(batch: Batch) -> bool:
    """Whether this retrieve of the batch is the one to account its cost from.

    A batch still in flight has nothing to price, and a "completed" batch can report
    no output_file_id for a moment before the output populates; pricing either records
    $0 under the batch's single spend row and pins it there. Final means a completed
    batch whose output file has arrived or whose counts prove no line succeeded, or
    any other terminal status (failed, cancelled, expired).
    """
    if batch.status not in _TERMINAL_BATCH_STATUSES:
        return False
    if batch.status not in _COMPLETED_BATCH_STATUSES or batch.output_file_id is not None:
        return True
    request_counts: Final = batch.request_counts
    return request_counts is not None and request_counts.total > 0 and request_counts.completed == 0


@dataclass(frozen=True, slots=True)
class PolledBatch:
    id: str
    cost_is_final: bool


def _polled_batch(kwargs: Mapping[str, object], response_obj: object) -> PolledBatch | None:
    """The retrieved batch, read from the logging kwargs when message redaction replaced the logged response."""
    if not is_batch_retrieve_call_type(kwargs.get("call_type")):
        return None
    if isinstance(response_obj, Batch):
        return PolledBatch(id=response_obj.id, cost_is_final=batch_cost_is_final(response_obj))
    stamped_id: Final = kwargs.get(POLLED_BATCH_ID_KWARG)
    stamped_cost_is_final: Final = kwargs.get(POLLED_BATCH_COST_IS_FINAL_KWARG)
    if not isinstance(stamped_id, str) or not stamped_id or not isinstance(stamped_cost_is_final, bool):
        return None
    return PolledBatch(id=stamped_id, cost_is_final=stamped_cost_is_final)


def defers_batch_cost(kwargs: Mapping[str, object], response_obj: object) -> bool:
    """A batch retrieve before the batch's cost is final charges nothing, guardrails included, since the retrieve
    that settles the batch charges its whole cost once."""
    polled: Final = _polled_batch(kwargs, response_obj)
    return polled is not None and not polled.cost_is_final


def pays_only_its_guardrail(call_type: object, kwargs: Mapping[str, object], response_cost: object) -> bool:
    """A video poll whose whole cost is its guardrails', charged like any other request."""
    if not is_video_status_call_type(call_type):
        return False
    return (
        job_cost_of_poll(response_cost, kwargs.get("standard_logging_object"))
        <= 0
        < (finite_cost(response_cost) or 0.0)
    )


def _requested_video_id(kwargs: Mapping[str, object]) -> str | None:
    litellm_params: Final = kwargs.get("litellm_params")
    requested: Final = (
        _PAYLOAD_FIELDS.validate_python(litellm_params).get("video_id") if isinstance(litellm_params, Mapping) else None
    )
    return requested if isinstance(requested, str) and requested else None


def charged_video_id(
    call_type: object, kwargs: Mapping[str, object], fallback_video_id: str | None, response_cost: object
) -> str | None:
    """The id a finished video's own cost is charged once under.

    That is the id the client polled, which names the deployment that made the video, since a
    custom provider's own id can repeat across deployments. The status comes from the logging
    kwargs because message redaction replaces the logged video object.
    """
    if not is_video_status_call_type(call_type) or kwargs.get(POLLED_VIDEO_STATUS_KWARG) != "completed":
        return None
    if job_cost_of_poll(response_cost, kwargs.get("standard_logging_object")) <= 0:
        return None
    video_id: Final = _requested_video_id(kwargs) or fallback_video_id
    return None if video_id is None else canonical_video_id(video_id)


def _charged_batch(kwargs: Mapping[str, object], response_obj: object, response_cost: float) -> ChargeOnceJob | None:
    polled: Final = _polled_batch(kwargs, response_obj)
    if polled is None or not polled.cost_is_final or response_cost <= 0:
        return None
    return ChargeOnceJob(kind="batch", job_id=polled.id)


def job_to_charge_once(
    kwargs: Mapping[str, object], response_obj: object, response_cost: float
) -> ChargeOnceJob | None:
    """A finished batch or video reports its whole cost on every poll, so it is charged once per counter."""
    call_type: Final = kwargs.get("call_type")
    if not is_video_status_call_type(call_type):
        return _charged_batch(kwargs, response_obj, response_cost)
    video_id: Final = charged_video_id(
        call_type, kwargs, response_obj.id if isinstance(response_obj, VideoObject) else None, response_cost
    )
    if video_id is None:
        return None
    return ChargeOnceJob(
        kind="video",
        job_id=video_id,
        repeat_poll_cost=repeat_poll_cost(call_type, response_cost, kwargs.get("standard_logging_object")),
    )
