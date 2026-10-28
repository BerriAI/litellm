import io
import json
from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from os import PathLike
from typing import Final, TypeVar

from openai.types import Batch
from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.types.llms.openai import FileTypes, OpenAIFilesPurpose
from litellm.types.utils import CallTypes, LlmProviders

_DECODED_JSONL_ROW: Final = TypeAdapter(object)
_DEPLOYMENT_FIELD: Final = TypeAdapter(Mapping[str, object])
_ENDPOINT_PATHS: Final = TypeAdapter(tuple[str, ...])

BATCH_ENDPOINT: Final = "/v1/batch"
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


def batch_charged_once_marker_key(spend_key: str, batch_id: str) -> str:
    return f"{spend_key}:batch:{batch_id}"


def batch_id_to_charge_once(call_type: object, response_obj: object, response_cost: float) -> str | None:
    """A finished batch reports its whole cost on every poll, so its id is charged once per counter."""
    if response_cost <= 0 or not is_batch_retrieve_call_type(call_type):
        return None
    return response_obj.id if isinstance(response_obj, Batch) else None
