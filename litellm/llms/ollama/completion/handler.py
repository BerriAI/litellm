"""
Ollama /chat/completion calls handled in llm_http_handler.py

[TODO]: migrate embeddings to a base handler as well.
"""

import base64
import struct
from collections.abc import Mapping, Sequence
from typing import Any, Final, Protocol, TypedDict

from typing_extensions import NotRequired, ReadOnly

import litellm
from litellm.llms.ollama.common_utils import OllamaError
from litellm.types.utils import EmbeddingResponse


class TokenEncoder(Protocol):
    """The tokenizer surface used to estimate prompt tokens."""

    def encode(self, text: str, /) -> Sequence[int]: ...


class OllamaEmbeddingResponse(TypedDict):
    """Body of an Ollama ``/api/embed`` response."""

    embeddings: ReadOnly[list[list[float]]]
    prompt_eval_count: ReadOnly[NotRequired[int]]


_RESPONSE_ONLY_PARAMS: Final = frozenset({"encoding_format"})


def _prepare_ollama_embedding_payload(
    model: str, prompts: list[str], optional_params: Mapping[str, object]
) -> dict[str, object]:
    data: Final[dict[str, object]] = {"model": model, "input": prompts}
    special_optional_params: Final = ["truncate", "options", "keep_alive", "dimensions"]

    for k, v in optional_params.items():
        if k in _RESPONSE_ONLY_PARAMS:
            continue
        if k in special_optional_params:
            data[k] = v
        else:
            data.setdefault("options", {})
            if isinstance(data["options"], dict):
                data["options"].update({k: v})
    return data


def _encode_embedding(embedding: Sequence[float], encoding_format: str | None) -> str | list[float]:
    """Ollama only returns float arrays, but a client that asked for base64 will run a
    float32 decoder over the response, shrinking a float list to a quarter of its width."""
    if encoding_format != "base64":
        return list(embedding)
    return base64.b64encode(struct.pack(f"<{len(embedding)}f", *embedding)).decode("utf-8")


def _validate_dimensions(embeddings: Sequence[Sequence[float]], requested_dimensions: int | None) -> None:
    """Ollama truncates to `dimensions` when it can, but silently returns the model's own
    width when asked for more, which otherwise surfaces as a rejected vector store write."""
    if requested_dimensions is None:
        return

    mismatched: Final = next((len(emb) for emb in embeddings if len(emb) != requested_dimensions), None)
    if mismatched is None:
        return

    raise OllamaError(
        status_code=400,
        message=(
            f"Ollama returned embeddings of width {mismatched}, but {requested_dimensions} was requested "
            "via `dimensions`. The model does not support this dimension."
        ),
        headers={},
    )


def _process_ollama_embedding_response(
    response_json: OllamaEmbeddingResponse,
    prompts: list[str],
    model: str,
    model_response: EmbeddingResponse,
    logging_obj: Any,
    encoding: TokenEncoder | None,
    optional_params: Mapping[str, object] | None = None,
) -> EmbeddingResponse:
    embeddings: Final[list[list[float]]] = response_json["embeddings"]
    params: Final[Mapping[str, object]] = optional_params or {}

    requested_dimensions: Final = params.get("dimensions")
    _validate_dimensions(embeddings, requested_dimensions if isinstance(requested_dimensions, int) else None)

    encoding_format: Final = params.get("encoding_format")
    resolved_format: Final = encoding_format if isinstance(encoding_format, str) else None
    output_data: Final = [
        {"object": "embedding", "index": idx, "embedding": _encode_embedding(emb, resolved_format)}
        for idx, emb in enumerate(embeddings)
    ]

    input_tokens = response_json.get("prompt_eval_count", None)

    if input_tokens is None:
        if encoding is not None:
            input_tokens = len(encoding.encode("".join(prompts)))
            if logging_obj:
                logging_obj.debug("Ollama response missing prompt_eval_count; estimated with encoding.")
        else:
            input_tokens = 0
            if logging_obj:
                logging_obj.warning("Missing prompt_eval_count and no encoding provided; defaulted to 0.")

    model_response.object = "list"
    model_response.data = output_data
    model_response.model = "ollama/" + model
    model_response.usage = litellm.Usage(
        prompt_tokens=input_tokens,
        completion_tokens=0,
        total_tokens=input_tokens,
        prompt_tokens_details=None,
        completion_tokens_details=None,
    )
    return model_response


async def ollama_aembeddings(
    api_base: str,
    model: str,
    prompts: list[str],
    model_response: EmbeddingResponse,
    optional_params: Mapping[str, object],
    logging_obj: Any,
    encoding: TokenEncoder | None,
):
    if not api_base.endswith("/api/embed"):
        api_base += "/api/embed"

    data: Final = _prepare_ollama_embedding_payload(model, prompts, optional_params)

    response: Final = await litellm.module_level_aclient.post(url=api_base, json=data)
    response_json: Final[OllamaEmbeddingResponse] = response.json()

    return _process_ollama_embedding_response(
        response_json=response_json,
        prompts=prompts,
        model=model,
        model_response=model_response,
        logging_obj=logging_obj,
        encoding=encoding,
        optional_params=optional_params,
    )


def ollama_embeddings(
    api_base: str,
    model: str,
    prompts: list[str],
    optional_params: Mapping[str, object],
    model_response: EmbeddingResponse,
    logging_obj: Any,
    encoding: TokenEncoder | None = None,
):
    if not api_base.endswith("/api/embed"):
        api_base += "/api/embed"

    data: Final = _prepare_ollama_embedding_payload(model, prompts, optional_params)

    response: Final = litellm.module_level_client.post(url=api_base, json=data)
    response_json: Final[OllamaEmbeddingResponse] = response.json()

    return _process_ollama_embedding_response(
        response_json=response_json,
        prompts=prompts,
        model=model,
        model_response=model_response,
        logging_obj=logging_obj,
        encoding=encoding,
        optional_params=optional_params,
    )
