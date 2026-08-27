"""
Ollama /chat/completion calls handled in llm_http_handler.py

[TODO]: migrate embeddings to a base handler as well.
"""

from typing import Any, Final

import litellm
from litellm.types.utils import EmbeddingResponse


def _prepare_ollama_embedding_payload(
    model: str, prompts: list[str], optional_params: dict[str, Any]
) -> dict[str, Any]:
    data: Final[dict[str, Any]] = {"model": model, "input": prompts}
    special_optional_params: Final = ["truncate", "options", "keep_alive", "dimensions"]

    for k, v in optional_params.items():
        if k in special_optional_params:
            data[k] = v
        else:
            data.setdefault("options", {})
            if isinstance(data["options"], dict):
                data["options"].update({k: v})
    return data


def _process_ollama_embedding_response(
    response_json: dict,
    prompts: list[str],
    model: str,
    model_response: EmbeddingResponse,
    logging_obj: Any,
    encoding: Any,
) -> EmbeddingResponse:
    output_data: Final = []
    embeddings: Final[list[list[float]]] = response_json["embeddings"]

    for idx, emb in enumerate(embeddings):
        output_data.append({"object": "embedding", "index": idx, "embedding": emb})

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
    optional_params: dict,
    logging_obj: Any,
    encoding: Any,
):
    if not api_base.endswith("/api/embed"):
        api_base += "/api/embed"

    data: Final = _prepare_ollama_embedding_payload(model, prompts, optional_params)

    response: Final = await litellm.module_level_aclient.post(url=api_base, json=data)
    response_json: Final = response.json()

    return _process_ollama_embedding_response(
        response_json=response_json,
        prompts=prompts,
        model=model,
        model_response=model_response,
        logging_obj=logging_obj,
        encoding=encoding,
    )


def ollama_embeddings(
    api_base: str,
    model: str,
    prompts: list[str],
    optional_params: dict,
    model_response: EmbeddingResponse,
    logging_obj: Any,
    encoding: Any = None,
):
    if not api_base.endswith("/api/embed"):
        api_base += "/api/embed"

    data: Final = _prepare_ollama_embedding_payload(model, prompts, optional_params)

    response: Final = litellm.module_level_client.post(url=api_base, json=data)
    response_json: Final = response.json()

    return _process_ollama_embedding_response(
        response_json=response_json,
        prompts=prompts,
        model=model,
        model_response=model_response,
        logging_obj=logging_obj,
        encoding=encoding,
    )
