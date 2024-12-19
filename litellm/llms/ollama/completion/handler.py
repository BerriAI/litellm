"""
Ollama /chat/completion calls handled in llm_http_handler.py

[TODO]: migrate embeddings to a base handler as well.
"""

import asyncio
from typing import Any, Dict, List

import litellm
from litellm.types.utils import EmbeddingResponse

# ollama wants plain base64 jpeg/png files as images.  strip any leading dataURI
# and convert to jpeg if necessary.


async def ollama_aembeddings(
    api_base: str,
    model: str,
    prompts: List[str],
    model_response: EmbeddingResponse,
    optional_params: dict,
    logging_obj: Any,
    encoding: Any,
):
    if api_base.endswith("/api/embed"):
        url = api_base
    else:
        url = f"{api_base}/api/embed"

    ## Load Config
    config = litellm.OllamaConfig.get_config()
    for k, v in config.items():
        if (
            k not in optional_params
        ):  # completion(top_k=3) > cohere_config(top_k=3) <- allows for dynamic variables to be passed in
            optional_params[k] = v

    data: Dict[str, Any] = {"model": model, "input": prompts}
    special_optional_params = ["truncate", "options", "keep_alive"]

    for k, v in optional_params.items():
        if k in special_optional_params:
            data[k] = v
        else:
            # Ensure "options" is a dictionary before updating it
            data.setdefault("options", {})
            if isinstance(data["options"], dict):
                data["options"].update({k: v})
    total_input_tokens = 0
    output_data = []

    response = await litellm.module_level_aclient.post(url=url, json=data)

    response_json = await response.json()

    embeddings: List[List[float]] = response_json["embeddings"]
    for idx, emb in enumerate(embeddings):
        output_data.append({"object": "embedding", "index": idx, "embedding": emb})

    input_tokens = response_json.get("prompt_eval_count") or len(
        encoding.encode("".join(prompt for prompt in prompts))
    )
    total_input_tokens += input_tokens

    model_response.object = "list"
    model_response.data = output_data
    model_response.model = "ollama/" + model
    setattr(
        model_response,
        "usage",
        litellm.Usage(
            prompt_tokens=total_input_tokens,
            completion_tokens=total_input_tokens,
            total_tokens=total_input_tokens,
            prompt_tokens_details=None,
            completion_tokens_details=None,
        ),
    )
    return model_response


def ollama_embeddings(
    api_base: str,
    model: str,
    prompts: list,
    optional_params: dict,
    model_response: EmbeddingResponse,
    logging_obj: Any,
    encoding=None,
):
    return asyncio.run(
        ollama_aembeddings(
            api_base=api_base,
            model=model,
            prompts=prompts,
            model_response=model_response,
            optional_params=optional_params,
            logging_obj=logging_obj,
            encoding=encoding,
        )
    )
