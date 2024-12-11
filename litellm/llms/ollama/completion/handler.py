import asyncio
import json
import time
import traceback
import types
import uuid
from copy import deepcopy
from itertools import chain
from typing import Any, Dict, List, Optional

import aiohttp
import httpx  # type: ignore
import requests  # type: ignore

import litellm
from litellm import verbose_logger
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import ModelInfo, ProviderField, StreamingChoices

from litellm.litellm_core_utils.prompt_templates.factory import custom_prompt, prompt_factory
from ..common_utils import OllamaError
from .transformation import OllamaConfig


# ollama wants plain base64 jpeg/png files as images.  strip any leading dataURI
# and convert to jpeg if necessary.
def _convert_image(image):
    import base64
    import io

    try:
        from PIL import Image
    except Exception:
        raise Exception(
            "ollama image conversion failed please run `pip install Pillow`"
        )

    orig = image
    if image.startswith("data:"):
        image = image.split(",")[-1]
    try:
        image_data = Image.open(io.BytesIO(base64.b64decode(image)))
        if image_data.format in ["JPEG", "PNG"]:
            return image
    except Exception:
        return orig
    jpeg_image = io.BytesIO()
    image_data.convert("RGB").save(jpeg_image, "JPEG")
    jpeg_image.seek(0)
    return base64.b64encode(jpeg_image.getvalue()).decode("utf-8")


# ollama implementation
def get_ollama_response(
    model_response: litellm.ModelResponse,
    model: str,
    prompt: str,
    optional_params: dict,
    logging_obj: Any,
    encoding: Any,
    acompletion: bool = False,
    api_base="http://localhost:11434",
):
    if api_base.endswith("/api/generate"):
        url = api_base
    else:
        url = f"{api_base}/api/generate"

    ## Load Config
    config = litellm.OllamaConfig.get_config()
    for k, v in config.items():
        if (
            k not in optional_params
        ):  # completion(top_k=3) > cohere_config(top_k=3) <- allows for dynamic variables to be passed in
            optional_params[k] = v

    stream = optional_params.pop("stream", False)
    format = optional_params.pop("format", None)
    images = optional_params.pop("images", None)
    data = {
        "model": model,
        "prompt": prompt,
        "options": optional_params,
        "stream": stream,
    }
    if format is not None:
        data["format"] = format
    if images is not None:
        data["images"] = [_convert_image(image) for image in images]

    ## LOGGING
    logging_obj.pre_call(
        input=None,
        api_key=None,
        additional_args={
            "api_base": url,
            "complete_input_dict": data,
            "headers": {},
            "acompletion": acompletion,
        },
    )
    if acompletion is True:
        if stream is True:
            response = ollama_async_streaming(
                url=url,
                data=data,
                model_response=model_response,
                encoding=encoding,
                logging_obj=logging_obj,
            )
        else:
            response = ollama_acompletion(
                url=url,
                data=data,
                model_response=model_response,
                encoding=encoding,
                logging_obj=logging_obj,
            )
        return response
    elif stream is True:
        return ollama_completion_stream(url=url, data=data, logging_obj=logging_obj)

    response = requests.post(
        url=f"{url}", json={**data, "stream": stream}, timeout=litellm.request_timeout
    )
    if response.status_code != 200:
        raise OllamaError(
            status_code=response.status_code,
            message=response.text,
            headers=dict(response.headers),
        )

    ## LOGGING
    logging_obj.post_call(
        input=prompt,
        api_key="",
        original_response=response.text,
        additional_args={
            "headers": None,
            "api_base": api_base,
        },
    )

    response_json = response.json()

    ## RESPONSE OBJECT
    model_response.choices[0].finish_reason = "stop"
    if data.get("format", "") == "json":
        function_call = json.loads(response_json["response"])
        message = litellm.Message(
            content=None,
            tool_calls=[
                {
                    "id": f"call_{str(uuid.uuid4())}",
                    "function": {
                        "name": function_call["name"],
                        "arguments": json.dumps(function_call["arguments"]),
                    },
                    "type": "function",
                }
            ],
        )
        model_response.choices[0].message = message  # type: ignore
        model_response.choices[0].finish_reason = "tool_calls"
    else:
        model_response.choices[0].message.content = response_json["response"]  # type: ignore
    model_response.created = int(time.time())
    model_response.model = "ollama/" + model
    prompt_tokens = response_json.get("prompt_eval_count", len(encoding.encode(prompt, disallowed_special=())))  # type: ignore
    completion_tokens = response_json.get(
        "eval_count", len(response_json.get("message", dict()).get("content", ""))
    )
    setattr(
        model_response,
        "usage",
        litellm.Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
    return model_response


def ollama_completion_stream(url, data, logging_obj):
    with httpx.stream(
        url=url, json=data, method="POST", timeout=litellm.request_timeout
    ) as response:
        try:
            if response.status_code != 200:
                raise OllamaError(
                    status_code=response.status_code,
                    message=str(response.read()),
                    headers=response.headers,
                )

            streamwrapper = litellm.CustomStreamWrapper(
                completion_stream=response.iter_lines(),
                model=data["model"],
                custom_llm_provider="ollama",
                logging_obj=logging_obj,
            )
            # If format is JSON, this was a function call
            # Gather all chunks and return the function call as one delta to simplify parsing
            if data.get("format", "") == "json":
                first_chunk = next(streamwrapper)
                content_chunks = []
                for chunk in chain([first_chunk], streamwrapper):
                    content_chunk = chunk.choices[0]
                    if (
                        isinstance(content_chunk, StreamingChoices)
                        and hasattr(content_chunk, "delta")
                        and hasattr(content_chunk.delta, "content")
                        and content_chunk.delta.content is not None
                    ):
                        content_chunks.append(content_chunk.delta.content)
                response_content = "".join(content_chunks)

                function_call = json.loads(response_content)
                delta = litellm.utils.Delta(
                    content=None,
                    tool_calls=[
                        {
                            "id": f"call_{str(uuid.uuid4())}",
                            "function": {
                                "name": function_call["name"],
                                "arguments": json.dumps(function_call["arguments"]),
                            },
                            "type": "function",
                        }
                    ],
                )
                model_response = first_chunk
                model_response.choices[0].delta = delta  # type: ignore
                model_response.choices[0].finish_reason = "tool_calls"
                yield model_response
            else:
                for transformed_chunk in streamwrapper:
                    yield transformed_chunk
        except Exception as e:
            raise e


async def ollama_async_streaming(url, data, model_response, encoding, logging_obj):
    try:
        _async_http_client = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.OLLAMA
        )
        client = _async_http_client.client
        async with client.stream(
            url=f"{url}", json=data, method="POST", timeout=litellm.request_timeout
        ) as response:
            if response.status_code != 200:
                raise OllamaError(
                    status_code=response.status_code,
                    message=str(await response.aread()),
                    headers=dict(response.headers),
                )

            streamwrapper = litellm.CustomStreamWrapper(
                completion_stream=response.aiter_lines(),
                model=data["model"],
                custom_llm_provider="ollama",
                logging_obj=logging_obj,
            )

            # If format is JSON, this was a function call
            # Gather all chunks and return the function call as one delta to simplify parsing
            if data.get("format", "") == "json":
                first_chunk = await anext(streamwrapper)  # noqa F821
                chunk_choice = first_chunk.choices[0]
                if (
                    isinstance(chunk_choice, StreamingChoices)
                    and hasattr(chunk_choice, "delta")
                    and hasattr(chunk_choice.delta, "content")
                ):
                    first_chunk_content = chunk_choice.delta.content or ""
                else:
                    first_chunk_content = ""

                content_chunks = []
                async for chunk in streamwrapper:
                    chunk_choice = chunk.choices[0]
                    if (
                        isinstance(chunk_choice, StreamingChoices)
                        and hasattr(chunk_choice, "delta")
                        and hasattr(chunk_choice.delta, "content")
                    ):
                        content_chunks.append(chunk_choice.delta.content)
                response_content = first_chunk_content + "".join(content_chunks)
                function_call = json.loads(response_content)
                delta = litellm.utils.Delta(
                    content=None,
                    tool_calls=[
                        {
                            "id": f"call_{str(uuid.uuid4())}",
                            "function": {
                                "name": function_call["name"],
                                "arguments": json.dumps(function_call["arguments"]),
                            },
                            "type": "function",
                        }
                    ],
                )
                model_response = first_chunk
                model_response.choices[0].delta = delta  # type: ignore
                model_response.choices[0].finish_reason = "tool_calls"
                yield model_response
            else:
                async for transformed_chunk in streamwrapper:
                    yield transformed_chunk
    except Exception as e:
        raise e  # don't use verbose_logger.exception, if exception is raised


async def ollama_acompletion(
    url, data, model_response: litellm.ModelResponse, encoding, logging_obj
):
    data["stream"] = False
    try:
        timeout = aiohttp.ClientTimeout(total=litellm.request_timeout)  # 10 minutes
        async with aiohttp.ClientSession(timeout=timeout) as session:
            resp = await session.post(url, json=data)

            if resp.status != 200:
                text = await resp.text()
                raise OllamaError(
                    status_code=resp.status,
                    message=text,
                    headers=dict(resp.headers),
                )

            ## LOGGING
            logging_obj.post_call(
                input=data["prompt"],
                api_key="",
                original_response=resp.text,
                additional_args={
                    "headers": None,
                    "api_base": url,
                },
            )

            response_json = await resp.json()
            ## RESPONSE OBJECT
            model_response.choices[0].finish_reason = "stop"
            if data.get("format", "") == "json":
                function_call = json.loads(response_json["response"])
                message = litellm.Message(
                    content=None,
                    tool_calls=[
                        {
                            "id": f"call_{str(uuid.uuid4())}",
                            "function": {
                                "name": function_call.get(
                                    "name", function_call.get("function", None)
                                ),
                                "arguments": json.dumps(function_call["arguments"]),
                            },
                            "type": "function",
                        }
                    ],
                )
                model_response.choices[0].message = message  # type: ignore
                model_response.choices[0].finish_reason = "tool_calls"
            else:
                model_response.choices[0].message.content = response_json["response"]  # type: ignore
            model_response.created = int(time.time())
            model_response.model = "ollama/" + data["model"]
            prompt_tokens = response_json.get("prompt_eval_count", len(encoding.encode(data["prompt"], disallowed_special=())))  # type: ignore
            completion_tokens = response_json.get(
                "eval_count",
                len(response_json.get("message", dict()).get("content", "")),
            )
            setattr(
                model_response,
                "usage",
                litellm.Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )
            return model_response
    except Exception as e:
        raise e  # don't use verbose_logger.exception, if exception is raised


async def ollama_aembeddings(
    api_base: str,
    model: str,
    prompts: List[str],
    model_response: litellm.EmbeddingResponse,
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

    timeout = aiohttp.ClientTimeout(total=litellm.request_timeout)  # 10 minutes
    async with aiohttp.ClientSession(timeout=timeout) as session:
        ## LOGGING
        logging_obj.pre_call(
            input=None,
            api_key=None,
            additional_args={
                "api_base": url,
                "complete_input_dict": data,
                "headers": {},
            },
        )

        response = await session.post(url, json=data)

        if response.status != 200:
            text = await response.text()
            raise OllamaError(
                status_code=response.status,
                message=text,
                headers=dict(response.headers),
            )

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
    model_response: litellm.EmbeddingResponse,
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
