# What is this?
## Handler file for OpenAI-like endpoints.
## Allows jina ai embedding calls - which don't allow 'encoding_format' in payload.

import json
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import httpx
from pydantic import TypeAdapter

import litellm
from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,
)
from litellm.types.utils import EmbeddingResponse

from ..common_utils import OpenAILikeBase, OpenAILikeError

_KEY_HEADERS_ADAPTER: Final = TypeAdapter(Mapping[str, str])
_EXTRA_HEADERS_ADAPTER: Final = TypeAdapter(Mapping[str, object])
_TRANSPORT_HEADER_NAMES: Final = frozenset(
    {
        "host",
        "content-length",
        "transfer-encoding",
        "connection",
        "keep-alive",
        "proxy-connection",
        "te",
        "trailer",
        "upgrade",
        "expect",
    }
)
_HEADER_NAME: Final = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")
_HEADER_VALUE: Final = re.compile(r"[\t\x20-\x7e]*")


def _header_text(name: str, value: object) -> str | None:
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        return None
    if _HEADER_NAME.fullmatch(name) is None:
        return None
    return str(value).strip(" \t\r\n")


def _extra_headers_as_text(extra_headers: Mapping[str, object], model: str, drop_params: bool) -> Mapping[str, str]:
    caller_set: Final = {
        name: value for name, value in extra_headers.items() if name.lower() not in _TRANSPORT_HEADER_NAMES
    }
    texts: Final = MappingProxyType({name: _header_text(name, value) for name, value in caller_set.items()})
    refused: Final = sorted(name for name, text in texts.items() if text is None)
    if refused and not drop_params:
        raise litellm.UnsupportedParamsError(
            message=f"extra_headers {', '.join(map(repr, refused))} cannot be sent. A header needs a name made of "
            "letters, digits or !#$%&'*+-.^_`|~ and a text or number value. "
            "To drop them from the call, set `litellm.drop_params = True`.",
            model=model,
        )
    named: Final = MappingProxyType({name: text for name, text in texts.items() if text is not None})
    unsendable: Final = sorted(name for name, text in named.items() if _HEADER_VALUE.fullmatch(text) is None)
    if unsendable:
        verbose_logger.warning(
            "extra_headers %s dropped for model %r, since a header value must be printable ASCII on one line",
            unsendable,
            model,
        )
    return MappingProxyType({name: text for name, text in named.items() if name not in unsendable})


class OpenAILikeEmbeddingHandler(OpenAILikeBase):
    def __init__(self, **kwargs):
        pass

    async def aembedding(
        self,
        input: list,
        data: dict,
        model_response: EmbeddingResponse,
        timeout: float,
        api_key: str,
        api_base: str,
        logging_obj,
        headers: dict,
        client=None,
    ) -> EmbeddingResponse:
        response = None
        try:
            if client is None or not isinstance(client, AsyncHTTPHandler):
                async_client = get_async_httpx_client(
                    llm_provider=litellm.LlmProviders.OPENAI,
                    params={"timeout": timeout},
                )
            else:
                async_client = client
            try:
                response = await async_client.post(
                    api_base,
                    headers=headers,
                    data=json.dumps(data),
                )

                response.raise_for_status()

                response_json: Final = response.json()
            except httpx.HTTPStatusError as e:
                raise OpenAILikeError(
                    status_code=e.response.status_code,
                    message=e.response.text if e.response else str(e),
                )
            except httpx.TimeoutException:
                raise OpenAILikeError(status_code=408, message="Timeout error occurred.")
            except Exception as e:
                raise OpenAILikeError(status_code=500, message=str(e))

            ## LOGGING
            logging_obj.post_call(
                input=input,
                api_key=api_key,
                additional_args={"complete_input_dict": data},
                original_response=response_json,
            )
            return EmbeddingResponse(**response_json)
        except Exception as e:
            ## LOGGING
            logging_obj.post_call(
                input=input,
                api_key=api_key,
                original_response=str(e),
            )
            raise e

    def embedding(
        self,
        model: str,
        input: list,
        timeout: float,
        logging_obj,
        api_key: str | None,
        api_base: str | None,
        optional_params: dict,
        model_response: EmbeddingResponse | None = None,
        client=None,
        aembedding=None,
        custom_endpoint: bool | None = None,
        headers: dict | None = None,
        drop_params: bool = False,
    ) -> EmbeddingResponse:
        api_base, headers = self._validate_environment(
            api_base=api_base,
            api_key=api_key,
            endpoint_type="embeddings",
            headers=headers,
            custom_endpoint=custom_endpoint,
        )
        model = model
        data: Final = {
            "model": model,
            "input": input,
            **{k: v for k, v in optional_params.items() if k != "extra_headers" and v not in (None, "")},
        }
        extra_headers: Final = _extra_headers_as_text(
            _EXTRA_HEADERS_ADAPTER.validate_python(optional_params.get("extra_headers") or {}), model, drop_params
        )
        key_headers: Final = _KEY_HEADERS_ADAPTER.validate_python(headers)
        extra_header_names: Final = frozenset(name.lower() for name in extra_headers)
        request_headers: Final = {
            **{name: value for name, value in key_headers.items() if name.lower() not in extra_header_names},
            **extra_headers,
        }

        ## LOGGING
        logging_obj.pre_call(
            input=input,
            api_key=api_key,
            additional_args={"complete_input_dict": data, "api_base": api_base},
        )

        if aembedding is True:
            return self.aembedding(
                data=data,
                input=input,
                logging_obj=logging_obj,
                model_response=model_response,
                api_base=api_base,
                api_key=api_key,
                timeout=timeout,
                client=client,
                headers=request_headers,
            )
        if client is None or isinstance(client, AsyncHTTPHandler):
            self.client = HTTPHandler(timeout=timeout)
        else:
            self.client = client

        ## EMBEDDING CALL
        try:
            response: Final = self.client.post(
                api_base,
                headers=request_headers,
                data=json.dumps(data),
            )

            response.raise_for_status()

            response_json: Final = response.json()
        except httpx.HTTPStatusError as e:
            raise OpenAILikeError(
                status_code=e.response.status_code,
                message=e.response.text,
            )
        except httpx.TimeoutException:
            raise OpenAILikeError(status_code=408, message="Timeout error occurred.")
        except Exception as e:
            raise OpenAILikeError(status_code=500, message=str(e))

        ## LOGGING
        logging_obj.post_call(
            input=input,
            api_key=api_key,
            additional_args={"complete_input_dict": data},
            original_response=response_json,
        )

        return litellm.EmbeddingResponse(**response_json)
