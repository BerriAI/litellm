from collections.abc import AsyncIterator, Iterator, Mapping
from typing import TYPE_CHECKING, Final

import httpx
from pydantic import TypeAdapter

import litellm
from litellm.llms.openai.chat.gpt_transformation import OpenAIChatCompletionStreamingHandler, OpenAIGPTConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse, ModelResponseStream, Usage

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.litellm_core_utils.tokenizer import Encoding

CATALOG_URL: Final = "https://platform.acedata.cloud/api/v1/models/catalog/"
_CATALOG: Final = TypeAdapter(dict[str, object])
_ITEMS: Final = TypeAdapter(tuple[Mapping[str, object], ...])


def request_price(model: str, override: Mapping[str, object] | None) -> float | None:
    info: Final = (
        override
        if override is not None
        else _CATALOG.validate_python(
            litellm.model_cost.get(  # pyright: ignore[reportUnknownMemberType]  # legacy registry has no parameterized row type
                f"acedatacloud/{model.removeprefix('acedatacloud/')}", {}
            )
        )
    )
    fee: Final = info.get("input_cost_per_request")
    return float(fee) if isinstance(fee, (int, float)) else None


def _without_credit_cost(usage: Usage | None) -> None:
    # Ace reports settlement Credits. LiteLLM's numeric cost field represents USD.
    cost: Final[object] = getattr(usage, "cost", None)
    if usage is not None and isinstance(cost, Mapping) and cost.get("currency") == "credit":
        del usage.cost


class _AceChatStream(OpenAIChatCompletionStreamingHandler):
    def chunk_parser(self, chunk: dict[str, object]) -> ModelResponseStream:
        result: Final = super().chunk_parser(chunk)
        _without_credit_cost(getattr(result, "usage", None))
        return result


class AceDataCloudChatConfig(OpenAIGPTConfig):
    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return api_key or get_secret_str("ACEDATACLOUD_API_KEY")

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: "LiteLLMLoggingObj",
        request_data: dict[str, object],
        messages: list[AllMessageValues],
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        encoding: "Encoding | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        response: Final = super().transform_response(
            model,
            raw_response,
            model_response,
            logging_obj,
            request_data,
            messages,
            optional_params,
            litellm_params,
            encoding,
            api_key,
            json_mode,
        )
        _without_credit_cost(getattr(response, "usage", None))
        return response

    def get_model_response_iterator(
        self,
        streaming_response: Iterator[str] | AsyncIterator[str] | ModelResponse,
        sync_stream: bool,
        json_mode: bool | None = False,
    ) -> _AceChatStream:
        return _AceChatStream(streaming_response=streaming_response, sync_stream=sync_stream, json_mode=json_mode)

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str:
        return api_base or "https://api.acedata.cloud/v1"

    def get_models(
        self, api_key: str | None = None, api_base: str | None = None
    ) -> list[str]:  # mutable-ok: provider discovery returns a list
        if api_base and api_base.rstrip("/") not in (
            "https://api.acedata.cloud",
            "https://api.acedata.cloud/v1",
            "https://api.acedata.cloud/openai",
        ):
            return OpenAIGPTConfig().get_models(
                api_key=api_key or get_secret_str("ACEDATACLOUD_API_KEY") or "", api_base=api_base
            )
        response: Final = litellm.module_level_client.get(url=CATALOG_URL)  # pyright: ignore[reportUnknownMemberType]  # legacy HTTP wrapper has untyped query parameters
        response.raise_for_status()
        catalog: Final = _CATALOG.validate_json(response.content)
        items: Final = _ITEMS.validate_python(catalog["items"])
        return [
            f"acedatacloud/{item['id']}"
            for item in items
            if item.get("modality") == "chat" and isinstance(item.get("id"), str)
        ]
