import types
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, cast

import httpx

from litellm.types.llms.openai import (
    ResponseInputParam,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
    ResponsesAPIStreamingResponse,
)
from litellm.types.responses.main import *
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

# Codex CLI ships tool definitions inside an input item of this type rather than the
# top-level ``tools`` array. One parser is shared by every caller so they cannot
# disagree: the chat bridge lifts these tools so the model sees them, and the
# authorization/guardrail extractors read them so a nested tool cannot slip past an
# allowlist that only inspects ``tools``.
ADDITIONAL_TOOLS_INPUT_ITEM_TYPE: Final = "additional_tools"
NAMESPACE_TOOL_TYPE: Final = "namespace"


def _tools_held_by(item: object, container_type: str) -> tuple[object, ...] | None:
    """Tools held by ``item`` when it is a container of ``container_type``, else ``None``.

    Returns an empty tuple for a malformed container, so callers can still recognise it.
    """
    if not isinstance(item, Mapping):
        return None
    entry: Final = cast("Mapping[str, object]", item)  # cast-ok: request items reach here as untyped mappings
    if entry.get("type") != container_type:
        return None
    nested: Final = entry.get("tools")
    if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
        return tuple(cast("Sequence[object]", nested))  # cast-ok: nested tool entries are untyped
    return ()


def additional_tools_of(item: object) -> tuple[object, ...] | None:
    """Nested tools when ``item`` is an ``additional_tools`` input item, else ``None``."""
    return _tools_held_by(item, ADDITIONAL_TOOLS_INPUT_ITEM_TYPE)


def flatten_namespace_tools(tools: Iterable[object]) -> tuple[object, ...]:
    """Expand ``namespace`` containers into the tools they hold, leaving others as-is.

    Codex groups its tools under namespaces (``functions``, ``collaboration``). A
    namespace entry carries only the group name, so anything reading tool *names* --
    allowlist and guardrail extraction -- must look at the leaves or it sees nothing
    enforceable at all.
    """
    flattened: Final[list[object]] = []  # mutable-ok: accumulator, returned as a tuple
    for tool in tools:
        members = _tools_held_by(tool, NAMESPACE_TOOL_TYPE)  # rebind-ok: loop-local; Final is invalid in a loop
        if members is None:
            flattened.append(tool)
        else:
            flattened.extend(members)
    return tuple(flattened)


def additional_tools_in(input: object) -> tuple[object, ...]:
    """Every tool nested in ``additional_tools`` items, leaving ``input`` untouched.

    For callers that must see the effective tool list without rewriting the request,
    such as allowlist and guardrail extraction.
    """
    if not isinstance(input, list):
        return ()
    return tuple(tool for item in input for tool in (additional_tools_of(item) or ()))


if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    from ..chat.transformation import BaseLLMException as _BaseLLMException

    LiteLLMLoggingObj = _LiteLLMLoggingObj
    BaseLLMException = _BaseLLMException
else:
    LiteLLMLoggingObj = Any
    BaseLLMException = Any


class BaseResponsesAPIConfig(ABC):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def custom_llm_provider(self) -> LlmProviders:
        pass

    @classmethod
    def get_config(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not k.startswith("_abc")
            and not isinstance(
                v,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    classmethod,
                    staticmethod,
                ),
            )
            and v is not None
        }

    def supports_native_file_search(self) -> bool:
        """Return True if this provider handles the file_search tool natively.

        Override in provider subclasses that support file_search without
        LiteLLM emulation (e.g. OpenAI, Azure OpenAI).
        """
        return False

    def sign_request(
        self,
        headers: dict,
        optional_params: dict,
        request_data: dict,
        api_base: str,
        api_key: str | None = None,
        model: str | None = None,
        stream: bool | None = None,
        fake_stream: bool | None = None,
    ) -> tuple[dict, bytes | None]:
        """Sign the request after the body is finalized.

        Default is a no-op (returns headers unchanged, no signed body). Providers
        whose endpoint requires request signing (e.g. Bedrock Mantle SigV4)
        override this and return the signed body bytes so the handler sends those
        exact bytes.
        """
        return headers, None

    @abstractmethod
    def get_supported_openai_params(self, model: str) -> list:
        pass

    @abstractmethod
    def map_openai_params(
        self,
        response_api_optional_params: ResponsesAPIOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict:
        pass

    @abstractmethod
    def validate_environment(self, headers: dict, model: str, litellm_params: GenericLiteLLMParams | None) -> dict:
        return {}

    @abstractmethod
    def get_complete_url(
        self,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        """
        OPTIONAL

        Get the complete url for the request

        Some providers need `model` in `api_base`
        """
        if api_base is None:
            raise ValueError("api_base is required")
        return api_base

    @abstractmethod
    def transform_responses_api_request(
        self,
        model: str,
        input: str | ResponseInputParam,
        response_api_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> dict:
        pass

    @abstractmethod
    def transform_response_api_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ResponsesAPIResponse:
        pass

    @abstractmethod
    def transform_streaming_response(
        self,
        model: str,
        parsed_chunk: dict,
        logging_obj: LiteLLMLoggingObj,
    ) -> ResponsesAPIStreamingResponse:
        """
        Transform a parsed streaming response chunk into a ResponsesAPIStreamingResponse
        """

    #########################################################
    ########## DELETE RESPONSE API TRANSFORMATION ##############
    #########################################################
    @abstractmethod
    def transform_delete_response_api_request(
        self,
        response_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        pass

    @abstractmethod
    def transform_delete_response_api_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> DeleteResponseResult:
        pass

    #########################################################
    ########## END DELETE RESPONSE API TRANSFORMATION #######
    #########################################################

    #########################################################
    ########## GET RESPONSE API TRANSFORMATION ###############
    #########################################################
    @abstractmethod
    def transform_get_response_api_request(
        self,
        response_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        pass

    @abstractmethod
    def transform_get_response_api_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ResponsesAPIResponse:
        pass

    #########################################################
    ########## LIST INPUT ITEMS API TRANSFORMATION ##########
    #########################################################
    @abstractmethod
    def transform_list_input_items_request(
        self,
        response_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        after: str | None = None,
        before: str | None = None,
        include: list[str] | None = None,
        limit: int = 20,
        order: Literal["asc", "desc"] = "desc",
    ) -> tuple[str, dict]:
        pass

    @abstractmethod
    def transform_list_input_items_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> dict:
        pass

    #########################################################
    ########## END GET RESPONSE API TRANSFORMATION ##########
    #########################################################

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        from ..chat.transformation import BaseLLMException

        raise BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

    def should_fake_stream(
        self,
        model: str | None,
        stream: bool | None,
        custom_llm_provider: str | None = None,
    ) -> bool:
        """Returns True if litellm should fake a stream for the given model and stream value"""
        return False

    def supports_native_websocket(self) -> bool:
        """
        Returns True if the provider has a native WebSocket endpoint for Responses API.

        Providers with native websocket support can connect directly to wss:// endpoints.
        Providers without native support will use the ManagedResponsesWebSocketHandler
        which makes HTTP streaming calls and forwards events over the websocket.

        Default: False (use managed websocket handler)
        """
        return False

    def get_websocket_url(
        self,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        """
        Return the wss:// URL for the provider's native Responses WebSocket endpoint.

        Defaults to converting the HTTP URL from get_complete_url. Providers whose
        WebSocket path differs from their HTTP path (e.g. Azure uses
        /openai/v1/responses without api-version) should override this.
        """
        http_url: Final = self.get_complete_url(api_base=api_base, litellm_params=litellm_params)
        return http_url.replace("https://", "wss://").replace("http://", "ws://")

    def model_in_websocket_url(self) -> bool:
        """
        Return True if the model should be appended as a ?model= query param to
        the WebSocket URL. Providers that identify the model via the request body
        (e.g. Azure Responses API) should override this to return False.
        """
        return True

    #########################################################
    ########## CANCEL RESPONSE API TRANSFORMATION ##########
    #########################################################
    @abstractmethod
    def transform_cancel_response_api_request(
        self,
        response_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        pass

    @abstractmethod
    def transform_cancel_response_api_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ResponsesAPIResponse:
        pass

    #########################################################
    ########## END CANCEL RESPONSE API TRANSFORMATION #######
    #########################################################

    #########################################################
    ########## COMPACT RESPONSE API TRANSFORMATION ##########
    #########################################################
    @abstractmethod
    def transform_compact_response_api_request(
        self,
        model: str,
        input: str | ResponseInputParam,
        response_api_optional_request_params: dict,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        pass

    @abstractmethod
    def transform_compact_response_api_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ResponsesAPIResponse:
        pass

    #########################################################
    ########## END COMPACT RESPONSE API TRANSFORMATION ######
    #########################################################

    @staticmethod
    def strip_custom_tool_call_namespace_from_responses_input(
        input: str | ResponseInputParam,
    ) -> str | ResponseInputParam:
        """
        Remove ``namespace`` from ``custom_tool_call`` input items.
        """
        if not isinstance(input, list):
            return input
        out: Final[list[Any]] = []
        for item in input:
            if isinstance(item, dict) and item.get("type") == "custom_tool_call":
                out.append({k: v for k, v in item.items() if k != "namespace"})
            else:
                out.append(item)
        return cast(ResponseInputParam, out)

    @staticmethod
    def normalize_responses_api_request_dict(data: dict[str, Any]) -> dict[str, Any]:
        """Apply provider-agnostic fixes to an outbound Responses API request dict."""
        if not isinstance(data, dict) or "input" not in data:
            return data
        return {
            **data,
            "input": BaseResponsesAPIConfig.strip_custom_tool_call_namespace_from_responses_input(data["input"]),
        }
