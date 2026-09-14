from __future__ import annotations

import re
from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol, TypeAlias

from pydantic import TypeAdapter, ValidationError

from litellm.types.utils import CallTypes

from ..base_utils import BaseLLMModelInfo

if TYPE_CHECKING:
    from httpx import URL, Headers, Response

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.llms.openai import ResponsesAPIResponse, ResponsesTerminalEvent
    from litellm.types.rerank import RerankResponse
    from litellm.types.utils import CostResponseTypes, StandardPassThroughResponseObject

    from ..chat.transformation import BaseLLMException
    from ..ocr.transformation import OCRResponse

    LoggedRelayResponse: TypeAlias = CostResponseTypes | RerankResponse | ResponsesAPIResponse | ResponsesTerminalEvent


RELAYED_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object])


def strip_leading_model_segment(endpoint: str, model_names: tuple[str, ...]) -> str:
    path: Final = endpoint.lstrip("/")
    for model_name in model_names:
        if not model_name:
            continue
        if path == model_name:
            return ""
        if path.startswith(f"{model_name}/"):
            return path[len(model_name) + 1 :]
    return path


def replace_path_segment(endpoint: str, segment: str, replacement: str) -> str:
    bounded_segment: Final = re.compile(rf"(?<![^/]){re.escape(segment)}(?![^/:])")
    return bounded_segment.sub(lambda _: replacement, endpoint)


def relayed_json_object(httpx_response: Response) -> Mapping[str, object] | None:
    if httpx_response.status_code != 200:
        return None
    try:
        return RELAYED_JSON_OBJECT.validate_python(httpx_response.json())
    except (ValueError, ValidationError):
        return None


@dataclass(frozen=True, slots=True)
class RelayShape:
    path_suffix: str
    call_type: CallTypes
    parse: Callable[[Mapping[str, object]], LoggedRelayResponse]


def logged_relay_shape(
    shapes: Sequence[RelayShape], httpx_response: Response, logging_obj: LiteLLMLoggingObj, endpoint: str
) -> LoggedRelayResponse | None:
    relayed_path: Final = f"/{endpoint.strip('/')}"
    shape: Final = next((candidate for candidate in shapes if relayed_path.endswith(candidate.path_suffix)), None)
    body: Final = relayed_json_object(httpx_response) if shape else None
    if shape is None or body is None:
        return None
    try:
        parsed: Final = shape.parse(body)
    except ValidationError:
        return None
    logging_obj.call_type = (
        shape.call_type.value
    )  # rebind-ok: routes cost calculation to the relayed shape's pricing path
    return parsed


class PassthroughStreamCollector(Protocol):
    """Consumes relayed stream bytes as they arrive and builds the response logged for spend tracking."""

    def add(self, chunk: bytes) -> None: ...

    def build_logged_response(self, litellm_logging_obj: LiteLLMLoggingObj) -> LoggedRelayResponse | None: ...


class RawBytesStreamCollector:
    def __init__(
        self, provider_config: BasePassthroughConfig, model: str, custom_llm_provider: str, endpoint: str
    ) -> None:
        self._provider_config = provider_config
        self._model = model
        self._custom_llm_provider = custom_llm_provider
        self._endpoint = endpoint
        self._raw_bytes: list[bytes] = []  # mutable-ok: instance buffer for streaming chunks

    def add(self, chunk: bytes) -> None:
        self._raw_bytes.append(chunk)

    def build_logged_response(self, litellm_logging_obj: LiteLLMLoggingObj) -> LoggedRelayResponse | None:
        all_chunks: Final = self._provider_config._convert_raw_bytes_to_str_lines(self._raw_bytes)
        return self._provider_config.handle_logging_collected_chunks(
            all_chunks=all_chunks,
            litellm_logging_obj=litellm_logging_obj,
            model=self._model,
            custom_llm_provider=self._custom_llm_provider,
            endpoint=self._endpoint,
        )


class BasePassthroughConfig(BaseLLMModelInfo):
    @abstractmethod
    def is_streaming_request(self, endpoint: str, request_data: dict) -> bool:
        """
        Check if the request is a streaming request
        """

    def format_url(
        self,
        endpoint: str,
        base_target_url: str,
        request_query_params: Mapping[str, object] | None,
    ) -> URL:
        """
        Helper function to add query params to the url
        Args:
            endpoint: str - the endpoint to add to the url
            base_target_url: str - the base url to add the endpoint to
            request_query_params: Optional[dict] - the query params to add to the url
        Returns:
            httpx.URL - the formatted url
        """
        from urllib.parse import urlencode

        import httpx

        base: Final = base_target_url.rstrip("/")
        endpoint = endpoint.lstrip("/")
        full_url: Final = f"{base}/{endpoint}"

        url = httpx.URL(full_url)

        if request_query_params:
            url = url.copy_with(query=urlencode(request_query_params).encode("ascii"))

        return url

    @abstractmethod
    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        endpoint: str,
        request_query_params: dict | None,
        litellm_params: dict,
    ) -> tuple[URL, str]:
        """
        Get the complete url for the request
        Returns:
            - complete_url: URL - the complete url for the request
            - base_target_url: str - the base url to add the endpoint to. Useful for auth headers.
        """

    def sign_request(
        self,
        headers: dict,
        litellm_params: dict,
        request_data: dict | None,
        api_base: str,
        model: str | None = None,
    ) -> tuple[dict, bytes | None]:
        """
        Some providers like Bedrock require signing the request. The sign request funtion needs access to `request_data` and `complete_url`
        Args:
            headers: dict
            optional_params: dict
            request_data: dict - the request body being sent in http request
            api_base: str - the complete url being sent in http request
        Returns:
            dict - the signed headers

        Update the headers with the signed headers in this function. The return values will be sent as headers in the http request.
        """
        return headers, None

    def get_error_class(self, error_message: str, status_code: int, headers: dict | Headers) -> BaseLLMException:
        from litellm.llms.base_llm.chat.transformation import BaseLLMException

        return BaseLLMException(status_code=status_code, message=error_message, headers=headers)

    def logging_non_streaming_response(
        self,
        model: str,
        custom_llm_provider: str,
        httpx_response: Response,
        request_data: dict,
        logging_obj: LiteLLMLoggingObj,
        endpoint: str,
    ) -> LoggedRelayResponse | OCRResponse | StandardPassThroughResponseObject | None:
        pass

    def handle_logging_collected_chunks(
        self,
        all_chunks: list[str],
        litellm_logging_obj: LiteLLMLoggingObj,
        model: str,
        custom_llm_provider: str,
        endpoint: str,
    ) -> LoggedRelayResponse | None:
        return None

    def create_stream_collector(
        self, model: str, custom_llm_provider: str, endpoint: str
    ) -> PassthroughStreamCollector:
        return RawBytesStreamCollector(
            provider_config=self, model=model, custom_llm_provider=custom_llm_provider, endpoint=endpoint
        )

    def _convert_raw_bytes_to_str_lines(self, raw_bytes: list[bytes]) -> list[str]:
        """
        Converts a list of raw bytes into a list of string lines, similar to aiter_lines()

        Args:
            raw_bytes: List of bytes chunks from aiter.bytes()

        Returns:
            List of string lines, with each line being a complete data: {} chunk
        """
        # Combine all bytes and decode to string
        combined_str: Final = b"".join(raw_bytes).decode("utf-8")

        # Split by newlines and filter out empty lines
        lines: Final = [line.strip() for line in combined_str.split("\n") if line.strip()]

        return lines
