from abc import abstractmethod
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from litellm.passthrough.utils import collect_lines_from_chunk

from ..base_utils import BaseLLMModelInfo

if TYPE_CHECKING:
    from httpx import URL, Headers, Response

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.utils import CostResponseTypes

    from ..chat.transformation import BaseLLMException


class PassthroughStreamingChunkProcessor:
    """
    Stateful processor that converts a stream of raw bytes into parsed string
    chunks incrementally. Lets streaming pass-through avoid buffering the full
    raw response in memory for logging — providers parse on the fly and only
    retain the small parsed messages.
    """

    def process(self, chunk: bytes) -> List[str]:
        """Feed a chunk and return any newly-available parsed lines/messages."""
        raise NotImplementedError

    def finalize(self) -> List[str]:
        """Flush any remaining state after the stream ends. Default: nothing."""
        return []


class _LineBufferedChunkProcessor(PassthroughStreamingChunkProcessor):
    """Default text/SSE processor: buffers partial lines, emits complete \\n-delimited lines."""

    def __init__(self) -> None:
        self._buffer = ""

    def process(self, chunk: bytes) -> List[str]:
        lines, self._buffer = collect_lines_from_chunk(self._buffer, chunk)
        return lines

    def finalize(self) -> List[str]:
        if self._buffer.strip():
            return [self._buffer.strip()]
        return []


class BasePassthroughConfig(BaseLLMModelInfo):
    @abstractmethod
    def is_streaming_request(self, endpoint: str, request_data: dict) -> bool:
        """
        Check if the request is a streaming request
        """
        pass

    def format_url(
        self,
        endpoint: str,
        base_target_url: str,
        request_query_params: Optional[dict],
    ) -> "URL":
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

        base = base_target_url.rstrip("/")
        endpoint = endpoint.lstrip("/")
        full_url = f"{base}/{endpoint}"

        url = httpx.URL(full_url)

        if request_query_params:
            url = url.copy_with(query=urlencode(request_query_params).encode("ascii"))

        return url

    @abstractmethod
    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        endpoint: str,
        request_query_params: Optional[dict],
        litellm_params: dict,
    ) -> Tuple["URL", str]:
        """
        Get the complete url for the request
        Returns:
            - complete_url: URL - the complete url for the request
            - base_target_url: str - the base url to add the endpoint to. Useful for auth headers.
        """
        pass

    def sign_request(
        self,
        headers: dict,
        litellm_params: dict,
        request_data: Optional[dict],
        api_base: str,
        model: Optional[str] = None,
    ) -> Tuple[dict, Optional[bytes]]:
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

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, "Headers"]
    ) -> "BaseLLMException":
        from litellm.llms.base_llm.chat.transformation import BaseLLMException

        return BaseLLMException(
            status_code=status_code, message=error_message, headers=headers
        )

    def logging_non_streaming_response(
        self,
        model: str,
        custom_llm_provider: str,
        httpx_response: "Response",
        request_data: dict,
        logging_obj: "LiteLLMLoggingObj",
        endpoint: str,
    ) -> Optional["CostResponseTypes"]:
        pass

    def handle_logging_collected_chunks(
        self,
        all_chunks: List[str],
        litellm_logging_obj: "LiteLLMLoggingObj",
        model: str,
        custom_llm_provider: str,
        endpoint: str,
    ) -> Optional["CostResponseTypes"]:
        return None

    def create_streaming_chunk_processor(self) -> PassthroughStreamingChunkProcessor:
        """
        Returns a stateful processor that incrementally converts raw response
        bytes into parsed string chunks for logging. Override for providers
        whose streaming wire format isn't newline-delimited UTF-8 (e.g. Bedrock
        AWS event-stream). Default is suitable for SSE/text providers.
        """
        return _LineBufferedChunkProcessor()
