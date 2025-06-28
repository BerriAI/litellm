from abc import abstractmethod
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from ..base_utils import BaseLLMModelInfo

if TYPE_CHECKING:
    from httpx import URL, Headers, Response

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.utils import CostResponseTypes

    from ..chat.transformation import BaseLLMException


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
            request_query_params: dict - the query params to add to the url
        Returns:
            str - the formatted url
        """
        from urllib.parse import urlencode

        import httpx

        encoded_endpoint = httpx.URL(endpoint).path

        # Ensure endpoint starts with '/' for proper URL construction
        if not encoded_endpoint.startswith("/"):
            encoded_endpoint = "/" + encoded_endpoint

        # Construct the full target URL using httpx
        base_url = httpx.URL(base_target_url)
        updated_url = base_url.copy_with(path=encoded_endpoint)

        if request_query_params:
            # Create a new URL with the merged query params
            updated_url = updated_url.copy_with(
                query=urlencode(request_query_params).encode("ascii")
            )
        return updated_url

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

    def _convert_raw_bytes_to_str_lines(self, raw_bytes: List[bytes]) -> List[str]:
        """
        Converts a list of raw bytes into a list of string lines, similar to aiter_lines()

        Args:
            raw_bytes: List of bytes chunks from aiter.bytes()

        Returns:
            List of string lines, with each line being a complete data: {} chunk
        """
        # Combine all bytes and decode to string
        combined_str = b"".join(raw_bytes).decode("utf-8")

        # Split by newlines and filter out empty lines
        lines = [line.strip() for line in combined_str.split("\n") if line.strip()]

        return lines
