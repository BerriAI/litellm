"""
Base Search transformation configuration.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union
from urllib.parse import urlsplit

import httpx
from pydantic import PrivateAttr

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.base import LiteLLMPydanticObjectBase

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


def _search_host(url: str) -> str:
    return urlsplit(url).netloc.lower()


def _is_trusted_search_api_base(
    caller_api_base: str,
    default_api_base: str | None,
    base_env_var: str | None,
) -> bool:
    candidate = _search_host(caller_api_base)
    if not candidate:
        return False
    trusted = {
        _search_host(base)
        for base in (
            default_api_base,
            get_secret_str(base_env_var) if base_env_var else None,
        )
        if base
    }
    return candidate in trusted


class SearchResult(LiteLLMPydanticObjectBase):
    """Single search result."""

    title: str
    url: str
    snippet: str
    date: Optional[str] = None
    last_updated: Optional[str] = None

    model_config = {"extra": "allow"}


class SearchResponse(LiteLLMPydanticObjectBase):
    """
    Standard Search response format.
    Standardized to Perplexity Search format - other providers should transform to this format.
    """

    results: List[SearchResult]
    object: str = "search"

    model_config = {"extra": "allow"}

    # Define private attributes using PrivateAttr
    _hidden_params: dict = PrivateAttr(default_factory=dict)


class BaseSearchConfig:
    """
    Base configuration for Search transformations.
    Handles provider-agnostic Search operations.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def ui_friendly_name() -> str:
        """
        UI-friendly name for the search provider.
        Override in provider-specific implementations.
        """
        return "Unknown Search Provider"

    def get_http_method(self) -> Literal["GET", "POST"]:
        """
        Get HTTP method for search requests.
        Override in provider-specific implementations if needed.

        Returns:
            HTTP method ('GET' or 'POST'). Default is 'POST'.
        """
        return "POST"

    @staticmethod
    def get_supported_perplexity_optional_params() -> set:
        """
        Get the set of Perplexity unified search parameters.
        These are the standard parameters that providers should transform from.

        Returns:
            Set of parameter names that are part of the unified spec
        """
        return {
            "max_results",
            "search_domain_filter",
            "country",
            "max_tokens_per_page",
        }

    def _assert_trusted_api_base_for_server_credential(
        self,
        caller_api_base: str | None,
        default_api_base: str | None,
        base_env_var: str | None,
        credential_name: str,
    ) -> None:
        """
        Block sending a server-managed credential to a caller-chosen host.

        A caller-supplied api_base is honored when constructing the request URL, so
        falling back to a server-configured secret while the caller controls the host
        leaks that secret. The provider default and the operator's own api_base
        override are the only trusted destinations for a server-managed credential.
        """
        if not caller_api_base:
            return
        if _is_trusted_search_api_base(caller_api_base, default_api_base, base_env_var):
            return
        raise ValueError(
            f"Refusing to send the server-configured {credential_name} to the "
            f"caller-supplied api_base '{caller_api_base}'. Pass an explicit api_key "
            f"when overriding api_base for this search provider."
        )

    def resolve_server_api_key(
        self,
        *,
        caller_api_key: str | None,
        caller_api_base: str | None,
        key_env_vars: tuple[str, ...],
        base_env_var: str | None,
        default_api_base: str | None,
    ) -> str | None:
        """
        Resolve a single-secret search API key, falling back to a server-managed
        secret only when the request targets a trusted host.

        Returns the caller's key when provided, otherwise the first set
        server-managed secret (or None when none is set, for keyless providers).
        """
        if caller_api_key:
            return caller_api_key
        server_key = next(
            (key for key in (get_secret_str(var) for var in key_env_vars) if key),
            None,
        )
        if server_key is None:
            return None
        self._assert_trusted_api_base_for_server_credential(
            caller_api_base, default_api_base, base_env_var, key_env_vars[0]
        )
        return server_key

    def validate_environment(
        self,
        headers: Dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        """
        Validate environment and return headers.
        Override in provider-specific implementations.
        """
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        optional_params: dict,
        data: Optional[Union[Dict, List[Dict]]] = None,
        **kwargs,
    ) -> str:
        """
        Get complete URL for Search endpoint.

        Args:
            api_base: Base URL for the API
            optional_params: Optional parameters for the request
            data: Transformed request body from transform_search_request().
                  Some providers (e.g., Google PSE) use GET requests and need
                  the request body to construct query parameters in the URL.
                  Can be a dict or list of dicts depending on provider.
            **kwargs: Additional keyword arguments

        Returns:
            Complete URL for the search endpoint

        Note:
            Override in provider-specific implementations.
        """
        raise NotImplementedError("get_complete_url must be implemented by provider")

    def transform_search_request(
        self,
        query: Union[str, List[str]],
        optional_params: dict,
        **kwargs,
    ) -> Union[Dict, List[Dict]]:
        """
        Transform Search request to provider-specific format.
        Override in provider-specific implementations.

        Args:
            query: Search query (string or list of strings)
            optional_params: Optional parameters for the request

        Returns:
            Dict with request data
        """
        raise NotImplementedError(
            "transform_search_request must be implemented by provider"
        )

    def transform_search_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        **kwargs,
    ) -> SearchResponse:
        """
        Transform provider-specific Search response to standard format.
        Override in provider-specific implementations.
        """
        raise NotImplementedError(
            "transform_search_response must be implemented by provider"
        )

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict,
    ) -> Exception:
        """Get appropriate error class for the provider."""
        return BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
