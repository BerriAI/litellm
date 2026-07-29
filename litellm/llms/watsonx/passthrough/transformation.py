from typing import TYPE_CHECKING, List, Optional, Tuple

from litellm.llms.base_llm.passthrough.transformation import BasePassthroughConfig
from litellm.llms.watsonx.common_utils import IBMWatsonXMixin

if TYPE_CHECKING:
    from httpx import URL


class WatsonxPassthroughConfig(IBMWatsonXMixin, BasePassthroughConfig):
    """
    Watsonx-specific passthrough configuration.
    """

    def is_streaming_request(self, endpoint: str, request_data: dict) -> bool:
        """Check if request should be streamed"""
        return request_data.get("stream", False)

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
        Construct complete Watsonx URL with version parameter.

        This ensures the version parameter is ALWAYS included in the URL,
        solving the query parameter issue.
        """
        base_target_url = str(self.get_api_base(api_base))

        # Use the format_url helper to construct URL with query params
        complete_url = self.format_url(
            endpoint=endpoint,
            base_target_url=base_target_url,
            request_query_params=request_query_params,
        )

        return (complete_url, base_target_url)

    @staticmethod
    def get_api_base(
        api_base: Optional[str] = None,
    ) -> Optional[str]:
        return api_base or IBMWatsonXMixin()._get_base_url(api_base=api_base)

    @staticmethod
    def get_api_key(
        api_key: Optional[str] = None,
    ) -> Optional[str]:
        return (
            api_key
            or IBMWatsonXMixin.get_watsonx_credentials(optional_params=dict(), api_base=None, api_key=api_key)[
                "api_key"
            ]
        )

    @staticmethod
    def get_base_model(model: str) -> Optional[str]:
        return model

    def get_models(self, api_key: Optional[str] = None, api_base: Optional[str] = None) -> List[str]:
        return super().get_models(api_key, api_base)
