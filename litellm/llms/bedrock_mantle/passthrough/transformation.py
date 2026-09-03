from collections.abc import Mapping
from typing import TYPE_CHECKING, Final, Literal, Optional

from httpx import Response

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.bedrock.passthrough.transformation import BedrockPassthroughConfig
from litellm.llms.bedrock_mantle.common_utils import (
    MANTLE_HOST_RE,
    resolve_mantle_bearer_token,
    resolve_mantle_region,
)
from litellm.types.utils import LlmProviders

if TYPE_CHECKING:
    from litellm.types.utils import CostResponseTypes


class BedrockMantlePassthroughConfig(BedrockPassthroughConfig):
    """Native Bedrock runtime passthrough (InvokeModel, Converse) for deployments declared as bedrock_mantle.

    The Mantle host only serves the OpenAI-compatible surface, so a Mantle api_base lends its region and the
    request itself goes to bedrock-runtime, signed with the deployment's Bearer token or SigV4 credentials.
    """

    def _get_aws_region_name(
        self,
        optional_params: Mapping[str, object],
        model: str | None = None,
        model_id: str | None = None,
    ) -> str:
        return resolve_mantle_region(optional_params)

    def get_runtime_endpoint(
        self,
        api_base: str | None,
        aws_bedrock_runtime_endpoint: str | None,
        aws_region_name: str,
        endpoint_type: Literal["runtime", "agent", "agentcore"] | None = "runtime",
    ) -> tuple[str, str]:
        is_mantle_host: Final = api_base is not None and MANTLE_HOST_RE.match(api_base.rstrip("/")) is not None
        return super().get_runtime_endpoint(
            api_base=None if is_mantle_host else api_base,
            aws_bedrock_runtime_endpoint=aws_bedrock_runtime_endpoint,
            aws_region_name=aws_region_name,
            endpoint_type=endpoint_type,
        )

    def get_bedrock_bearer_token(self, litellm_params: Mapping[str, object]) -> str | None:
        api_key: Final = litellm_params.get("api_key")
        return resolve_mantle_bearer_token(api_key if isinstance(api_key, str) else None)

    def logging_non_streaming_response(
        self,
        model: str,
        custom_llm_provider: str,
        httpx_response: Response,
        request_data: dict,  # mutable-ok: mirrors the inherited BedrockPassthroughConfig signature
        logging_obj: Logging,
        endpoint: str,
    ) -> Optional["CostResponseTypes"]:
        is_converse: Final = "invoke" not in endpoint and "converse" in endpoint
        shape_provider: Final = LlmProviders.BEDROCK.value if is_converse else custom_llm_provider
        return super().logging_non_streaming_response(
            model=model,
            custom_llm_provider=shape_provider,
            httpx_response=httpx_response,
            request_data=request_data,
            logging_obj=logging_obj,
            endpoint=endpoint,
        )
