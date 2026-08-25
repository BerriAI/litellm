from collections.abc import Mapping
from typing import Final, Literal

from litellm.llms.bedrock.passthrough.transformation import BedrockPassthroughConfig
from litellm.llms.bedrock_mantle.common_utils import (
    MANTLE_HOST_RE,
    resolve_mantle_bearer_token,
    resolve_mantle_region,
)


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
