from collections.abc import Mapping
from typing import Union

import httpx

from litellm.llms.cloudflare.chat.transformation import CloudflareChatConfig, CloudflareError
from litellm.llms.vercel_ai_gateway.embedding.transformation import VercelAIGatewayEmbeddingConfig


class CloudflareEmbeddingConfig(VercelAIGatewayEmbeddingConfig):
    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[object, object],
        litellm_params: Mapping[object, object],
        stream: bool | None = None,
    ) -> str:
        resolved_base = CloudflareChatConfig.resolve_api_base(api_base).rstrip("/")
        if resolved_base.endswith("/embeddings"):
            return resolved_base
        return f"{resolved_base}/embeddings"

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Union[Mapping[object, object], httpx.Headers],
    ) -> CloudflareError:
        return CloudflareError(status_code=status_code, message=error_message)
