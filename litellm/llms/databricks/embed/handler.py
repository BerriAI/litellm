"""
Calling logic for Databricks embeddings
"""

import os
from collections.abc import Mapping
from typing import Final

from litellm.utils import EmbeddingResponse

from ...openai_like.embedding.handler import OpenAILikeEmbeddingHandler
from ..common_utils import DatabricksBase


class DatabricksEmbeddingHandler(OpenAILikeEmbeddingHandler, DatabricksBase):
    def embedding(
        self,
        model: str,
        input: list,
        timeout: float,
        logging_obj,
        api_key: str | None,
        api_base: str | None,
        optional_params: dict,
        model_response: EmbeddingResponse | None = None,
        client=None,
        aembedding=None,
        custom_endpoint: bool | None = None,
        headers: dict | None = None,
        drop_params: bool = False,
        client_id: str | None = None,
        client_secret: str | None = None,
        litellm_params: dict | None = None,
        **kwargs,
    ) -> EmbeddingResponse:
        # Check for custom user agent in optional_params or environment
        # This allows partners building on LiteLLM to set their own telemetry
        # Use pop() to remove these keys so they don't get sent to the API
        optional_params = optional_params or {}
        custom_user_agent: Final = (
            optional_params.pop("user_agent", None)
            or optional_params.pop("databricks_user_agent", None)
            or os.getenv("LITELLM_USER_AGENT")
            or os.getenv("DATABRICKS_USER_AGENT")
        )

        for secret_param in ("client_id", "client_secret", "databricks_client_id", "databricks_client_secret"):
            val = optional_params.get(secret_param) or kwargs.get(secret_param)
            if isinstance(val, str) and (val.startswith("os.environ/") or val.startswith("oidc/")):
                from ..common_utils import DatabricksException

                raise DatabricksException(
                    status_code=400,
                    message=f"Request-supplied {secret_param} cannot be an os.environ/ or oidc/ secret reference",
                )

        resolved_client_id: Final = (
            client_id
            or (litellm_params.get("client_id") if isinstance(litellm_params, Mapping) else None)
            or (litellm_params.get("databricks_client_id") if isinstance(litellm_params, Mapping) else None)
            or optional_params.pop("client_id", None)
            or optional_params.pop("databricks_client_id", None)
            or kwargs.pop("client_id", None)
            or kwargs.pop("databricks_client_id", None)
        )
        resolved_client_secret: Final = (
            client_secret
            or (litellm_params.get("client_secret") if isinstance(litellm_params, Mapping) else None)
            or (litellm_params.get("databricks_client_secret") if isinstance(litellm_params, Mapping) else None)
            or optional_params.pop("client_secret", None)
            or optional_params.pop("databricks_client_secret", None)
            or kwargs.pop("client_secret", None)
            or kwargs.pop("databricks_client_secret", None)
        )

        api_base, headers = self.databricks_validate_environment(
            api_base=api_base,
            api_key=api_key,
            endpoint_type="embeddings",
            custom_endpoint=custom_endpoint,
            headers=headers,
            custom_user_agent=custom_user_agent,
            client_id=resolved_client_id,
            client_secret=resolved_client_secret,
            litellm_params=litellm_params,
        )
        return super().embedding(
            model=model,
            input=input,
            timeout=timeout,
            logging_obj=logging_obj,
            api_key=api_key,
            api_base=api_base,
            optional_params=optional_params,
            model_response=model_response,
            client=client,
            aembedding=aembedding,
            custom_endpoint=True,
            headers=headers,
            drop_params=drop_params,
        )
