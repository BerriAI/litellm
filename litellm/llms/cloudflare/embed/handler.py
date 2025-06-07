"""
Cloudflare Workers AI embedding handler
"""

from typing import Optional, Union, List
import httpx
from litellm.types.utils import EmbeddingResponse
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.cloudflare.embed.transformation import CloudflareEmbeddingConfig
from litellm.secret_managers.main import get_secret_str


async def aembedding(
    model: str,
    input: Union[str, List[str]],
    api_base: Optional[str],
    api_key: Optional[str],
    logging_obj: Logging,
    model_response: EmbeddingResponse,
    timeout: Optional[float] = None,
    optional_params: Optional[dict] = None,
    client: Optional[httpx.AsyncClient] = None,
    aembedding: Optional[bool] = True,
    **kwargs,
) -> EmbeddingResponse:
    """
    Async embedding function for Cloudflare Workers AI
    """
    config = CloudflareEmbeddingConfig()

    # Ensure we have the required parameters
    if api_key is None:
        api_key = get_secret_str("CLOUDFLARE_API_KEY")

    if api_key is None:
        raise ValueError("Missing Cloudflare API Key")

    # Get complete URL
    url = config.get_complete_url(
        api_base=api_base,
        api_key=api_key,
        model=model,
        optional_params=optional_params or {},
        litellm_params=kwargs.get("litellm_params", {}),
    )

    # Validate environment
    config.validate_environment(
        headers={},
        model=model,
        messages=[],  # Not used for embeddings
        optional_params=optional_params or {},
        litellm_params=kwargs.get("litellm_params", {}),
        api_key=api_key,
    )

    # Transform request
    data = config.transform_embedding_request(
        model=model,
        input=input,
        optional_params=optional_params or {},
        headers={},
    )

    # Prepare headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Make request
    if client is None:
        client = httpx.AsyncClient()
        close_client = True
    else:
        close_client = False

    try:
        response = await client.post(
            url=url,
            json=data,
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()

        # Transform response
        return config.transform_embedding_response(
            model=model,
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            api_key=api_key,
            request_data={"input": input},
            optional_params=optional_params or {},
            litellm_params=kwargs.get("litellm_params", {}),
        )
    finally:
        if close_client:
            await client.aclose()


def embedding(
    model: str,
    input: Union[str, List[str]],
    api_base: Optional[str],
    api_key: Optional[str],
    logging_obj: Logging,
    model_response: EmbeddingResponse,
    timeout: Optional[float] = None,
    optional_params: Optional[dict] = None,
    client: Optional[httpx.Client] = None,
    aembedding: Optional[bool] = False,
    **kwargs,
) -> EmbeddingResponse:
    """
    Sync embedding function for Cloudflare Workers AI
    """
    config = CloudflareEmbeddingConfig()

    # Ensure we have the required parameters
    if api_key is None:
        api_key = get_secret_str("CLOUDFLARE_API_KEY")

    if api_key is None:
        raise ValueError("Missing Cloudflare API Key")

    # Get complete URL
    url = config.get_complete_url(
        api_base=api_base,
        api_key=api_key,
        model=model,
        optional_params=optional_params or {},
        litellm_params=kwargs.get("litellm_params", {}),
    )

    # Validate environment
    config.validate_environment(
        headers={},
        model=model,
        messages=[],  # Not used for embeddings
        optional_params=optional_params or {},
        litellm_params=kwargs.get("litellm_params", {}),
        api_key=api_key,
    )

    # Transform request
    data = config.transform_embedding_request(
        model=model,
        input=input,
        optional_params=optional_params or {},
        headers={},
    )

    # Prepare headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Make request
    if client is None:
        client = httpx.Client()
        close_client = True
    else:
        close_client = False

    try:
        response = client.post(
            url=url,
            json=data,
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()

        # Transform response
        return config.transform_embedding_response(
            model=model,
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            api_key=api_key,
            request_data={"input": input},
            optional_params=optional_params or {},
            litellm_params=kwargs.get("litellm_params", {}),
        )
    finally:
        if close_client:
            client.close()
