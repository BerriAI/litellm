from typing import TYPE_CHECKING, Any, Optional, Union

from pydantic import ConfigDict
from semantic_router.encoders import DenseEncoder
from semantic_router.encoders.base import AsymmetricDenseMixin

import litellm

if TYPE_CHECKING:
    from litellm.router import Router
else:
    Router = Any


def litellm_to_list(embeds: litellm.EmbeddingResponse) -> list[list[float]]:
    """Convert a LiteLLM embedding response to a list of embeddings.

    :param embeds: The LiteLLM embedding response.
    :return: A list of embeddings.
    """
    if (
        not embeds
        or not isinstance(embeds, litellm.EmbeddingResponse)
        or not embeds.data
    ):
        raise ValueError("No embeddings found in LiteLLM embedding response.")
    return [x["embedding"] for x in embeds.data]


class CustomDenseEncoder(DenseEncoder):
    model_config = ConfigDict(extra='allow')

    def __init__(self, litellm_router_instance: Optional["Router"] = None, **kwargs):
        # Extract litellm_router_instance from kwargs if passed there
        if 'litellm_router_instance' in kwargs:
            litellm_router_instance = kwargs.pop('litellm_router_instance')
        
        super().__init__(**kwargs)
        self.litellm_router_instance = litellm_router_instance


class LiteLLMRouterEncoder(CustomDenseEncoder, AsymmetricDenseMixin):
    """LiteLLM encoder class for generating embeddings using LiteLLM.

    The LiteLLMRouterEncoder class is a subclass of DenseEncoder and utilizes the LiteLLM Router SDK
    to generate embeddings for given documents. It supports all encoders supported by LiteLLM
    and supports customization of the score threshold for filtering or processing the embeddings.
    """

    type: str = "internal_litellm_router"

    def __init__(
        self,
        litellm_router_instance: "Router",
        model_name: str,
        score_threshold: Union[float, None] = None,
    ):
        """Initialize the LiteLLMEncoder.

        :param litellm_router_instance: The instance of the LiteLLM Router.
        :type litellm_router_instance: Router
        :param model_name: The name of the embedding model to use. Must use LiteLLM naming
            convention (e.g. "openai/text-embedding-3-small" or "mistral/mistral-embed").
        :type model_name: str
        :param score_threshold: The score threshold for the embeddings.
        :type score_threshold: float
        """
        super().__init__(
            name=model_name,
            score_threshold=score_threshold if score_threshold is not None else 0.3,
        )
        self.model_name = model_name
        self.litellm_router_instance = litellm_router_instance

    def __call__(self, docs: list[Any], **kwargs) -> list[list[float]]:
        """Encode a list of text documents into embeddings using LiteLLM.

        :param docs: List of text documents to encode.
        :return: List of embeddings for each document."""
        return self.encode_queries(docs, **kwargs)

    async def acall(self, docs: list[Any], **kwargs) -> list[list[float]]:
        """Encode a list of documents into embeddings using LiteLLM asynchronously.

        :param docs: List of documents to encode.
        :return: List of embeddings for each document."""
        return await self.aencode_queries(docs, **kwargs)

    def encode_queries(self, docs: list[str], **kwargs) -> list[list[float]]:
        if self.litellm_router_instance is None:
            raise ValueError("litellm_router_instance is not set")
        try:
            embeds = self.litellm_router_instance.embedding(
                input=docs, 
                model=self.model_name, 
                **kwargs
            )
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(
                f"{self.type.capitalize()} API call failed. Error: {e}"
            ) from e

    def encode_documents(self, docs: list[str], **kwargs) -> list[list[float]]:
        if self.litellm_router_instance is None:
            raise ValueError("litellm_router_instance is not set")
        try:
            embeds = self.litellm_router_instance.embedding(
                input=docs, 
                model=self.model_name, 
                **kwargs
            )
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(
                f"{self.type.capitalize()} API call failed. Error: {e}"
            ) from e

    async def aencode_queries(self, docs: list[str], **kwargs) -> list[list[float]]:
        if self.litellm_router_instance is None:
            raise ValueError("litellm_router_instance is not set")
        try:
            embeds = await self.litellm_router_instance.aembedding(
                input=docs, 
                model=self.model_name,
                **kwargs
            )
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(
                f"{self.type.capitalize()} API call failed. Error: {e}"
            ) from e

    async def aencode_documents(self, docs: list[str], **kwargs) -> list[list[float]]:
        if self.litellm_router_instance is None:
            raise ValueError("litellm_router_instance is not set")
        try:
            embeds = await self.litellm_router_instance.aembedding(
                input=docs, 
                model=self.model_name,
                **kwargs
            )
            return litellm_to_list(embeds)
        except Exception as e:
            raise ValueError(
                f"{self.type.capitalize()} API call failed. Error: {e}"
            ) from e
