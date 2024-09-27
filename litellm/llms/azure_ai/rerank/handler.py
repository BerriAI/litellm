from typing import Any, Dict, List, Optional, Union

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.cohere.rerank import CohereRerank
from litellm.rerank_api.types import RerankResponse


class AzureAIRerank(CohereRerank):

    def get_base_model(self, azure_model_group: Optional[str]) -> Optional[str]:
        if azure_model_group is None:
            return None
        if azure_model_group == "offer-cohere-rerank-mul-paygo":
            return "azure_ai/cohere-rerank-v3-multilingual"
        if azure_model_group == "offer-cohere-rerank-eng-paygo":
            return "azure_ai/cohere-rerank-v3-english"
        return azure_model_group

    async def async_azure_rerank(
        self,
        model: str,
        api_key: str,
        api_base: str,
        query: str,
        documents: List[Union[str, Dict[str, Any]]],
        headers: Optional[dict],
        litellm_logging_obj: LiteLLMLoggingObj,
        top_n: Optional[int] = None,
        rank_fields: Optional[List[str]] = None,
        return_documents: Optional[bool] = True,
        max_chunks_per_doc: Optional[int] = None,
    ):
        returned_response: RerankResponse = await super().rerank(  # type: ignore
            model=model,
            api_key=api_key,
            api_base=api_base,
            query=query,
            documents=documents,
            top_n=top_n,
            rank_fields=rank_fields,
            return_documents=return_documents,
            max_chunks_per_doc=max_chunks_per_doc,
            _is_async=True,
            headers=headers,
            litellm_logging_obj=litellm_logging_obj,
        )

        # get base model
        additional_headers = (
            returned_response._hidden_params.get("additional_headers") or {}
        )

        base_model = self.get_base_model(
            additional_headers.get("llm_provider-azureml-model-group")
        )
        returned_response._hidden_params["model"] = base_model

        return returned_response

    def rerank(
        self,
        model: str,
        api_key: str,
        api_base: str,
        query: str,
        documents: List[Union[str, Dict[str, Any]]],
        headers: Optional[dict],
        litellm_logging_obj: LiteLLMLoggingObj,
        top_n: Optional[int] = None,
        rank_fields: Optional[List[str]] = None,
        return_documents: Optional[bool] = True,
        max_chunks_per_doc: Optional[int] = None,
        _is_async: Optional[bool] = False,
    ) -> RerankResponse:

        if headers is None:
            headers = {"Authorization": "Bearer {}".format(api_key)}
        else:
            headers = {**headers, "Authorization": "Bearer {}".format(api_key)}

        # Assuming api_base is a string representing the base URL
        api_base_url = httpx.URL(api_base)

        # Replace the path with '/v1/rerank' if it doesn't already end with it
        if not api_base_url.path.endswith("/v1/rerank"):
            api_base = str(api_base_url.copy_with(path="/v1/rerank"))

        if _is_async:
            return self.async_azure_rerank(  # type: ignore
                model=model,
                api_key=api_key,
                api_base=api_base,
                query=query,
                documents=documents,
                top_n=top_n,
                rank_fields=rank_fields,
                return_documents=return_documents,
                max_chunks_per_doc=max_chunks_per_doc,
                headers=headers,
                litellm_logging_obj=litellm_logging_obj,
            )
        else:
            returned_response = super().rerank(
                model=model,
                api_key=api_key,
                api_base=api_base,
                query=query,
                documents=documents,
                top_n=top_n,
                rank_fields=rank_fields,
                return_documents=return_documents,
                max_chunks_per_doc=max_chunks_per_doc,
                _is_async=_is_async,
                headers=headers,
                litellm_logging_obj=litellm_logging_obj,
            )

            # get base model
            base_model = self.get_base_model(
                returned_response._hidden_params.get("llm_provider-azureml-model-group")
            )
            returned_response._hidden_params["model"] = base_model
            return returned_response
