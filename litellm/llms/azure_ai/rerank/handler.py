from typing import Any, Dict, List, Optional, Union

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.cohere.rerank import CohereRerank
from litellm.rerank_api.types import RerankResponse


class AzureAIRerank(CohereRerank):
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

        return super().rerank(
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
