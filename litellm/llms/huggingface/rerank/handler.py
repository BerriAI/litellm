## Uses the huggingface text embedding inference API
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    cast,
    get_args,
)

import httpx
import uuid
import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj # Add logging later
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
)
from litellm.llms.huggingface.chat.transformation import (
    HuggingfaceChatConfig as HuggingfaceConfig,
)

from ...base import BaseLLM
from litellm.types.rerank import RerankRequest, RerankResponse, RerankResponseResult, RerankResponseDocument


class HuggingfaceRerank(BaseLLM):
    _client_session: Optional[httpx.Client] = None
    _aclient_session: Optional[httpx.AsyncClient] = None

    def __init__(self) -> None:
        super().__init__()

    def rerank(
        self,
        model: str,
        api_key: str,
        api_base: Optional[str],
        query: str,
        documents: List[Union[str, Dict[str, Any]]],
        client: Optional[HTTPHandler] = None, 
    ):
        
        request_data = RerankRequest(
            model=model,
            query=query,
            documents=documents,
        )

        request_data_dict = request_data.model_dump(exclude_none=True, by_alias=True)
        
        if client is None:
            client = _get_httpx_client()

        response = client.post( 
            f"{api_base}/rerank",
            headers={
                "accept": "application/json",
                "content-type": "application/json",
                "authorization": f"Bearer {api_key}",
            },
            json=request_data_dict,
        )

        if response.status_code != 200: # takes care of times when response is an object
            raise Exception(response.error)

        _json_response = response.json()
        results = []
        # Used TEI Reference to build this RerankResponse
        #  https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
        for res in _json_response:
            result = RerankResponseResult(
                index=res["index"],
                relevance_score=res["score"],
                document=RerankResponseDocument(text=res["text"])
            )
            results.append(result)

        return RerankResponse(
            id=str(uuid.uuid4()),
            results=results
        )