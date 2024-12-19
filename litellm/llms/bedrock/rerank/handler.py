import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, cast

import httpx

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LitellmLogging
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.types.llms.bedrock import BedrockPreparedRequest
from litellm.types.rerank import RerankRequest
from litellm.types.utils import RerankResponse

from ..base_aws_llm import BaseAWSLLM
from ..common_utils import BedrockError
from .transformation import BedrockRerankConfig

if TYPE_CHECKING:
    from botocore.awsrequest import AWSPreparedRequest
else:
    AWSPreparedRequest = Any


class BedrockRerankHandler(BaseAWSLLM):
    async def arerank(
        self,
        prepared_request: BedrockPreparedRequest,
    ):
        client = get_async_httpx_client(llm_provider=litellm.LlmProviders.BEDROCK)
        try:
            response = await client.post(url=prepared_request["endpoint_url"], headers=prepared_request["prepped"].headers, data=prepared_request["body"])  # type: ignore
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            error_code = err.response.status_code
            raise BedrockError(status_code=error_code, message=err.response.text)
        except httpx.TimeoutException:
            raise BedrockError(status_code=408, message="Timeout error occurred.")

        return BedrockRerankConfig()._transform_response(response.json())

    def rerank(
        self,
        model: str,
        query: str,
        documents: List[Union[str, Dict[str, Any]]],
        optional_params: dict,
        logging_obj: LitellmLogging,
        top_n: Optional[int] = None,
        rank_fields: Optional[List[str]] = None,
        return_documents: Optional[bool] = True,
        max_chunks_per_doc: Optional[int] = None,
        _is_async: Optional[bool] = False,
        api_base: Optional[str] = None,
        extra_headers: Optional[dict] = None,
    ) -> RerankResponse:
        request_data = RerankRequest(
            model=model,
            query=query,
            documents=documents,
            top_n=top_n,
            rank_fields=rank_fields,
            return_documents=return_documents,
        )
        data = BedrockRerankConfig()._transform_request(request_data)

        prepared_request = self._prepare_request(
            optional_params=optional_params,
            api_base=api_base,
            extra_headers=extra_headers,
            data=cast(dict, data),
        )

        logging_obj.pre_call(
            input=data,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": prepared_request["endpoint_url"],
                "headers": prepared_request["prepped"].headers,
            },
        )

        if _is_async:
            return self.arerank(prepared_request)  # type: ignore

        client = _get_httpx_client()
        try:
            response = client.post(url=prepared_request["endpoint_url"], headers=prepared_request["prepped"].headers, data=prepared_request["body"])  # type: ignore
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            error_code = err.response.status_code
            raise BedrockError(status_code=error_code, message=err.response.text)
        except httpx.TimeoutException:
            raise BedrockError(status_code=408, message="Timeout error occurred.")

        return BedrockRerankConfig()._transform_response(response.json())

    def _prepare_request(
        self,
        api_base: Optional[str],
        extra_headers: Optional[dict],
        data: dict,
        optional_params: dict,
    ) -> BedrockPreparedRequest:
        try:
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
        except ImportError:
            raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")
        boto3_credentials_info = self._get_boto_credentials_from_optional_params(
            optional_params
        )

        ### SET RUNTIME ENDPOINT ###
        _, proxy_endpoint_url = self.get_runtime_endpoint(
            api_base=api_base,
            aws_bedrock_runtime_endpoint=boto3_credentials_info.aws_bedrock_runtime_endpoint,
            aws_region_name=boto3_credentials_info.aws_region_name,
        )
        proxy_endpoint_url = proxy_endpoint_url.replace(
            "bedrock-runtime", "bedrock-agent-runtime"
        )
        proxy_endpoint_url = f"{proxy_endpoint_url}/rerank"
        sigv4 = SigV4Auth(
            boto3_credentials_info.credentials,
            "bedrock",
            boto3_credentials_info.aws_region_name,
        )
        # Make POST Request
        body = json.dumps(data).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        if extra_headers is not None:
            headers = {"Content-Type": "application/json", **extra_headers}
        request = AWSRequest(
            method="POST", url=proxy_endpoint_url, data=body, headers=headers
        )
        sigv4.add_auth(request)
        if (
            extra_headers is not None and "Authorization" in extra_headers
        ):  # prevent sigv4 from overwriting the auth header
            request.headers["Authorization"] = extra_headers["Authorization"]
        prepped = request.prepare()

        return BedrockPreparedRequest(
            endpoint_url=proxy_endpoint_url,
            prepped=prepped,
            body=body,
            data=data,
        )
