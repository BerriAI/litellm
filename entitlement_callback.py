import os, time, asyncio
import httpx  # Using httpx for async HTTP calls (pip install httpx if not installed)
from litellm.integrations.custom_logger import CustomLogger
from typing import Any, Union, AsyncGenerator
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import (
    AdapterCompletionStreamWrapper,
    EmbeddingResponse,
    ImageResponse,
    ModelResponse,
    ModelResponseStream,
    StandardCallbackDynamicParams,
    StandardLoggingPayload,
    StreamingChoices,
    Delta
)

from fastapi import HTTPException

def get_utc_datetime():
    import datetime as dt
    from datetime import datetime

    if hasattr(dt, "UTC"):
        return datetime.now(dt.UTC)  # type: ignore
    else:
        return datetime.utcnow()  # type: ignore
    
class EntitlementCallback(CustomLogger):
    def __init__(self):
        # Read any required config (e.g., publisher_id, API base URL) from environment
        self.publisher_id = os.environ.get("PUBLISHER_ID", "GPTPortalHub")
        self.api_base = os.environ.get("ENTITLEMENT_API_BASE", "").rstrip("/")
        # Construct full URL for the entitlement API endpoint
        self.auth_url = f"{self.api_base}/v1/entitlement/authorize"
        # Optionally, define a default action or other constants
        self.default_action = os.environ.get("ENTITLEMENT_ACTION", "read")
        super().__init__()  # initialize base class if needed
    
    # async def async_post_call_success_hook(
    #     self,
    #     data: dict,
    #     user_api_key_dict: UserAPIKeyAuth,
    #     response: Union[Any, ModelResponse, EmbeddingResponse, ImageResponse],
    # ) -> Any:
        # model = data["model"]
        # external_customer_id = data["user"]
        # response_cost = float(data["metadata"]['hidden_params']['response_cost'])
        # self.handle_entitlement_check(external_customer_id, response_cost)
        # return data
        
    # async def async_post_call_streaming_hook(
    #     self,
    #     data: dict,
    #     user_api_key_dict: UserAPIKeyAuth,
    #     response: Union[Any, ModelResponse, EmbeddingResponse, ImageResponse],
    # ) -> Any:
    #     # khong co data
    #     return await self.handle_entitlement_check(data, user_api_key_dict, response)
    
    async def handle_entitlement_check(
        self,
        external_customer_id,
        response_cost
    ):
        # if "Hello world" in formatted_prompt:
        print("callme")
        
        # return "This is an invalid response"
        # external_customer_id = data["user"]
        # response_cost = float(data["metadata"]['hidden_params']['response_cost'])
        
        if not external_customer_id:
            return "Missing required customer ID for entitlement check."

        entitlement_resource = { 
            "id": 1, 
            "name": "", 
            "type": "article",
            "author": "any.email@is.fine",
            "tags": [],
            "amount": response_cost * 100
        }
        
        payload = {
            "external_customer_id": external_customer_id,
            "publisher_id": self.publisher_id,
            "action_name": self.default_action,
            "context": [],
            "resource": entitlement_resource,
            "properties": {
                "credits_in_cent": response_cost * 100
            },
            "timestamp": int(time.time())
        }

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                headers = { "Authorization": f"Bearer {os.environ.get('MONETA_LAGO_API_KEY')}" }
                print(f"Entitlement auth_url: {self.auth_url}")
                print(f"Entitlement headers: {headers}")
                print(f"Entitlement payload: {payload}")
                response = await client.post(self.auth_url, json=payload, headers=headers)
                print(f"Entitlement response: {response}")
                print(f"response body: {response.text}")
        except Exception as e:
            return f"Entitlement check error: {e}"

        if response.status_code != 200:
            raise HTTPException(status_code=403, detail={"error": "Access denied by entitlement"})
        
        result = response.json()
        allowed = bool(result.get("status") == "Allow")
        if not allowed:
            raise HTTPException(status_code=403, detail={"error": "Access denied by entitlement"})

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        user_id = request_data["metadata"]["headers"]["x-openwebui-user-id"]
        external_customer_id = user_id
        user_api_key_spend = request_data["metadata"]['user_api_key_spend']
        response_cost = user_api_key_spend
        access_denied = False
        error_detail = "Access denied by entitlement" # Default error message
        try:
            await self.handle_entitlement_check(external_customer_id, response_cost)
        except HTTPException as e:
             print(f"Entitlement check failed (HTTPException): {e.detail}")
             access_denied = True
             error_detail = e.detail.get("error", error_detail) if isinstance(e.detail, dict) else str(e.detail)
        except Exception as e:
            print(f"Entitlement check failed (Exception): {e}")
            access_denied = True
            error_detail = f"Entitlement check error: {e}"
        
        if access_denied:
             print(f"ACCESS DENIED for {external_customer_id} due to entitlement failure: {error_detail}")
             # Simplest approach: Yield a single final error chunk and stop.
             # Note: This is in a post-call hook, so stream might have already started.
             error_chunk = ModelResponseStream(
                 id='error-chunk-final',
                 created=int(time.time()),
                 model=request_data.get("model", "unknown-model"),
                 object='chat.completion.chunk',
                 choices=[StreamingChoices(
                     finish_reason='stop', 
                     index=0, 
                     delta=Delta(role="assistant", content=error_detail) # Use captured error detail
                 )]
             )
             yield error_chunk
             return # Stop the generator here

        # If access was not denied, proceed with the original stream
        print("access granted")
        async for item in response:
            yield item
    
        

# Instantiate the callback (instance name will be used in config)
entitlement_checker = EntitlementCallback()
