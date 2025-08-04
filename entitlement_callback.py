import os, time, asyncio
import uuid
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

from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    List,
    Literal,
    Optional,
    Tuple,
    Union
)

from litellm.caching.caching import DualCache
from datetime import datetime
from fastapi import HTTPException

def get_utc_datetime():
    import datetime as dt
    

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
    
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings", 
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank",
        ],
    ) -> Optional[Union[Exception, str, dict]]:
        # Implement entitlement check here
        try: 
            print("pre call hook")
            print("data", data)
            user_id = data["metadata"]["headers"]["x-openwebui-user-id"]
            litellm_call_id = data["litellm_call_id"]
            external_customer_id = user_id
            external_subscription_id = await self.handle_entitlement_check(external_customer_id)
            print("external_subscription_id", external_subscription_id)
            return data
        except Exception as e:
            raise HTTPException(status_code=402, detail={"error": "Access denied by entitlement"})

    async def async_log_success_event(
        self, 
        kwargs: dict, 
        response_obj: Any,  
        start_time: datetime, 
        end_time: datetime
    ) -> None:
        # Implement usage reporting here
        usage = response_obj.get("usage", {})
        model = kwargs.get("model")
        litellm_call_id = kwargs["litellm_call_id"]
        response_cost = kwargs.get("response_cost", 0)
        print("kwargs", kwargs)
        # Get customer ID (same logic as pre-call)
       
        pass

    async def handle_entitlement_check(
        self,
        external_customer_id
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
            "tags": []
        }
        
        payload = {
            "external_customer_id": external_customer_id,
            "publisher_id": self.publisher_id,
            "emit_event": False,
            "action_name": self.default_action,
            "context": [],
            "resource": entitlement_resource,
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
        external_subscription_id = result.get("extra").get("authorized_subscription").get("external_id")
        return external_subscription_id

    async def send_usage_event(self, external_subscription_id, response_cost):
        usage_event_api = f"{self.api_base}/api/v1/events"
        # TODO: send event
        payload = {
            "transaction_id": str(uuid.uuid4()),
            "external_subscription_id": external_subscription_id,
            "code": "credits_in_cent",
            "properties": {
                "credits_in_cent": response_cost * 100
            }   
        }
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                headers = { "Authorization": f"Bearer {os.environ.get('MONETA_LAGO_API_KEY')}" }
                response = await client.post(usage_event_api, json=payload, headers=headers)
                print(f"Usage event response: {response}")
                print(f"Usage event response body: {response.text}")
                return response
        except Exception as e:
            raise HTTPException(status_code=403, detail={"error": "Usage event error"})

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        async for item in response:
            yield item
    #     self,
    #     user_api_key_dict: UserAPIKeyAuth,
    #     response: Any,
    #     request_data: dict,
    # ) -> AsyncGenerator[ModelResponseStream, None]:
    #     print("begin stream hook")
    #     print("request_data", request_data)
    #     print("response", response)
    #     print("end stream hook")
    #     user_id = request_data["metadata"]["headers"]["x-openwebui-user-id"]
    #     external_customer_id = user_id
    #     user_api_key_spend = request_data["metadata"]['user_api_key_spend']
    #     response_cost = user_api_key_spend
    #     access_denied = False
    #     error_de3tail = "Access denied by entitlement" # Default error message
    #     print("response_cost", response_cost)
    #     try:
    #         await self.handle_entitlement_check(external_customer_id, response_cost)
    #     except HTTPException as e:
    #          print(f"Entitlement check failed (HTTPException): {e.detail}")
    #          access_denied = True
    #          error_detail = e.detail.get("error", error_detail) if isinstance(e.detail, dict) else str(e.detail)
    #     except Exception as e:
    #         print(f"Entitlement check failed (Exception): {e}")
    #         access_denied = True
    #         error_detail = f"Entitlement check error: {e}"
        
    #     if access_denied:
    #          print(f"ACCESS DENIED for {external_customer_id} due to entitlement failure: {error_detail}")
    #          # Simplest approach: Yield a single final error chunk and stop.
    #          # Note: This is in a post-call hook, so stream might have already started.
    #          error_chunk = ModelResponseStream(
    #              id='error-chunk-final',
    #              created=int(time.time()),
    #              model=request_data.get("model", "unknown-model"),
    #              object='chat.completion.chunk',
    #              choices=[StreamingChoices(
    #                  finish_reason='stop', 
    #                  index=0, 
    #                  delta=Delta(role="assistant", content=error_detail) # Use captured error detail
    #              )]
    #          )
    #          yield error_chunk
    #          return # Stop the generator here

    #     # If access was not denied, proceed with the original stream
    #     print("access granted")
    #     async for item in response:
    #         yield item
    
        

# Instantiate the callback (instance name will be used in config)
entitlement_checker = EntitlementCallback()
