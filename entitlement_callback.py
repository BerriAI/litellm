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
        self.default_action = os.environ.get("ENTITLEMENT_ACTION", "answer")
        super().__init__()  # initialize base class if needed
    
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: Union[Any, ModelResponse, EmbeddingResponse, ImageResponse],
    ) -> Any:
        model = data["model"]
        external_customer_id = data["user"]
        import pdb; pdb.set_trace()
        response_cost = float(data["metadata"]['hidden_params']['response_cost'])
        self.handle_entitlement_check(external_customer_id, response_cost)
        return data
        
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
            "amount": response_cost * 10000 # for testing
        }
        
        payload = {
            "external_customer_id": external_customer_id,
            "publisher_id": self.publisher_id,
            "action_name": self.default_action,
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
                import pdb; pdb.set_trace()
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
        # request_data = {'stream': True, 'model': 'gemini-2.0-flash-lite', 'messages': [{'role': 'user', 'content': 'hello'}, {'role': 'assistant', 'content': 'Hello! How can I help you today?'}, {'role': 'user', 'content': 'who are you'}], 'user': {'name': '14a56377-0e0e-4226-a938-d2e727ee4355@monetanetwork.com', 'id': 'b16b8dfc-7838-4e5b-8a8e-2e94daa677fe', 'email': '14a56377-0e0e-4226-a938-d2e727ee4355@monetanetwork.com', 'role': 'user'}, 'proxy_server_request': {'url': 'http://localhost:4000/chat/completions', 'method': 'POST', 'headers': {'host': 'localhost:4000', 'content-type': 'application/json', 'accept': '*/*', 'accept-encoding': 'gzip, deflate', 'user-agent': 'Python/3.11 aiohttp/3.11.11', 'content-length': '421'}, 'body': {'stream': True, 'model': 'gemini-2.0-flash-lite', 'messages': [{'role': 'user', 'content': 'hello'}, {'role': 'assistant', 'content': 'Hello! How can I help you today?'}, {'role': 'user', 'content': 'who are you'}], 'user': {'name': '14a56377-0e0e-4226-a938-d2e727ee4355@monetanetwork.com', 'id': 'b16b8dfc-7838-4e5b-8a8e-2e94daa677fe', 'email': '14a56377-0e0e-4226-a938-d2e727ee4355@monetanetwork.com', 'role': 'user'}}}, 'metadata': {'requester_metadata': {}, 'user_api_key_hash': '03aba17bfded56944d24dd00cdd2b1ce052bbe9b8708b79ba47f95ae328e5a84', 'user_api_key_alias': 'demo', 'user_api_key_team_id': None, 'user_api_key_user_id': 'default_user_id', 'user_api_key_org_id': None, 'user_api_key_team_alias': None, 'user_api_key_end_user_id': "{'name': '14a56377-0e0e-4226-a938-d2e727ee4355@monetanetwork.com', 'id': 'b16b8dfc-7838-4e5b-8a8e-2e94daa677fe', 'email': '14a56377-0e0e-4226-a938-d2e727ee4355@monetanetwork.com', 'role': 'user'}", 'user_api_key_user_email': None, 'user_api_key': '03aba17bfded56944d24dd00cdd2b1ce052bbe9b8708b79ba47f95ae328e5a84', 'user_api_end_user_max_budget': None, 'litellm_api_version': '1.65.4.post1', 'global_max_parallel_requests': None, 'user_api_key_team_max_budget': None, 'user_api_key_team_spend': None, 'user_api_key_spend': 0.0010011749999999998, 'user_api_key_max_budget': None, 'user_api_key_model_max_budget': {}, 'user_api_key_metadata': {}, 'headers': {'host': 'localhost:4000', 'content-type': 'application/json', 'accept': '*/*', 'accept-encoding': 'gzip, deflate', 'user-agent': 'Python/3.11 aiohttp/3.11.11', 'content-length': '421'}, 'endpoint': 'http://localhost:4000/chat/completions', 'litellm_parent_otel_span': None, 'requester_ip_address': '', 'model_group': 'gemini-2.0-flash-lite', 'model_group_size': 1, 'deployment': 'gemini/gemini-2.0-flash-lite', 'model_info': {'id': '3e3bc7b32bc5e8868a58b30e7b4d82aa80a72e046340863b47e6cbe72b809271', 'db_model': False}, 'caching_groups': None}, 'litellm_call_id': '32b694f4-10b4-4da2-a7f3-b179f4a498a8', 'litellm_logging_obj': <litellm.litellm_core_utils.litellm_logging.Logging object at 0x10e529940>, 'deployment': Deployment(model_name='gemini-2.0-flash-lite', litellm_params=LiteLLM_Params(api_key='AIzaSyCpgVr271A6OG4bT9C5Jcy_463gIs80sco', api_base=None, api_version=None, vertex_project=None, vertex_location=None, vertex_credentials=None, region_name=None, aws_access_key_id=None, aws_secret_access_key=None, aws_region_name=None, watsonx_region_name=None, custom_llm_provider=None, tpm=None, rpm=None, timeout=None, stream_timeout=None, max_retries=None, organization=None, configurable_clientside_auth_params=None, litellm_credential_name=None, litellm_trace_id=None, input_cost_per_token=None, output_cost_per_token=None, input_cost_per_second=None, output_cost_per_second=None, max_file_size_mb=None, max_budget=None, budget_duration=None, use_in_pass_through=False, merge_reasoning_content_in_choices=False, model_info=None, model='gemini/gemini-2.0-flash-lite'), model_info=ModelInfo(id='3e3bc7b32bc5e8868a58b30e7b4d82aa80a72e046340863b47e6cbe72b809271', db_model=False, updated_at=None, updated_by=None, created_at=None, created_by=None, base_model=None, tier=None, team_id=None, team_public_model_name=None))}
        # handle lago from here.
        user_id = request_data["metadata"]["headers"]["x-openwebui-user-id"]
        # user_id = "6af82351-ab67-4b45-8ef5-51aa17c79ef8"
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
        
        # import pdb; pdb.set_trace()
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
        # debug_items = [] # Keep commented unless debugging
        async for item in response:
            # debug_items.append(item) # Keep commented unless debugging
            yield item
    
        

# Instantiate the callback (instance name will be used in config)
entitlement_checker = EntitlementCallback()