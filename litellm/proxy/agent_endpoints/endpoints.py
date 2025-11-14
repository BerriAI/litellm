# """
# Agent endpoints for registering + discovering agents via LiteLLM.

# Follows the A2A Spec.

# 1. Register an agent via POST `/v1/agents`
# 2. Discover agents via GET `/v1/agents`
# 3. Get specific agent via GET `/v1/agents/{agent_id}`
# """

# import asyncio

# from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

# import litellm
# from litellm._logging import verbose_proxy_logger
# from litellm.proxy._types import *
# from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
# from litellm.proxy.common_request_processing import (
#     ProxyBaseLLMRequestProcessing,
#     create_streaming_response,
# )
# from litellm.proxy.common_utils.http_parsing_utils import _read_request_body
# from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request
# from litellm.types.utils import TokenCountResponse

# router = APIRouter()


# @router.get(
#     "/v1/messages/count_tokens",
#     tags=["[beta] Anthropic Messages Token Counting"],
#     dependencies=[Depends(user_api_key_auth)],
# )
# async def count_tokens(
#     request: Request,
#     user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),  # Used for auth
# ):
#     """
#     Count tokens for Anthropic Messages API format.

#     This endpoint follows the Anthropic Messages API token counting specification.
#     It accepts the same parameters as the /v1/messages endpoint but returns
#     token counts instead of generating a response.

#     Example usage:
#     ```
#     curl -X POST "http://localhost:4000/v1/messages/count_tokens?beta=true" \
#       -H "Content-Type: application/json" \
#       -H "Authorization: Bearer your-key" \
#       -d '{
#         "model": "claude-3-sonnet-20240229",
#         "messages": [{"role": "user", "content": "Hello Claude!"}]
#       }'
#     ```

#     Returns: {"input_tokens": <number>}
#     """
#     from litellm.proxy.proxy_server import token_counter as internal_token_counter

#     try:
#         request_data = await _read_request_body(request=request)
#         data: dict = {**request_data}

#         # Extract required fields
#         model_name = data.get("model")
#         messages = data.get("messages", [])

#         if not model_name:
#             raise HTTPException(
#                 status_code=400, detail={"error": "model parameter is required"}
#             )

#         if not messages:
#             raise HTTPException(
#                 status_code=400, detail={"error": "messages parameter is required"}
#             )

#         # Create TokenCountRequest for the internal endpoint
#         from litellm.proxy._types import TokenCountRequest

#         token_request = TokenCountRequest(model=model_name, messages=messages)

#         # Call the internal token counter function with direct request flag set to False
#         token_response = await internal_token_counter(
#             request=token_request,
#             call_endpoint=True,
#         )
#         _token_response_dict: dict = {}
#         if isinstance(token_response, TokenCountResponse):
#             _token_response_dict = token_response.model_dump()
#         elif isinstance(token_response, dict):
#             _token_response_dict = token_response

#         # Convert the internal response to Anthropic API format
#         return {"input_tokens": _token_response_dict.get("total_tokens", 0)}

#     except HTTPException:
#         raise
#     except Exception as e:
#         verbose_proxy_logger.exception(
#             "litellm.proxy.anthropic_endpoints.count_tokens(): Exception occurred - {}".format(
#                 str(e)
#             )
#         )
#         raise HTTPException(
#             status_code=500, detail={"error": f"Internal server error: {str(e)}"}
#         )
