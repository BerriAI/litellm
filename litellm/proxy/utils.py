# import litellm
# from litellm import ModelResponse
# from proxy_server import update_verification_token_cost
# from typing import Optional
# from fastapi import HTTPException, status
# import asyncio

# def track_cost_callback(
#     kwargs,                                       # kwargs to completion
#     completion_response: ModelResponse,           # response from completion
#     start_time = None,
#     end_time = None,                              # start/end time for completion
# ):
#     try:
#         # init logging config
#         api_key = kwargs["litellm_params"]["metadata"]["api_key"]
#         # check if it has collected an entire stream response
#         if "complete_streaming_response" in kwargs:
#             # for tracking streaming cost we pass the "messages" and the output_text to litellm.completion_cost 
#             completion_response=kwargs["complete_streaming_response"]
#             input_text = kwargs["messages"]
#             output_text = completion_response["choices"][0]["message"]["content"]
#             response_cost = litellm.completion_cost(
#                 model = kwargs["model"],
#                 messages = input_text,
#                 completion=output_text
#             )
#             print(f"LiteLLM Proxy: streaming response_cost: {response_cost} for api_key: {api_key}")
#         # for non streaming responses
#         else:
#             # we pass the completion_response obj
#             if kwargs["stream"] != True:
#                 response_cost = litellm.completion_cost(completion_response=completion_response)
#                 print(f"\n LiteLLM Proxy: regular response_cost: {response_cost} for api_key: {api_key}")
        
#         ########### write costs to DB api_key / cost map
#         asyncio.run(
#             update_verification_token_cost(token=api_key, additional_cost=response_cost)
#         )
#     except:
#         pass