# ####################################
# ######### DEPRECATED FILE ##########
# ####################################
# # logic moved to `bedrock_httpx.py` #

# import copy
# import json
# import os
# import time
# import types
# import uuid
# from enum import Enum
# from typing import Any, Callable, List, Optional, Union

# import httpx
# from openai.types.image import Image

# import litellm
# from litellm.litellm_core_utils.core_helpers import map_finish_reason
# from litellm.types.utils import ImageResponse, ModelResponse, Usage
# from litellm.utils import get_secret

# from .prompt_templates.factory import (
#     construct_tool_use_system_prompt,
#     contains_tag,
#     custom_prompt,
#     extract_between_tags,
#     parse_xml_params,
#     prompt_factory,
# )


# def convert_messages_to_prompt(model, messages, provider, custom_prompt_dict):
#     # handle anthropic prompts and amazon titan prompts
#     chat_template_provider = ["anthropic", "amazon", "mistral", "meta"]
#     if model in custom_prompt_dict:
#         # check if the model has a registered custom prompt
#         model_prompt_details = custom_prompt_dict[model]
#         prompt = custom_prompt(
#             role_dict=model_prompt_details["roles"],
#             initial_prompt_value=model_prompt_details["initial_prompt_value"],
#             final_prompt_value=model_prompt_details["final_prompt_value"],
#             messages=messages,
#         )
#     else:
#         if provider in chat_template_provider:
#             prompt = prompt_factory(
#                 model=model, messages=messages, custom_llm_provider="bedrock"
#             )
#         else:
#             prompt = ""
#             for message in messages:
#                 if "role" in message:
#                     if message["role"] == "user":
#                         prompt += f"{message['content']}"
#                     else:
#                         prompt += f"{message['content']}"
#                 else:
#                     prompt += f"{message['content']}"
#     return prompt


# """
# BEDROCK AUTH Keys/Vars
# os.environ['AWS_ACCESS_KEY_ID'] = ""
# os.environ['AWS_SECRET_ACCESS_KEY'] = ""
# """


# # set os.environ['AWS_REGION_NAME'] = <your-region_name>


# def completion(
#     model: str,
#     messages: list,
#     custom_prompt_dict: dict,
#     model_response: ModelResponse,
#     print_verbose: Callable,
#     encoding,
#     logging_obj,
#     optional_params=None,
#     litellm_params=None,
#     logger_fn=None,
#     timeout=None,
#     extra_headers: Optional[dict] = None,
# ):
#     exception_mapping_worked = False
#     _is_function_call = False
#     json_schemas: dict = {}
#     try:
#         # pop aws_secret_access_key, aws_access_key_id, aws_region_name from kwargs, since completion calls fail with them
#         aws_secret_access_key = optional_params.pop("aws_secret_access_key", None)
#         aws_access_key_id = optional_params.pop("aws_access_key_id", None)
#         aws_region_name = optional_params.pop("aws_region_name", None)
#         aws_role_name = optional_params.pop("aws_role_name", None)
#         aws_session_name = optional_params.pop("aws_session_name", None)
#         aws_profile_name = optional_params.pop("aws_profile_name", None)
#         aws_bedrock_runtime_endpoint = optional_params.pop(
#             "aws_bedrock_runtime_endpoint", None
#         )
#         aws_web_identity_token = optional_params.pop("aws_web_identity_token", None)

#         # use passed in BedrockRuntime.Client if provided, otherwise create a new one
#         client = optional_params.pop("aws_bedrock_client", None)

#         # only init client, if user did not pass one
#         if client is None:
#             client = init_bedrock_client(
#                 aws_access_key_id=aws_access_key_id,
#                 aws_secret_access_key=aws_secret_access_key,
#                 aws_region_name=aws_region_name,
#                 aws_bedrock_runtime_endpoint=aws_bedrock_runtime_endpoint,
#                 aws_role_name=aws_role_name,
#                 aws_session_name=aws_session_name,
#                 aws_profile_name=aws_profile_name,
#                 aws_web_identity_token=aws_web_identity_token,
#                 extra_headers=extra_headers,
#                 timeout=timeout,
#             )

#         model = model
#         modelId = (
#             optional_params.pop("model_id", None) or model
#         )  # default to model if not passed
#         provider = model.split(".")[0]
#         prompt = convert_messages_to_prompt(
#             model, messages, provider, custom_prompt_dict
#         )
#         inference_params = copy.deepcopy(optional_params)
#         stream = inference_params.pop("stream", False)
#         if provider == "anthropic":
#             if model.startswith("anthropic.claude-3"):
#                 # Separate system prompt from rest of message
#                 system_prompt_idx: list[int] = []
#                 system_messages: list[str] = []
#                 for idx, message in enumerate(messages):
#                     if message["role"] == "system":
#                         system_messages.append(message["content"])
#                         system_prompt_idx.append(idx)
#                 if len(system_prompt_idx) > 0:
#                     inference_params["system"] = "\n".join(system_messages)
#                     messages = [
#                         i for j, i in enumerate(messages) if j not in system_prompt_idx
#                     ]
#                 # Format rest of message according to anthropic guidelines
#                 messages = prompt_factory(
#                     model=model, messages=messages, custom_llm_provider="anthropic_xml"
#                 )
#                 ## LOAD CONFIG
#                 config = litellm.AmazonAnthropicClaude3Config.get_config()
#                 for k, v in config.items():
#                     if (
#                         k not in inference_params
#                     ):  # completion(top_k=3) > anthropic_config(top_k=3) <- allows for dynamic variables to be passed in
#                         inference_params[k] = v
#                 ## Handle Tool Calling
#                 if "tools" in inference_params:
#                     _is_function_call = True
#                     for tool in inference_params["tools"]:
#                         json_schemas[tool["function"]["name"]] = tool["function"].get(
#                             "parameters", None
#                         )
#                     tool_calling_system_prompt = construct_tool_use_system_prompt(
#                         tools=inference_params["tools"]
#                     )
#                     inference_params["system"] = (
#                         inference_params.get("system", "\n")
#                         + tool_calling_system_prompt
#                     )  # add the anthropic tool calling prompt to the system prompt
#                     inference_params.pop("tools")
#                 data = json.dumps({"messages": messages, **inference_params})
#             else:
#                 ## LOAD CONFIG
#                 config = litellm.AmazonAnthropicConfig.get_config()
#                 for k, v in config.items():
#                     if (
#                         k not in inference_params
#                     ):  # completion(top_k=3) > anthropic_config(top_k=3) <- allows for dynamic variables to be passed in
#                         inference_params[k] = v
#                 data = json.dumps({"prompt": prompt, **inference_params})
#         elif provider == "ai21":
#             ## LOAD CONFIG
#             config = litellm.AmazonAI21Config.get_config()
#             for k, v in config.items():
#                 if (
#                     k not in inference_params
#                 ):  # completion(top_k=3) > anthropic_config(top_k=3) <- allows for dynamic variables to be passed in
#                     inference_params[k] = v

#             data = json.dumps({"prompt": prompt, **inference_params})
#         elif provider == "cohere":
#             ## LOAD CONFIG
#             config = litellm.AmazonCohereConfig.get_config()
#             for k, v in config.items():
#                 if (
#                     k not in inference_params
#                 ):  # completion(top_k=3) > anthropic_config(top_k=3) <- allows for dynamic variables to be passed in
#                     inference_params[k] = v
#             if optional_params.get("stream", False) == True:
#                 inference_params["stream"] = (
#                     True  # cohere requires stream = True in inference params
#                 )
#             data = json.dumps({"prompt": prompt, **inference_params})
#         elif provider == "meta":
#             ## LOAD CONFIG
#             config = litellm.AmazonLlamaConfig.get_config()
#             for k, v in config.items():
#                 if (
#                     k not in inference_params
#                 ):  # completion(top_k=3) > anthropic_config(top_k=3) <- allows for dynamic variables to be passed in
#                     inference_params[k] = v
#             data = json.dumps({"prompt": prompt, **inference_params})
#         elif provider == "amazon":  # amazon titan
#             ## LOAD CONFIG
#             config = litellm.AmazonTitanConfig.get_config()
#             for k, v in config.items():
#                 if (
#                     k not in inference_params
#                 ):  # completion(top_k=3) > amazon_config(top_k=3) <- allows for dynamic variables to be passed in
#                     inference_params[k] = v

#             data = json.dumps(
#                 {
#                     "inputText": prompt,
#                     "textGenerationConfig": inference_params,
#                 }
#             )
#         elif provider == "mistral":
#             ## LOAD CONFIG
#             config = litellm.AmazonMistralConfig.get_config()
#             for k, v in config.items():
#                 if (
#                     k not in inference_params
#                 ):  # completion(top_k=3) > amazon_config(top_k=3) <- allows for dynamic variables to be passed in
#                     inference_params[k] = v

#             data = json.dumps({"prompt": prompt, **inference_params})
#         else:
#             data = json.dumps({})

#         ## COMPLETION CALL
#         accept = "application/json"
#         contentType = "application/json"
#         if stream == True and _is_function_call == False:
#             if provider == "ai21":
#                 ## LOGGING
#                 request_str = f"""
#                 response = client.invoke_model(
#                     body={data},
#                     modelId={modelId},
#                     accept=accept,
#                     contentType=contentType
#                 )
#                 """
#                 logging_obj.pre_call(
#                     input=prompt,
#                     api_key="",
#                     additional_args={
#                         "complete_input_dict": data,
#                         "request_str": request_str,
#                     },
#                 )

#                 response = client.invoke_model(
#                     body=data, modelId=modelId, accept=accept, contentType=contentType
#                 )

#                 response = response.get("body").read()
#                 return response
#             else:
#                 ## LOGGING
#                 request_str = f"""
#                 response = client.invoke_model_with_response_stream(
#                     body={data},
#                     modelId={modelId},
#                     accept=accept,
#                     contentType=contentType
#                 )
#                 """
#                 logging_obj.pre_call(
#                     input=prompt,
#                     api_key="",
#                     additional_args={
#                         "complete_input_dict": data,
#                         "request_str": request_str,
#                     },
#                 )

#                 response = client.invoke_model_with_response_stream(
#                     body=data, modelId=modelId, accept=accept, contentType=contentType
#                 )
#                 response = response.get("body")
#                 return response
#         try:
#             ## LOGGING
#             request_str = f"""
#             response = client.invoke_model(
#                 body={data},
#                 modelId={modelId},
#                 accept=accept,
#                 contentType=contentType
#             )
#             """
#             logging_obj.pre_call(
#                 input=prompt,
#                 api_key="",
#                 additional_args={
#                     "complete_input_dict": data,
#                     "request_str": request_str,
#                 },
#             )
#             response = client.invoke_model(
#                 body=data, modelId=modelId, accept=accept, contentType=contentType
#             )
#         except client.exceptions.ValidationException as e:
#             if "The provided model identifier is invalid" in str(e):
#                 raise BedrockError(status_code=404, message=str(e))
#             raise BedrockError(status_code=400, message=str(e))
#         except Exception as e:
#             raise BedrockError(status_code=500, message=str(e))

#         response_body = json.loads(response.get("body").read())

#         ## LOGGING
#         logging_obj.post_call(
#             input=prompt,
#             api_key="",
#             original_response=json.dumps(response_body),
#             additional_args={"complete_input_dict": data},
#         )
#         print_verbose(f"raw model_response: {response_body}")
#         ## RESPONSE OBJECT
#         outputText = "default"
#         if provider == "ai21":
#             outputText = response_body.get("completions")[0].get("data").get("text")
#         elif provider == "anthropic":
#             if model.startswith("anthropic.claude-3"):
#                 outputText = response_body.get("content")[0].get("text", None)
#                 if outputText is not None and contains_tag(
#                     "invoke", outputText
#                 ):  # OUTPUT PARSE FUNCTION CALL
#                     function_name = extract_between_tags("tool_name", outputText)[0]
#                     function_arguments_str = extract_between_tags("invoke", outputText)[
#                         0
#                     ].strip()
#                     function_arguments_str = (
#                         f"<invoke>{function_arguments_str}</invoke>"
#                     )
#                     function_arguments = parse_xml_params(
#                         function_arguments_str,
#                         json_schema=json_schemas.get(
#                             function_name, None
#                         ),  # check if we have a json schema for this function name)
#                     )
#                     _message = litellm.Message(
#                         tool_calls=[
#                             {
#                                 "id": f"call_{uuid.uuid4()}",
#                                 "type": "function",
#                                 "function": {
#                                     "name": function_name,
#                                     "arguments": json.dumps(function_arguments),
#                                 },
#                             }
#                         ],
#                         content=None,
#                     )
#                     model_response.choices[0].message = _message  # type: ignore
#                     model_response._hidden_params["original_response"] = (
#                         outputText  # allow user to access raw anthropic tool calling response
#                     )
#                 if _is_function_call == True and stream is not None and stream == True:
#                     print_verbose(
#                         f"INSIDE BEDROCK STREAMING TOOL CALLING CONDITION BLOCK"
#                     )
#                     # return an iterator
#                     streaming_model_response = ModelResponse(stream=True)
#                     streaming_model_response.choices[0].finish_reason = (
#                         model_response.choices[0].finish_reason
#                     )
#                     # streaming_model_response.choices = [litellm.utils.StreamingChoices()]
#                     streaming_choice = litellm.utils.StreamingChoices()
#                     streaming_choice.index = model_response.choices[0].index
#                     _tool_calls = []
#                     print_verbose(
#                         f"type of model_response.choices[0]: {type(model_response.choices[0])}"
#                     )
#                     print_verbose(f"type of streaming_choice: {type(streaming_choice)}")
#                     if isinstance(model_response.choices[0], litellm.Choices):
#                         if getattr(
#                             model_response.choices[0].message, "tool_calls", None
#                         ) is not None and isinstance(
#                             model_response.choices[0].message.tool_calls, list
#                         ):
#                             for tool_call in model_response.choices[
#                                 0
#                             ].message.tool_calls:
#                                 _tool_call = {**tool_call.dict(), "index": 0}
#                                 _tool_calls.append(_tool_call)
#                         delta_obj = litellm.utils.Delta(
#                             content=getattr(
#                                 model_response.choices[0].message, "content", None
#                             ),
#                             role=model_response.choices[0].message.role,
#                             tool_calls=_tool_calls,
#                         )
#                         streaming_choice.delta = delta_obj
#                         streaming_model_response.choices = [streaming_choice]
#                         completion_stream = ModelResponseIterator(
#                             model_response=streaming_model_response
#                         )
#                         print_verbose(
#                             f"Returns anthropic CustomStreamWrapper with 'cached_response' streaming object"
#                         )
#                         return litellm.CustomStreamWrapper(
#                             completion_stream=completion_stream,
#                             model=model,
#                             custom_llm_provider="cached_response",
#                             logging_obj=logging_obj,
#                         )

#                 model_response.choices[0].finish_reason = map_finish_reason(
#                     response_body["stop_reason"]
#                 )
#                 _usage = litellm.Usage(
#                     prompt_tokens=response_body["usage"]["input_tokens"],
#                     completion_tokens=response_body["usage"]["output_tokens"],
#                     total_tokens=response_body["usage"]["input_tokens"]
#                     + response_body["usage"]["output_tokens"],
#                 )
#                 setattr(model_response, "usage", _usage)
#             else:
#                 outputText = response_body["completion"]
#                 model_response.choices[0].finish_reason = response_body["stop_reason"]
#         elif provider == "cohere":
#             outputText = response_body["generations"][0]["text"]
#         elif provider == "meta":
#             outputText = response_body["generation"]
#         elif provider == "mistral":
#             outputText = response_body["outputs"][0]["text"]
#             model_response.choices[0].finish_reason = response_body["outputs"][0][
#                 "stop_reason"
#             ]
#         else:  # amazon titan
#             outputText = response_body.get("results")[0].get("outputText")

#         response_metadata = response.get("ResponseMetadata", {})

#         if response_metadata.get("HTTPStatusCode", 500) >= 400:
#             raise BedrockError(
#                 message=outputText,
#                 status_code=response_metadata.get("HTTPStatusCode", 500),
#             )
#         else:
#             try:
#                 if (
#                     len(outputText) > 0
#                     and hasattr(model_response.choices[0], "message")
#                     and getattr(model_response.choices[0].message, "tool_calls", None)
#                     is None
#                 ):
#                     model_response.choices[0].message.content = outputText
#                 elif (
#                     hasattr(model_response.choices[0], "message")
#                     and getattr(model_response.choices[0].message, "tool_calls", None)
#                     is not None
#                 ):
#                     pass
#                 else:
#                     raise Exception()
#             except:
#                 raise BedrockError(
#                     message=json.dumps(outputText),
#                     status_code=response_metadata.get("HTTPStatusCode", 500),
#                 )

#         ## CALCULATING USAGE - bedrock charges on time, not tokens - have some mapping of cost here.
#         if not hasattr(model_response, "usage"):
#             setattr(model_response, "usage", Usage())
#         if getattr(model_response.usage, "total_tokens", None) is None:  # type: ignore
#             prompt_tokens = response_metadata.get(
#                 "x-amzn-bedrock-input-token-count", len(encoding.encode(prompt))
#             )
#             _text_response = model_response["choices"][0]["message"].get("content", "")
#             completion_tokens = response_metadata.get(
#                 "x-amzn-bedrock-output-token-count",
#                 len(
#                     encoding.encode(
#                         _text_response,
#                         disallowed_special=(),
#                     )
#                 ),
#             )
#             usage = Usage(
#                 prompt_tokens=prompt_tokens,
#                 completion_tokens=completion_tokens,
#                 total_tokens=prompt_tokens + completion_tokens,
#             )
#             setattr(model_response, "usage", usage)

#         model_response.created = int(time.time())
#         model_response.model = model

#         model_response._hidden_params["region_name"] = client.meta.region_name
#         print_verbose(f"model_response._hidden_params: {model_response._hidden_params}")
#         return model_response
#     except BedrockError as e:
#         exception_mapping_worked = True
#         raise e
#     except Exception as e:
#         if exception_mapping_worked:
#             raise e
#         else:
#             import traceback

#             raise BedrockError(status_code=500, message=traceback.format_exc())
