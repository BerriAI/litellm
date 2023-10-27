# import sys
# import os
# import io
# #
# sys.path.insert(0, os.path.abspath('../..'))
# import litellm
# from litellm import completion
# from traceloop.sdk import Traceloop
# Traceloop.init(app_name="test_traceloop", disable_batch=True, traceloop_sync_enabled=False)
# litellm.success_callback = ["traceloop"]


# def test_traceloop_logging():
#     try:
#         print('making completion call')
#         response = completion(
#             model="claude-instant-1.2",
#             messages=[
#                 {"role": "user", "content": "Tell me a joke about OpenTelemetry"}
#             ],
#             max_tokens=10,
#             temperature=0.2,
#         )
#         print(response)
#     except Exception as e:
#         print(e)


# # test_traceloop_logging()


# def test_traceloop_tracing_function_calling():
#     function1 = [
#         {
#             "name": "get_current_weather",
#             "description": "Get the current weather in a given location",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "location": {
#                         "type": "string",
#                         "description": "The city and state, e.g. San Francisco, CA",
#                     },
#                     "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
#                 },
#                 "required": ["location"],
#             },
#         }
#     ]
#     try:
#         response = completion(
#             model="gpt-3.5-turbo",
#             messages=[{"role": "user", "content": "what's the weather in boston"}],
#             temperature=0.1,
#             functions=function1,
#         )
#         print(response)
#     except Exception as e:
#         print(e)


# # test_traceloop_tracing_function_calling()
