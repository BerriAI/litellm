# import openai, json
# client = openai.OpenAI(
#     api_key="sk-1234",
#     base_url="http://0.0.0.0:8000"
# )

# super_fake_messages = [
#   {
#     "role": "user",
#     "content": "What's the weather like in San Francisco, Tokyo, and Paris?"
#   },
#   {
#     "content": None,
#     "role": "assistant",
#     "tool_calls": [
#       {
#         "id": "1",
#         "function": {
#           "arguments": "{\"location\": \"San Francisco\", \"unit\": \"celsius\"}",
#           "name": "get_current_weather"
#         },
#         "type": "function"
#       },
#       {
#         "id": "2",
#         "function": {
#           "arguments": "{\"location\": \"Tokyo\", \"unit\": \"celsius\"}",
#           "name": "get_current_weather"
#         },
#         "type": "function"
#       },
#       {
#         "id": "3",
#         "function": {
#           "arguments": "{\"location\": \"Paris\", \"unit\": \"celsius\"}",
#           "name": "get_current_weather"
#         },
#         "type": "function"
#       }
#     ]
#   },
#   {
#     "tool_call_id": "1",
#     "role": "tool",
#     "name": "get_current_weather",
#     "content": "{\"location\": \"San Francisco\", \"temperature\": \"90\", \"unit\": \"celsius\"}"
#   },
#   {
#     "tool_call_id": "2",
#     "role": "tool",
#     "name": "get_current_weather",
#     "content": "{\"location\": \"Tokyo\", \"temperature\": \"30\", \"unit\": \"celsius\"}"
#   },
#   {
#     "tool_call_id": "3",
#     "role": "tool",
#     "name": "get_current_weather",
#     "content": "{\"location\": \"Paris\", \"temperature\": \"50\", \"unit\": \"celsius\"}"
#   }
# ]

# super_fake_response = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=super_fake_messages,
#     seed=1337,
#     stream=False
# )  # get a new response from the model where it can see the function response

# print(json.dumps(super_fake_response.model_dump(), indent=4))