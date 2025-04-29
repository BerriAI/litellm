# import openai, json, time, asyncio
# client = openai.AsyncOpenAI(
#     api_key="sk-1234",
#     base_url="http://0.0.0.0:8000"
# )

# super_fake_messages = [
#   {
#     "role": "user",
#     "content": f"What's the weather like in San Francisco, Tokyo, and Paris? {time.time()}"
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

# async def chat_completions():
#     super_fake_response = await client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=super_fake_messages,
#         seed=1337,
#         stream=False
#     )  # get a new response from the model where it can see the function response
#     await asyncio.sleep(1)
#     return super_fake_response

# async def loadtest_fn(n = 1):
#     global num_task_cancelled_errors, exception_counts, chat_completions
#     start = time.time()
#     tasks = [chat_completions() for _ in range(n)]
#     chat_completions = await asyncio.gather(*tasks)
#     successful_completions = [c for c in chat_completions if c is not None]
#     print(n, time.time() - start, len(successful_completions))

# # print(json.dumps(super_fake_response.model_dump(), indent=4))

# asyncio.run(loadtest_fn())
