# # This tests the litelm proxy
# # it makes async Completion requests with streaming
# import openai

# openai.base_url = "http://0.0.0.0:8000"
# openai.api_key = "temp-key"
# print(openai.base_url)

# async def test_async_completion():
#     response = await (
#         model="gpt-3.5-turbo",
#         prompt='this is a test request, write a short poem',
#     )
#     print(response)

#     print("test_streaming")
#     response = await openai.chat.completions.create(
#         model="gpt-3.5-turbo",
#         prompt='this is a test request, write a short poem',
#         stream=True
#     )
#     print(response)
#     async for chunk in response:
#         print(chunk)


# import asyncio
# asyncio.run(test_async_completion())
