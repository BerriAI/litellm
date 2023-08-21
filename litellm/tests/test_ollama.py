###### THESE TESTS CAN ONLY RUN LOCALLY WITH THE OLLAMA SERVER RUNNING ######
# import aiohttp
# import json
# import asyncio
# import requests

# async def get_ollama_response_stream(api_base="http://localhost:11434", model="llama2", prompt="Why is the sky blue?"):
#     session = aiohttp.ClientSession()
#     url = f'{api_base}/api/generate'
#     data = {
#         "model": model,
#         "prompt": prompt,
#     }

#     response = ""

#     try:
#         async with session.post(url, json=data) as resp:
#             async for line in resp.content.iter_any():
#                 if line:
#                     try:
#                         json_chunk = line.decode("utf-8")
#                         chunks = json_chunk.split("\n")
#                         for chunk in chunks:
#                             if chunk.strip() != "":
#                                 j = json.loads(chunk)
#                                 if "response" in j:
#                                     print(j["response"])
#                                     yield {
#                                         "role": "assistant",
#                                         "content": j["response"]
#                                     }
#                                     # self.responses.append(j["response"])
#                                     # yield "blank"
#                     except Exception as e:
#                         print(f"Error decoding JSON: {e}")
#     finally:
#         await session.close()

# # async def get_ollama_response_no_stream(api_base="http://localhost:11434", model="llama2", prompt="Why is the sky blue?"):
# #     generator =  get_ollama_response_stream(api_base="http://localhost:11434", model="llama2", prompt="Why is the sky blue?")
# #     response = ""
# #     async for elem in generator:
# #         print(elem)
# #         response += elem["content"]
# #     return response

# # #generator = get_ollama_response_stream()

# # result = asyncio.run(get_ollama_response_no_stream())
# # print(result)

# # # return this generator to the client for streaming requests


# # async def get_response():
# #     global generator
# #     async for elem in generator:
# #         print(elem)

# # asyncio.run(get_response())
