# import openai
# client = openai.OpenAI(
#     api_key="anything",
#     base_url="http://0.0.0.0:8000"
# )

# # request sent to model set on litellm proxy, `litellm --model`
# response = client.chat.completions.create(
#     model="azure/chatgpt-v-2",
#     messages = [
#         {
#             "role": "user",
#             "content": "this is a test request, write a short poem"
#         }
#     ],
#     extra_body={
#         "metadata": {
#             "generation_name": "ishaan-generation-openai-client",
#             "generation_id": "openai-client-gen-id22",
#             "trace_id": "openai-client-trace-id22",
#             "trace_user_id": "openai-client-user-id2"
#         }
#     }
# )

# print(response)
