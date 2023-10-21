import openai
openai.api_base = "http://0.0.0.0:8000"
print("making request")
openai.api_key = "anything" # this gets passed as a header 


response = openai.ChatCompletion.create(
    model = "bedrock/anthropic.claude-instant-v1",
    messages = [
        {
            "role": "user",
            "content": "this is a test message, what model / llm are you"
        }
    ],
    aws_access_key_id="",
    aws_secret_access_key="",
    aws_region_name="us-west-2",
    max_tokens = 10,
)


print(response)


# response = openai.ChatCompletion.create(
#     model = "gpt-3.5-turbo",
#     messages = [
#         {
#             "role": "user",
#             "content": "this is a test message, what model / llm are you"
#         }
#     ],
#     max_tokens = 10,
#     stream=True
# )


# for chunk in response:
#     print(chunk)