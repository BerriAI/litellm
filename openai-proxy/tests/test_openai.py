import openai
openai.api_base = "http://127.0.0.1:8000"
openai.api_key = "this can be anything"

print("making request")

api_key = "" 
response = openai.ChatCompletion.create(
    model = "gpt-3.5-turbo",
    messages = [
        {
            "role": "user",
            "content": "this is a test message, what model / llm are you"
        }
    ],
    api_key=api_key,
    max_tokens = 10,
)


print(response)


response = openai.ChatCompletion.create(
    model = "gpt-3.5-turbo",
    messages = [
        {
            "role": "user",
            "content": "this is a test message, what model / llm are you"
        }
    ],
    api_key=api_key,
    max_tokens = 10,
    stream=True
)


for chunk in response:
    print(chunk)