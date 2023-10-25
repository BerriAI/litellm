import openai, os, dotenv, traceback, time
openai.api_base = "http://0.0.0.0:8000"
dotenv.load_dotenv()
openai.api_key = os.getenv("ANTHROPIC_API_KEY") # this gets passed as a header 


response1 = openai.ChatCompletion.create(
    model = "claude-instant-1",
    messages = [
        {
            "role": "user",
            "content": "write a short poem about litellm"
        }
    ],
)

try:
    print(f"response: {response1['choices'][0]['message']['content']}")
except:
    print(f"response: {response1}")

time.sleep(1) # allow time for request to be stored 

response2 = openai.ChatCompletion.create(
    model = "claude-instant-1",
    messages = [
        {
            "role": "user",
            "content": "write a short poem about litellm"
        }
    ],
)

try:
    print(f"response: {response2['choices'][0]['message']['content']}")
except:
    print(f"response: {response2}")

openai.api_key = os.getenv("OPENAI_API_KEY")

try: 
    response3 = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [
            {
                "role": "user",
                "content": "write a short poem about litellm"
            }
        ],
    )
except Exception as e: 
    traceback.print_exc()

try:
    print(f"response: {response3['choices'][0]['message']['content']}")
except:
    print(f"response: {response3}")

openai.api_key = os.getenv("ANTHROPIC_API_KEY") # this gets passed as a header 
# switch caching off using cache flag
response4 = openai.ChatCompletion.create(
    model = "claude-instant-1",
    messages = [
        {
            "role": "user",
            "content": "write a short poem about litellm"
        }
    ],
    caching = False,
)

try:
    print(f"response: {response4['choices'][0]['message']['content']}")
except:
    print(f"response: {response4}")

assert response1["choices"][0]["message"]["content"] == response2["choices"][0]["message"]["content"] 
assert response1["choices"][0]["message"]["content"] != response4["choices"][0]["message"]["content"] 

assert response1["choices"][0]["message"]["content"] != response3["choices"][0]["message"]["content"] 