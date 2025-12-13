import os
import time
from openai import OpenAI
import httpx

# Wait for proxy to start
print("Waiting for proxy to start...")
# simple loop to check if port 4000 is open
for i in range(30):
    try:
        httpx.get("http://0.0.0.0:4000/health/liveness")
        print("Proxy is up!")
        break
    except:
        time.sleep(1)
        print("Waiting for proxy...")

client = OpenAI(
    api_key="sk-1234", 
    base_url="http://0.0.0.0:4000"
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit a file in the workspace",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"}
                },
                "required": ["path", "old_string", "new_string"]
            }
        }
    }
]

messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Please edit the file /workspace/test.txt. Change 'hello' to 'world'."}
]

print("Sending request to proxy...")
try:
    response = client.chat.completions.create(
        model="anthropic/claude-3-opus-20240229",
        messages=messages,
        tools=tools,
        tool_choice="auto" 
    )
    
    print("Response received:")
    # print full response for debugging
    print(response.model_dump_json(indent=2))
    
    if response.choices[0].message.tool_calls:
        print("Success! Tool calls found.")
        for tc in response.choices[0].message.tool_calls:
            print(f"Tool: {tc.function.name}")
            print(f"Args: {tc.function.arguments}")
    else:
        print("No tool calls found in response.")

except Exception as e:
    print(f"Error: {e}")
