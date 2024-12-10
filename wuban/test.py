from litellm import completion
import os

if __name__ == "__main__":
    os.environ['DEEPSEEK_API_KEY'] = "sk-354c3ec4f73542f681d15f2369d9fa95"
    response = completion(
    model="deepseek/deepseek-chat",
    messages=[
       {"role": "user", "content": "hello from litellm"}
   ],
    )
    print(response)