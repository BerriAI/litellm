import os
from openai import OpenAI
from dotenv import load_dotenv
import concurrent.futures

load_dotenv()

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)


def create_chat_completion():
    return client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Say this is a test. Respond in 20 lines",
            }
        ],
        model="gpt-3.5-turbo",
    )


with concurrent.futures.ThreadPoolExecutor() as executor:
    # Set a timeout of 10 seconds
    future = executor.submit(create_chat_completion)
    try:
        chat_completion = future.result(timeout=0.00001)
        print(chat_completion)
    except concurrent.futures.TimeoutError:
        print("Operation timed out.")
