import time
import asyncio
import os
from openai import AsyncOpenAI, AsyncAzureOpenAI
import uuid
import traceback
from large_text import text
from dotenv import load_dotenv
from statistics import mean, median

litellm_client = AsyncOpenAI(base_url="http://0.0.0.0:4000/", api_key="sk-1234")


async def litellm_completion():
    try:
        start_time = time.time()
        response = await litellm_client.chat.completions.create(
            model="fake-openai-endpoint",
            messages=[
                {
                    "role": "user",
                    "content": f"This is a test{uuid.uuid4()}",
                }
            ],
            user="my-new-end-user-1",
        )
        end_time = time.time()
        latency = end_time - start_time
        print("response time=", latency)
        return response, latency

    except Exception as e:
        with open("error_log.txt", "a") as error_log:
            error_log.write(f"Error during completion: {str(e)}\n")
        return None, 0


async def main():
    latencies = []
    for i in range(5):
        start = time.time()
        n = 100  # Number of concurrent tasks
        tasks = [litellm_completion() for _ in range(n)]

        chat_completions = await asyncio.gather(*tasks)

        successful_completions = [c for c, l in chat_completions if c is not None]
        completion_latencies = [l for c, l in chat_completions if c is not None]
        latencies.extend(completion_latencies)

        with open("error_log.txt", "a") as error_log:
            for completion, latency in chat_completions:
                if isinstance(completion, str):
                    error_log.write(completion + "\n")

        print(n, time.time() - start, len(successful_completions))

    if latencies:
        average_latency = mean(latencies)
        median_latency = median(latencies)
        print(f"Average Latency per Response: {average_latency} seconds")
        print(f"Median Latency per Response: {median_latency} seconds")


if __name__ == "__main__":
    open("error_log.txt", "w").close()

    asyncio.run(main())
