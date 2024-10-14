import sys, os
import traceback
from dotenv import load_dotenv
import copy

load_dotenv()
sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import asyncio
from litellm import Router, Timeout
import time
from litellm.caching.caching import Cache
import litellm
import openai

### Test just calling AsyncAzureOpenAI

openai_client = openai.AsyncAzureOpenAI(
    azure_endpoint=os.getenv("AZURE_API_BASE"),
    api_key=os.getenv("AZURE_API_KEY"),
)


async def call_acompletion(semaphore, input_data):
    async with semaphore:
        try:
            # Use asyncio.wait_for to set a timeout for the task
            response = await openai_client.chat.completions.create(**input_data)
            # Handle the response as needed
            print(response)
            return response
        except Timeout:
            print(f"Task timed out: {input_data}")
            return None  # You may choose to return something else or raise an exception


async def main():
    # Initialize the Router

    # Create a semaphore with a capacity of 100
    semaphore = asyncio.Semaphore(100)

    # List to hold all task references
    tasks = []
    start_time_all_tasks = time.time()
    # Launch 1000 tasks
    for _ in range(500):
        task = asyncio.create_task(
            call_acompletion(
                semaphore,
                {
                    "model": "chatgpt-v-2",
                    "messages": [{"role": "user", "content": "Hey, how's it going?"}],
                },
            )
        )
        tasks.append(task)

    # Wait for all tasks to complete
    responses = await asyncio.gather(*tasks)
    # Process responses as needed
    # Record the end time for all tasks
    end_time_all_tasks = time.time()
    # Calculate the total time for all tasks
    total_time_all_tasks = end_time_all_tasks - start_time_all_tasks
    print(f"Total time for all tasks: {total_time_all_tasks} seconds")

    # Calculate the average time per response
    average_time_per_response = total_time_all_tasks / len(responses)
    print(f"Average time per response: {average_time_per_response} seconds")
    print(f"NUMBER OF COMPLETED TASKS: {len(responses)}")


# Run the main function
asyncio.run(main())
