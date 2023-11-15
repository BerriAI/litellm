import sys, os
import traceback
from dotenv import load_dotenv
import copy

load_dotenv()
sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import asyncio
from litellm import Router

async def call_acompletion(semaphore, router: Router, input_data):
    async with semaphore:
        # Replace 'input_data' with appropriate parameters for acompletion
        response = await router.acompletion(**input_data)
        # Handle the response as needed
        return response

async def main():
    # Initialize the Router
    model_list= [{
            "model_name": "gpt-3.5-turbo", 
            "litellm_params": {
                "model": "gpt-3.5-turbo", 
                "api_key": os.getenv("OPENAI_API_KEY"), 
            },
        }]
    router = Router(model_list=model_list, num_retries=3)

    # Create a semaphore with a capacity of 100
    semaphore = asyncio.Semaphore(100)

    # List to hold all task references
    tasks = []

    # Launch 1000 tasks
    for _ in range(1000):
        task = asyncio.create_task(call_acompletion(semaphore, router, {"model": "gpt-3.5-turbo", "messages": [{"role":"user", "content": "Hey, how's it going?"}]}))
        tasks.append(task)

    # Wait for all tasks to complete
    responses = await asyncio.gather(*tasks)
    # Process responses as needed

# Run the main function
asyncio.run(main())
