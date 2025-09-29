# import sys, os
# import traceback
# from dotenv import load_dotenv
# import copy

# load_dotenv()
# sys.path.insert(
#     0, os.path.abspath("../..")
# )  # Adds the parent directory to the system path
# import asyncio
# from litellm import Router, Timeout
# import time


# async def call_acompletion(semaphore, router: Router, input_data):
#     async with semaphore:
#         try:
#             # Use asyncio.wait_for to set a timeout for the task
#             response = await router.acompletion(**input_data)
#             # Handle the response as needed
#             print(response)
#             return response
#         except Timeout:
#             print(f"Task timed out: {input_data}")
#             return None  # You may choose to return something else or raise an exception


# async def main():
#     # Initialize the Router
#     model_list = [
#         {
#             "model_name": "gpt-3.5-turbo",
#             "litellm_params": {
#                 "model": "gpt-3.5-turbo",
#                 "api_key": os.getenv("OPENAI_API_KEY"),
#             },
#         },
#         {
#             "model_name": "gpt-3.5-turbo",
#             "litellm_params": {
#                 "model": "azure/gpt-4.1-nano",
#                 "api_key": os.getenv("AZURE_API_KEY"),
#                 "api_base": os.getenv("AZURE_API_BASE"),
#                 "api_version": os.getenv("AZURE_API_VERSION"),
#             },
#         },
#     ]
#     router = Router(model_list=model_list, num_retries=3, timeout=10)

#     # Create a semaphore with a capacity of 100
#     semaphore = asyncio.Semaphore(100)

#     # List to hold all task references
#     tasks = []
#     start_time_all_tasks = time.time()
#     # Launch 1000 tasks
#     for _ in range(500):
#         task = asyncio.create_task(
#             call_acompletion(
#                 semaphore,
#                 router,
#                 {
#                     "model": "gpt-3.5-turbo",
#                     "messages": [{"role": "user", "content": "Hey, how's it going?"}],
#                 },
#             )
#         )
#         tasks.append(task)

#     # Wait for all tasks to complete
#     responses = await asyncio.gather(*tasks)
#     # Process responses as needed
#     # Record the end time for all tasks
#     end_time_all_tasks = time.time()
#     # Calculate the total time for all tasks
#     total_time_all_tasks = end_time_all_tasks - start_time_all_tasks
#     print(f"Total time for all tasks: {total_time_all_tasks} seconds")

#     # Calculate the average time per response
#     average_time_per_response = total_time_all_tasks / len(responses)
#     print(f"Average time per response: {average_time_per_response} seconds")
#     print(f"NUMBER OF COMPLETED TASKS: {len(responses)}")


# # Run the main function
# asyncio.run(main())
