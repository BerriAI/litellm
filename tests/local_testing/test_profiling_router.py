# #### What this tests ####
# #    This profiles a router call to find where calls are taking the most time.

# import sys, os, time, logging
# import traceback, asyncio, uuid
# import pytest
# import cProfile
# from pstats import Stats
# sys.path.insert(
#     0, os.path.abspath("../..")
# )  # Adds the parent directory to the system path
# import litellm
# from litellm import Router
# from concurrent.futures import ThreadPoolExecutor
# from dotenv import load_dotenv
# from aiodebug import log_slow_callbacks  # Import the aiodebug utility for logging slow callbacks

# # litellm.telemetry = False

# load_dotenv()

# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s %(levelname)s: %(message)s',
#     datefmt='%I:%M:%S %p',
#     filename='aiologs.log',   # Name of the log file where logs will be written
#     filemode='w'              # 'w' to overwrite the log file on each run, use 'a' to append
# )

# # Dictionary to store exception types and their counts
# exception_counts = {}
# exception_data = []

# litellm.telemetry = False

# num_task_cancelled_errors = 0

# model_list = [{
#     "model_name": "azure-model",
#     "litellm_params": {
#         "model": "azure/gpt-turbo",
#         "api_key": "os.environ/AZURE_FRANCE_API_KEY",
#         "api_base": "https://openai-france-1234.openai.azure.com",
#         "rpm": 1440,
#     }
# }, {
#     "model_name": "azure-model",
#     "litellm_params": {
#         "model": "azure/gpt-35-turbo",
#         "api_key": "os.environ/AZURE_EUROPE_API_KEY",
#         "api_base": "https://my-endpoint-europe-berri-992.openai.azure.com",
#         "rpm": 6
#     }
# }, {
#     "model_name": "azure-model",
#     "litellm_params": {
#         "model": "azure/gpt-35-turbo",
#         "api_key": "os.environ/AZURE_CANADA_API_KEY",
#         "api_base": "https://my-endpoint-canada-berri992.openai.azure.com",
#         "rpm": 6
#     }
# }]

# router = Router(model_list=model_list, set_verbose=False, num_retries=3)

# async def router_completion():
#     global num_task_cancelled_errors, exception_counts
#     try:
#         messages=[{"role": "user", "content": f"This is a test: {uuid.uuid4()}"}]
#         response = await router.acompletion(model="azure-model", messages=messages)
#         return response
#     except asyncio.exceptions.CancelledError:
#         exception_type = "CancelledError"
#         exception_counts[exception_type] = exception_counts.get(exception_type, 0) + 1
#         print("Task was cancelled")
#         num_task_cancelled_errors += 1
#         exception_data.append({
#             "type": exception_type,
#             "traceback": None
#         })
#         return None
#     except Exception as e:
#         exception_type = type(e).__name__
#         exception_counts[exception_type] = exception_counts.get(exception_type, 0) + 1
#         exception_data.append({
#             "type": exception_type,
#             "traceback": traceback.format_exc()
#         })
#         return None

# async def loadtest_fn(n = 1452):
#     global num_task_cancelled_errors, exception_counts
#     start = time.time()
#     tasks = [router_completion() for _ in range(n)]
#     chat_completions = await asyncio.gather(*tasks)
#     successful_completions = [c for c in chat_completions if c is not None]
#     print(n, time.time() - start, len(successful_completions))

#     # Print exception breakdown
#     print("Exception Breakdown:")
#     for exception_type, count in exception_counts.items():
#         print(f"{exception_type}: {count}")

#     # Store exception_data in a file
#     with open('exception_data.txt', 'w') as file:
#         for data in exception_data:
#             file.write(f"Type: {data['type']}\n")
#             if data['traceback']:
#                 file.write(f"Traceback:\n{data['traceback']}\n\n")

# loop = asyncio.get_event_loop()
# loop.set_debug(True)
# log_slow_callbacks.enable(0.05)  # Log callbacks slower than 0.05 seconds

# # Excute the load testing function within the asyncio event loop
# loop.run_until_complete(loadtest_fn())

# # ### SUSTAINED LOAD TESTS ###
# # import time, asyncio
# # async def make_requests(n):
# #     tasks = [router_completion() for _ in range(n)]
# #     print(f"num tasks: {len(tasks)}")
# #     chat_completions = await asyncio.gather(*tasks)
# #     successful_completions = [c for c in chat_completions if c is not None]
# #     print(f"successful_completions: {len(successful_completions)}")
# #     return successful_completions

# # async def main():
# #   start_time = time.time()
# #   total_successful_requests = 0
# #   request_limit = 1000
# #   batches = 2  # batches of 1k requests
# #   start = time.time()
# #   tasks = []  # list to hold all tasks

# #   async def request_loop():
# #     nonlocal tasks
# #     for _ in range(batches):
# #         # Make 1,000 requests
# #         task = asyncio.create_task(make_requests(request_limit))
# #         tasks.append(task)

# #         # Introduce a delay to achieve 1,000 requests per second
# #         await asyncio.sleep(1)

# #   await request_loop()
# #   results = await asyncio.gather(*tasks)
# #   total_successful_requests = sum(len(res) for res in results)

# #   print(request_limit*batches, time.time() - start, total_successful_requests)

# # asyncio.run(main())
