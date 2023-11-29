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

# load_dotenv()

# logging.basicConfig(
#     level=logging.DEBUG,
#     format='%(asctime)s %(levelname)s: %(message)s',
#     datefmt='%I:%M:%S %p',
#     filename='aiologs.log',   # Name of the log file where logs will be written
#     filemode='w'              # 'w' to overwrite the log file on each run, use 'a' to append
# )


# model_list = [{
#     "model_name": "azure-model",
#     "litellm_params": {
#         "model": "azure/gpt-turbo",
#         "api_key": "os.environ/AZURE_FRANCE_API_KEY",
#         "api_base": "https://openai-france-1234.openai.azure.com"
#     }
# }, {
#     "model_name": "azure-model",
#     "litellm_params": {
#         "model": "azure/gpt-35-turbo",
#         "api_key": "os.environ/AZURE_EUROPE_API_KEY",
#         "api_base": "https://my-endpoint-europe-berri-992.openai.azure.com"
#     }
# }, {
#     "model_name": "azure-model",
#     "litellm_params": {
#         "model": "azure/gpt-35-turbo",
#         "api_key": "os.environ/AZURE_CANADA_API_KEY",
#         "api_base": "https://my-endpoint-canada-berri992.openai.azure.com"
#     }
# }]

# router = Router(model_list=model_list, set_verbose=False, num_retries=3)

# async def router_completion(): 
#     try: 
#         messages=[{"role": "user", "content": f"This is a test: {uuid.uuid4()}"}]
#         response = await router.acompletion(model="azure-model", messages=messages)
#         return response
#     except Exception as e: 
#         print(e, file=sys.stderr)
#         traceback.print_exc()
#         return None

# async def loadtest_fn():
#     start = time.time()
#     n = 1000
#     tasks = [router_completion() for _ in range(n)]
#     chat_completions = await asyncio.gather(*tasks)
#     successful_completions = [c for c in chat_completions if c is not None]
#     print(n, time.time() - start, len(successful_completions))

# loop = asyncio.get_event_loop()
# loop.set_debug(True)
# log_slow_callbacks.enable(0.05)  # Log callbacks slower than 0.05 seconds

# # Excute the load testing function within the asyncio event loop
# loop.run_until_complete(loadtest_fn())