# #### What this tests ####

# from memory_profiler import profile, memory_usage
# import sys, os, time
# import traceback, asyncio
# import pytest

# sys.path.insert(
#     0, os.path.abspath("../..")
# )  # Adds the parent directory to the system path
# import litellm
# from litellm import Router
# from concurrent.futures import ThreadPoolExecutor
# from collections import defaultdict
# from dotenv import load_dotenv
# from litellm._uuid import uuid
# import tracemalloc
# import objgraph

# objgraph.growth(shortnames=True)
# objgraph.show_most_common_types(limit=10)

# from mem_top import mem_top

# load_dotenv()


# model_list = [
#     {
#         "model_name": "gpt-3.5-turbo",  # openai model name
#         "litellm_params": {  # params for litellm completion/embedding call
#             "model": "azure/chatgpt-v-3",
#             "api_key": os.getenv("AZURE_API_KEY"),
#             "api_version": os.getenv("AZURE_API_VERSION"),
#             "api_base": os.getenv("AZURE_API_BASE"),
#         },
#         "tpm": 240000,
#         "rpm": 1800,
#     },
#     {
#         "model_name": "bad-model",  # openai model name
#         "litellm_params": {  # params for litellm completion/embedding call
#             "model": "azure/chatgpt-v-3",
#             "api_key": "bad-key",
#             "api_version": os.getenv("AZURE_API_VERSION"),
#             "api_base": os.getenv("AZURE_API_BASE"),
#         },
#         "tpm": 240000,
#         "rpm": 1800,
#     },
#     {
#         "model_name": "text-embedding-ada-002",
#         "litellm_params": {
#             "model": "azure/azure-embedding-model",
#             "api_key": os.environ["AZURE_API_KEY"],
#             "api_base": os.environ["AZURE_API_BASE"],
#         },
#         "tpm": 100000,
#         "rpm": 10000,
#     },
# ]
# litellm.set_verbose = True
# litellm.cache = litellm.Cache(
#     type="s3", s3_bucket_name="litellm-my-test-bucket-2", s3_region_name="us-east-1"
# )
# router = Router(
#     model_list=model_list,
#     fallbacks=[
#         {"bad-model": ["gpt-3.5-turbo"]},
#     ],
# )  # type: ignore


# async def router_acompletion():
#     # embedding call
#     question = f"This is a test: {uuid.uuid4()}" * 1

#     response = await router.acompletion(
#         model="bad-model", messages=[{"role": "user", "content": question}]
#     )
#     print("completion-resp", response)
#     return response


# async def main():
#     for i in range(1):
#         start = time.time()
#         n = 15  # Number of concurrent tasks
#         tasks = [router_acompletion() for _ in range(n)]

#         chat_completions = await asyncio.gather(*tasks)

#         successful_completions = [c for c in chat_completions if c is not None]

#         # Write errors to error_log.txt
#         with open("error_log.txt", "a") as error_log:
#             for completion in chat_completions:
#                 if isinstance(completion, str):
#                     error_log.write(completion + "\n")

#         print(n, time.time() - start, len(successful_completions))
#     print()
#     print(vars(router))
#     prev_models = router.previous_models

#     print("vars in prev_models")
#     print(prev_models[0].keys())


# if __name__ == "__main__":
#     # Blank out contents of error_log.txt
#     open("error_log.txt", "w").close()

#     import tracemalloc

#     tracemalloc.start(25)

#     # ... run your application ...

#     asyncio.run(main())
#     print(mem_top())

#     snapshot = tracemalloc.take_snapshot()
#     # top_stats = snapshot.statistics('lineno')

#     # print("[ Top 10 ]")
#     # for stat in top_stats[:50]:
#     #     print(stat)

#     top_stats = snapshot.statistics("traceback")

#     # pick the biggest memory block
#     stat = top_stats[0]
#     print("%s memory blocks: %.1f KiB" % (stat.count, stat.size / 1024))
#     for line in stat.traceback.format():
#         print(line)
#     print()
#     stat = top_stats[1]
#     print("%s memory blocks: %.1f KiB" % (stat.count, stat.size / 1024))
#     for line in stat.traceback.format():
#         print(line)

#     print()
#     stat = top_stats[2]
#     print("%s memory blocks: %.1f KiB" % (stat.count, stat.size / 1024))
#     for line in stat.traceback.format():
#         print(line)
#     print()

#     stat = top_stats[3]
#     print("%s memory blocks: %.1f KiB" % (stat.count, stat.size / 1024))
#     for line in stat.traceback.format():
#         print(line)
