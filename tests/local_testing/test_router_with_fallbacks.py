# [LOCAL TEST] - runs against mock openai proxy
# # What this tests?
# ## This tests if fallbacks works for 429 errors

# import sys, os, time
# import traceback, asyncio
# import pytest

# sys.path.insert(
#     0, os.path.abspath("../..")
# )  # Adds the parent directory to the system path
# import litellm
# from litellm import Router

# model_list = [
#     {  # list of model deployments
#         "model_name": "text-embedding-ada-002",  # model alias
#         "litellm_params": {  # params for litellm completion/embedding call
#             "model": "text-embedding-ada-002",  # actual model name
#             "api_key": "sk-fakekey",
#             "api_base": "http://0.0.0.0:8080",
#         },
#         "tpm": 1000,
#         "rpm": 6,
#     },
#     {
#         "model_name": "text-embedding-ada-002-fallback",
#         "litellm_params": {  # params for litellm completion/embedding call
#             "model": "openai/text-embedding-ada-002-anything-else",  # actual model name
#             "api_key": "sk-fakekey2",
#             "api_base": "http://0.0.0.0:8080",
#         },
#         "tpm": 1000,
#         "rpm": 6,
#     },
# ]

# router = Router(
#     model_list=model_list,
#     fallbacks=[
#         {"text-embedding-ada-002": ["text-embedding-ada-002-fallback"]},
#         {"text-embedding-ada-002-fallback": ["text-embedding-ada-002"]},
#     ],
#     set_verbose=True,
#     num_retries=0,
#     debug_level="INFO",
#     routing_strategy="usage-based-routing",
# )


# def test_embedding_with_fallbacks():
#     response = router.embedding(model="text-embedding-ada-002", input=["Hello world"])
#     print(f"response: {response}")


# test_embedding_with_fallbacks()
