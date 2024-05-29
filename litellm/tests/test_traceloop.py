# Commented out for now - since traceloop break ci/cd
# import sys
# import os
# import io, asyncio

# sys.path.insert(0, os.path.abspath('../..'))

# from litellm import completion
# import litellm
# litellm.num_retries = 3
# litellm.success_callback = [""]
# import time
# import pytest
# from traceloop.sdk import Traceloop
# Traceloop.init(app_name="test-litellm", disable_batch=True)


# def test_traceloop_logging():
#     try:
#         litellm.set_verbose = True
#         response = litellm.completion(
#             model="gpt-3.5-turbo",
#             messages=[{"role": "user", "content":"This is a test"}],
#             max_tokens=1000,
#             temperature=0.7,
#             timeout=5,
#         )
#         print(f"response: {response}")
#     except Exception as e:
#         pytest.fail(f"An exception occurred - {e}")
# # test_traceloop_logging()


# # def test_traceloop_logging_async():
# #     try:
# #         litellm.set_verbose = True
# #         async def test_acompletion():
# #             return await litellm.acompletion(
# #                 model="gpt-3.5-turbo",
# #                 messages=[{"role": "user", "content":"This is a test"}],
# #                 max_tokens=1000,
# #                 temperature=0.7,
# #                 timeout=5,
# #             )
# #         response = asyncio.run(test_acompletion())
# #         print(f"response: {response}")
# #     except Exception as e:
# #         pytest.fail(f"An exception occurred - {e}")
# # test_traceloop_logging_async()
