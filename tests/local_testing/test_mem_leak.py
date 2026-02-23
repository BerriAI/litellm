# import io
# import os
# import sys

# sys.path.insert(0, os.path.abspath("../.."))

# import litellm
# from memory_profiler import profile
# from litellm.utils import (
#     ModelResponseIterator,
#     ModelResponseListIterator,
#     CustomStreamWrapper,
# )
# from litellm.types.utils import ModelResponse, Choices, Message
# import time
# import pytest


# # @app.post("/debug")
# # async def debug(body: ExampleRequest) -> str:
# #     return await main_logic(body.query)
# def model_response_list_factory():
#     chunks = [
#         {
#             "id": "chatcmpl-9SQxdH5hODqkWyJopWlaVOOUnFwlj",
#             "choices": [
#                 {
#                     "delta": {"content": "", "role": "assistant"},
#                     "finish_reason": None,
#                     "index": 0,
#                 }
#             ],
#             "created": 1716563849,
#             "model": "gpt-4o-2024-05-13",
#             "object": "chat.completion.chunk",
#             "system_fingerprint": "fp_5f4bad809a",
#         },
#         {
#             "id": "chatcmpl-9SQxdH5hODqkWyJopWlaVOOUnFwlj",
#             "choices": [
#                 {"delta": {"content": "This"}, "finish_reason": None, "index": 0}
#             ],
#             "created": 1716563849,
#             "model": "gpt-4o-2024-05-13",
#             "object": "chat.completion.chunk",
#             "system_fingerprint": "fp_5f4bad809a",
#         },
#         {
#             "id": "chatcmpl-9SQxdH5hODqkWyJopWlaVOOUnFwlj",
#             "choices": [
#                 {"delta": {"content": " is"}, "finish_reason": None, "index": 0}
#             ],
#             "created": 1716563849,
#             "model": "gpt-4o-2024-05-13",
#             "object": "chat.completion.chunk",
#             "system_fingerprint": "fp_5f4bad809a",
#         },
#         {
#             "id": "chatcmpl-9SQxdH5hODqkWyJopWlaVOOUnFwlj",
#             "choices": [
#                 {"delta": {"content": " a"}, "finish_reason": None, "index": 0}
#             ],
#             "created": 1716563849,
#             "model": "gpt-4o-2024-05-13",
#             "object": "chat.completion.chunk",
#             "system_fingerprint": "fp_5f4bad809a",
#         },
#         {
#             "id": "chatcmpl-9SQxdH5hODqkWyJopWlaVOOUnFwlj",
#             "choices": [
#                 {"delta": {"content": " dummy"}, "finish_reason": None, "index": 0}
#             ],
#             "created": 1716563849,
#             "model": "gpt-4o-2024-05-13",
#             "object": "chat.completion.chunk",
#             "system_fingerprint": "fp_5f4bad809a",
#         },
#         {
#             "id": "chatcmpl-9SQxdH5hODqkWyJopWlaVOOUnFwlj",
#             "choices": [
#                 {
#                     "delta": {"content": " response"},
#                     "finish_reason": None,
#                     "index": 0,
#                 }
#             ],
#             "created": 1716563849,
#             "model": "gpt-4o-2024-05-13",
#             "object": "chat.completion.chunk",
#             "system_fingerprint": "fp_5f4bad809a",
#         },
#         {
#             "id": "",
#             "choices": [
#                 {
#                     "finish_reason": None,
#                     "index": 0,
#                     "content_filter_offsets": {
#                         "check_offset": 35159,
#                         "start_offset": 35159,
#                         "end_offset": 36150,
#                     },
#                     "content_filter_results": {
#                         "hate": {"filtered": False, "severity": "safe"},
#                         "self_harm": {"filtered": False, "severity": "safe"},
#                         "sexual": {"filtered": False, "severity": "safe"},
#                         "violence": {"filtered": False, "severity": "safe"},
#                     },
#                 }
#             ],
#             "created": 0,
#             "model": "",
#             "object": "",
#         },
#         {
#             "id": "chatcmpl-9SQxdH5hODqkWyJopWlaVOOUnFwlj",
#             "choices": [{"delta": {"content": "."}, "finish_reason": None, "index": 0}],
#             "created": 1716563849,
#             "model": "gpt-4o-2024-05-13",
#             "object": "chat.completion.chunk",
#             "system_fingerprint": "fp_5f4bad809a",
#         },
#         {
#             "id": "chatcmpl-9SQxdH5hODqkWyJopWlaVOOUnFwlj",
#             "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
#             "created": 1716563849,
#             "model": "gpt-4o-2024-05-13",
#             "object": "chat.completion.chunk",
#             "system_fingerprint": "fp_5f4bad809a",
#         },
#         {
#             "id": "",
#             "choices": [
#                 {
#                     "finish_reason": None,
#                     "index": 0,
#                     "content_filter_offsets": {
#                         "check_offset": 36150,
#                         "start_offset": 36060,
#                         "end_offset": 37029,
#                     },
#                     "content_filter_results": {
#                         "hate": {"filtered": False, "severity": "safe"},
#                         "self_harm": {"filtered": False, "severity": "safe"},
#                         "sexual": {"filtered": False, "severity": "safe"},
#                         "violence": {"filtered": False, "severity": "safe"},
#                     },
#                 }
#             ],
#             "created": 0,
#             "model": "",
#             "object": "",
#         },
#     ]

#     chunk_list = []
#     for chunk in chunks:
#         new_chunk = litellm.ModelResponse(stream=True, id=chunk["id"])
#         if "choices" in chunk and isinstance(chunk["choices"], list):
#             new_choices = []
#             for choice in chunk["choices"]:
#                 if isinstance(choice, litellm.utils.StreamingChoices):
#                     _new_choice = choice
#                 elif isinstance(choice, dict):
#                     _new_choice = litellm.utils.StreamingChoices(**choice)
#                 new_choices.append(_new_choice)
#             new_chunk.choices = new_choices
#         chunk_list.append(new_chunk)

#     return ModelResponseListIterator(model_responses=chunk_list)


# async def mock_completion(*args, **kwargs):
#     completion_stream = model_response_list_factory()
#     return litellm.CustomStreamWrapper(
#         completion_stream=completion_stream,
#         model="gpt-4-0613",
#         custom_llm_provider="cached_response",
#         logging_obj=litellm.Logging(
#             model="gpt-4-0613",
#             messages=[{"role": "user", "content": "Hey"}],
#             stream=True,
#             call_type="completion",
#             start_time=time.time(),
#             litellm_call_id="12345",
#             function_id="1245",
#         ),
#     )


# @profile
# async def main_logic() -> str:
#     stream = await mock_completion()
#     result = ""
#     async for chunk in stream:
#         result += chunk.choices[0].delta.content or ""
#     return result


# import asyncio

# for _ in range(100):
#     asyncio.run(main_logic())


# # @pytest.mark.asyncio
# # def test_memory_profile(capsys):
# #     # Run the async function
# #     result = asyncio.run(main_logic())

# #     # Verify the result
# #     assert result == "This is a dummy response."

# #     # Capture the output
# #     captured = capsys.readouterr()

# #     # Print memory output for debugging
# #     print("Memory Profiler Output:")
# #     print(f"captured out: {captured.out}")

# #     # Basic memory leak checks
# #     for idx, line in enumerate(captured.out.split("\n")):
# #         if idx % 2 == 0 and "MiB" in line:
# #             print(f"line: {line}")

# #     # mem_lines = [line for line in captured.out.split("\n") if "MiB" in line]

# #     print(mem_lines)

# #     # Ensure we have some memory lines
# #     assert len(mem_lines) > 0, "No memory profiler output found"

# #     # Optional: Add more specific memory leak detection
# #     for line in mem_lines:
# #         # Extract memory increment
# #         parts = line.split()
# #         if len(parts) >= 3:
# #             try:
# #                 mem_increment = float(parts[2].replace("MiB", ""))
# #                 # Assert that memory increment is below a reasonable threshold
# #                 assert mem_increment < 1.0, f"Potential memory leak detected: {line}"
# #             except (ValueError, IndexError):
# #                 pass  # Skip lines that don't match expected format
