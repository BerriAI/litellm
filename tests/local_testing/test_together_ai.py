# import sys, os
# import traceback

# sys.path.insert(
#     0, os.path.abspath("../..")
# )  # Adds the parent directory to the system path
# import time
# import litellm
# import openai
# import pytest

# ### Together AI
# import together
# together.api_key = ""

# sample_message = [
#     {"role": "user", "content": "Who are you"},
#     {"role": "assistant", "content": "I am your helpful assistant."},
#     {"role": "user", "content": "Tell me a joke"},
# ]


# def format_prompt_togetherai(messages, prompt_format, stop_words):
#     start_token, end_token = prompt_format.split('{prompt}')
#     prompt = ''
#     for message in messages:
#         role = message['role']
#         message_content = message['content']
#         if role == 'system':
#             prompt += f"{start_token}\n<<SYS>>\n{message_content}\n<</SYS>>\n"
#         elif role == 'user':
#             prompt += f"{start_token}{message_content}{end_token}"
#         else:
#             prompt += f'{message_content}{stop_words[0]}'
#     return prompt


# model = 'togethercomputer/CodeLlama-13b-Instruct'
# stop_words = list(together.Models.info(model)['config']['stop'])
# prompt_format = str(together.Models.info(model)['config']['prompt_format'])
# formatted_prompt = format_prompt_togetherai(
#     messages=sample_message, prompt_format=prompt_format, stop_words=stop_words)
# for token in together.Complete.create_streaming(prompt=formatted_prompt,
#                                                 model=model, stop=stop_words, max_tokens=512):
#     print(token, end="")


# ### litellm

# import os
# from litellm import completion

# os.environ["TOGETHERAI_API_KEY"] = ""

# sample_message = [
#     {"role": "user", "content": "Who are you"},
#     {"role": "assistant", "content": "I am your helpful assistant."},
#     {"role": "user", "content": "Tell me a joke"},
# ]

# res = completion(model="together_ai/togethercomputer/CodeLlama-13b-Instruct",
#                  messages=sample_message, stream=False, max_tokens=1000)

# print(list(res))
