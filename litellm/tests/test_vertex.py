# import sys, os
# import traceback
# from dotenv import load_dotenv
# load_dotenv()
# import os
# sys.path.insert(0, os.path.abspath('../..'))  # Adds the parent directory to the system path
# import pytest
# import litellm
# from litellm import embedding, completion


# litellm.vertex_project = "hardy-device-386718"
# litellm.vertex_location = "us-central1"
# litellm.set_verbose = True

# user_message = "what's the weather in SF "
# messages = [{ "content": user_message,"role": "user"}]
# def logger_fn(user_model_dict):
#     print(f"user_model_dict: {user_model_dict}")

# # chat-bison
# # response = completion(model="chat-bison", messages=messages, temperature=0.5, top_p=0.1)
# # print(response)

# #text-bison

# response = completion(model="text-bison", messages=messages)
# print(response)

# response = completion(model="text-bison@001", messages=messages, temperature=0.1, logger_fn=logger_fn)
# print(response)

# response = completion(model="text-bison", messages=messages, temperature=0.4, top_p=0.1, logger_fn=logger_fn)
# print(response)

# response = completion(model="text-bison", messages=messages, temperature=0.8, top_p=0.4, top_k=30, logger_fn=logger_fn)
# print(response)

# response = completion(model="text-bison@001", messages=messages, temperature=0.8, top_p=0.4, top_k=30, logger_fn=logger_fn)
# print(response)

# # chat_model = ChatModel.from_pretrained("chat-bison@001")
# # parameters = {
# #     "temperature": 0.2,
# #     "max_output_tokens": 256,
# #     "top_p": 0.8,
# #     "top_k": 40
# # }

# # chat = chat_model.start_chat()
# # response = chat.send_message("who are u? write a sentence", **parameters)
# # print(f"Response from Model: {response.text}")
