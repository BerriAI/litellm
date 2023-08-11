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

# user_message = "what's the weather in SF "
# messages = [{ "content": user_message,"role": "user"}]

# response = completion(model="chat-bison", messages=messages, temperature=0.5, top_p=0.1)
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