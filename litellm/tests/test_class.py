# #### What this tests ####
# #    This tests the LiteLLM Class

# import sys, os
# import traceback
# import pytest
# sys.path.insert(
#     0, os.path.abspath("../..")
# )  # Adds the parent directory to the system path
# import litellm
# litellm.set_verbose = True
# from litellm import Router
# import instructor
# from pydantic import BaseModel

# # This enables response_model keyword
# # from client.chat.completions.create
# client = instructor.patch(Router(model_list=[{
#     "model_name": "gpt-3.5-turbo", # openai model name 
#     "litellm_params": { # params for litellm completion/embedding call 
#         "model": "azure/chatgpt-v-2", 
#         "api_key": os.getenv("AZURE_API_KEY"),
#         "api_version": os.getenv("AZURE_API_VERSION"),
#         "api_base": os.getenv("AZURE_API_BASE")
#     }
# }]))

# class UserDetail(BaseModel):
#     name: str
#     age: int

# user = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     response_model=UserDetail,
#     messages=[
#         {"role": "user", "content": "Extract Jason is 25 years old"},
#     ]
# )

# assert isinstance(user, UserDetail)
# assert user.name == "Jason"
# assert user.age == 25

# print(f"user: {user}")