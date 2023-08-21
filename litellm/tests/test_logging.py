#### What this tests ####
#    This tests error logging (with custom user functions) for the raw `completion` + `embedding` endpoints

import sys, os
import traceback

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import embedding, completion

litellm.set_verbose = False

score = 0


def logger_fn(model_call_object: dict):
    print(f"model call details: {model_call_object}")


user_message = "Hello, how are you?"
messages = [{"content": user_message, "role": "user"}]

# test on openai completion call
try:
    response = completion(model="gpt-3.5-turbo", messages=messages, logger_fn=logger_fn)
    score += 1
except:
    print(f"error occurred: {traceback.format_exc()}")
    pass

# test on non-openai completion call
try:
    response = completion(
        model="claude-instant-1", messages=messages, logger_fn=logger_fn
    )
    print(f"claude response: {response}")
    score += 1
except:
    print(f"error occurred: {traceback.format_exc()}")
    pass

# # test on openai embedding call
# try:
#     response = embedding(model='text-embedding-ada-002', input=[user_message], logger_fn=logger_fn)
#     score +=1
# except:
#     traceback.print_exc()

# # test on bad azure openai embedding call -> missing azure flag and this isn't an embedding model
# try:
#     response = embedding(model='chatgpt-test', input=[user_message], logger_fn=logger_fn)
# except:
#     score +=1 # expect this to fail
#     traceback.print_exc()

# # test on good azure openai embedding call
# try:
#     response = embedding(model='azure-embedding-model', input=[user_message], azure=True, logger_fn=logger_fn)
#     score +=1
# except:
#     traceback.print_exc()


# print(f"Score: {score}, Overall score: {score/5}")
