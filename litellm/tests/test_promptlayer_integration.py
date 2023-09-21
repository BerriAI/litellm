import sys
import os
import io

sys.path.insert(0, os.path.abspath('../..'))

from litellm import completion
import litellm

litellm.success_callback = ["promptlayer"]
litellm.set_verbose = True
import time



# def test_promptlayer_logging():
#     try:
#         # Redirect stdout
#         old_stdout = sys.stdout
#         sys.stdout = new_stdout = io.StringIO()


#         response = completion(model="claude-instant-1.2",
#                               messages=[{
#                                   "role": "user",
#                                   "content": "Hi 👋 - i'm claude"
#                               }])

#         # Restore stdout
#         time.sleep(1)
#         sys.stdout = old_stdout
#         output = new_stdout.getvalue().strip()
#         print(output)
#         if "LiteLLM: Prompt Layer Logging: success" not in output:
#             raise Exception("Required log message not found!")

#     except Exception as e:
#         print(e)

# test_promptlayer_logging()


def test_promptlayer_logging_with_metadata():
    try:
        # Redirect stdout
        old_stdout = sys.stdout
        sys.stdout = new_stdout = io.StringIO()


        response = completion(model="j2-light",
                              messages=[{
                                  "role": "user",
                                  "content": "Hi 👋 - i'm ai21"
                              }], 
                              metadata={"model": "ai21"})

        # Restore stdout
        time.sleep(1)
        sys.stdout = old_stdout
        output = new_stdout.getvalue().strip()
        print(output)
        if "LiteLLM: Prompt Layer Logging: success" not in output:
            raise Exception("Required log message not found!")

    except Exception as e:
        print(e)

# test_promptlayer_logging_with_metadata()




# def test_chat_openai():
#     try:
#         response = completion(model="replicate/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1",
#                               messages=[{
#                                   "role": "user",
#                                   "content": "Hi 👋 - i'm openai"
#                               }])

#         print(response)
#     except Exception as e:
#         print(e)

# test_chat_openai()
