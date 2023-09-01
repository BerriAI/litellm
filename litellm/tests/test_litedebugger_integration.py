#### What this tests ####
#    This tests if logging to the litedebugger integration actually works

# Test Scenarios (test across normal completion, streaming)
## 1: Pre-API-Call
## 2: Post-API-Call
## 3: On LiteLLM Call success
## 4: On LiteLLM Call failure


import sys, os, io
import traceback, logging
import pytest
import dotenv
dotenv.load_dotenv()

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a stream handler
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)

# Create a function to log information
def logger_fn(message):
    logger.info(message)

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import completion_with_split_tests
from openai.error import AuthenticationError
litellm.set_verbose = True

score = 0
split_per_model = {
	"gpt-4": 0, 
	"claude-instant-1.2": 1
}


user_message = "Hello, how are you?"
messages = [{"content": user_message, "role": "user"}]

# #Test 1: On completion call - without setting client to true -> ensure litedebugger is not initialized
# try:
#     # Redirect stdout
#     old_stdout = sys.stdout
#     sys.stdout = new_stdout = io.StringIO()

#     response = completion_with_split_tests(models=split_per_model, messages=messages)

#     # Restore stdout
#     sys.stdout = old_stdout
#     output = new_stdout.getvalue().strip()

#     if "LiteLLMDebugger" in output:
#         raise Exception("LiteLLM Debugger should not be called!")
#     score += 1
# except Exception as e:
#     pytest.fail(f"Error occurred: {e}")


# # Test 2: On normal completion call - setting client to true
# def test_completion_with_client():
#     try:
#         # Redirect stdout
#         old_stdout = sys.stdout
#         sys.stdout = new_stdout = io.StringIO()

#         response = completion_with_split_tests(models=split_per_model, messages=messages, use_client=True, id="6d383c99-488d-481d-aa1b-1f94935cec44")

#         # Restore stdout
#         sys.stdout = old_stdout
#         output = new_stdout.getvalue().strip()

#         if "LiteDebugger: Pre-API Call Logging" not in output:
#             raise Exception(f"LiteLLMDebugger: pre-api call not logged!")
#         if "LiteDebugger: Post-API Call Logging" not in output:
#             raise Exception("LiteLLMDebugger: post-api call not logged!")
#         if "LiteDebugger: Success/Failure Call Logging" not in output:
#             raise Exception("LiteLLMDebugger: success/failure call not logged!")
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# # Test 3: On streaming completion call - setting client to true
# try:
#     # Redirect stdout
#     old_stdout = sys.stdout
#     sys.stdout = new_stdout = io.StringIO()

#     response = completion_with_split_tests(models=split_per_model, messages=messages, stream=True, use_client=True, override_client=True, id="6d383c99-488d-481d-aa1b-1f94935cec44")
#     for data in response:
#         continue
#     # Restore stdout
#     sys.stdout = old_stdout
#     output = new_stdout.getvalue().strip()

#     if "LiteDebugger: Pre-API Call Logging" not in output:
#         raise Exception("LiteLLMDebugger: pre-api call not logged!")
#     if "LiteDebugger: Post-API Call Logging" not in output:
#         raise Exception("LiteLLMDebugger: post-api call not logged!")
#     if "LiteDebugger: Success/Failure Call Logging" not in output:
#         raise Exception("LiteLLMDebugger: success/failure call not logged!")
# except Exception as e:
#     pytest.fail(f"Error occurred: {e}")

