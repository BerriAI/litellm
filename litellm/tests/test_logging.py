#### What this tests ####
#    This tests error logging (with custom user functions) for the raw `completion` + `embedding` endpoints

# Test Scenarios (test across completion, streaming, embedding)
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
from litellm import embedding, completion
from openai.error import AuthenticationError
litellm.set_verbose = True

score = 0

user_message = "Hello, how are you?"
messages = [{"content": user_message, "role": "user"}]

# 1. On Call Success
# normal completion 
## test on openai completion call
try:
    # Redirect stdout
    old_stdout = sys.stdout
    sys.stdout = new_stdout = io.StringIO()

    response = completion(model="gpt-3.5-turbo", messages=messages)
    # Restore stdout
    sys.stdout = old_stdout
    output = new_stdout.getvalue().strip()

    if "Logging Details Pre-API Call" not in output:
        raise Exception("Required log message not found!")
    elif "Logging Details Post-API Call" not in output:
        raise Exception("Required log message not found!")
    elif "Logging Details LiteLLM-Success Call" not in output:
        raise Exception("Required log message not found!")
    score += 1
except Exception as e:
    pytest.fail(f"Error occurred: {e}")
    pass

## test on non-openai completion call
try:
    # Redirect stdout
    old_stdout = sys.stdout
    sys.stdout = new_stdout = io.StringIO()

    response = completion(model="claude-instant-1", messages=messages)
    
    # Restore stdout
    sys.stdout = old_stdout
    output = new_stdout.getvalue().strip()

    if "Logging Details Pre-API Call" not in output:
        raise Exception("Required log message not found!")
    elif "Logging Details Post-API Call" not in output:
        raise Exception("Required log message not found!")
    elif "Logging Details LiteLLM-Success Call" not in output:
        raise Exception("Required log message not found!")
    score += 1
except Exception as e:
    pytest.fail(f"Error occurred: {e}")
    pass

# streaming completion
## test on openai completion call
try:
    # Redirect stdout
    old_stdout = sys.stdout
    sys.stdout = new_stdout = io.StringIO()

    response = completion(model="gpt-3.5-turbo", messages=messages)

    # Restore stdout
    sys.stdout = old_stdout
    output = new_stdout.getvalue().strip()

    if "Logging Details Pre-API Call" not in output:
        raise Exception("Required log message not found!")
    elif "Logging Details Post-API Call" not in output:
        raise Exception("Required log message not found!")
    elif "Logging Details LiteLLM-Success Call" not in output:
        raise Exception("Required log message not found!")
    score += 1
except Exception as e:
    pytest.fail(f"Error occurred: {e}")
    pass

## test on non-openai completion call
try:
    # Redirect stdout
    old_stdout = sys.stdout
    sys.stdout = new_stdout = io.StringIO()

    response = completion(model="claude-instant-1", messages=messages)
    
    # Restore stdout
    sys.stdout = old_stdout
    output = new_stdout.getvalue().strip()

    if "Logging Details Pre-API Call" not in output:
        raise Exception("Required log message not found!")
    elif "Logging Details Post-API Call" not in output:
        raise Exception("Required log message not found!")
    elif "Logging Details LiteLLM-Success Call" not in output:
        raise Exception("Required log message not found!")
    score += 1
except Exception as e:
    pytest.fail(f"Error occurred: {e}")
    pass

# embedding

try:
     # Redirect stdout
    old_stdout = sys.stdout
    sys.stdout = new_stdout = io.StringIO()

    response = embedding(model="text-embedding-ada-002", input=["good morning from litellm"])

    # Restore stdout
    sys.stdout = old_stdout
    output = new_stdout.getvalue().strip()

    if "Logging Details Pre-API Call" not in output:
        raise Exception("Required log message not found!")
    elif "Logging Details Post-API Call" not in output:
        raise Exception("Required log message not found!")
    elif "Logging Details LiteLLM-Success Call" not in output:
        raise Exception("Required log message not found!")
except Exception as e:
    pytest.fail(f"Error occurred: {e}")

# ## 2. On LiteLLM Call failure
# ## TEST BAD KEY

# # normal completion 
# ## test on openai completion call
# try:
#     temporary_oai_key = os.environ["OPENAI_API_KEY"]
#     os.environ["OPENAI_API_KEY"] = "bad-key"

#     temporary_anthropic_key = os.environ["ANTHROPIC_API_KEY"]
#     os.environ["ANTHROPIC_API_KEY"] = "bad-key"


#     # Redirect stdout
#     old_stdout = sys.stdout
#     sys.stdout = new_stdout = io.StringIO()
    
#     try:
#         response = completion(model="gpt-3.5-turbo", messages=messages)
#     except AuthenticationError:
#         print(f"raised auth error")
#         pass
#     # Restore stdout
#     sys.stdout = old_stdout
#     output = new_stdout.getvalue().strip()

#     print(output)

#     if "Logging Details Pre-API Call" not in output:
#         raise Exception("Required log message not found!")
#     elif "Logging Details Post-API Call" not in output: 
#         raise Exception("Required log message not found!")
#     elif "Logging Details LiteLLM-Failure Call" not in output:
#         raise Exception("Required log message not found!")

#     os.environ["OPENAI_API_KEY"] = temporary_oai_key
#     os.environ["ANTHROPIC_API_KEY"] = temporary_anthropic_key
    
#     score += 1
# except Exception as e:
#     print(f"exception type: {type(e).__name__}")
#     pytest.fail(f"Error occurred: {e}")
#     pass

# ## test on non-openai completion call
# try:
#     temporary_oai_key = os.environ["OPENAI_API_KEY"]
#     os.environ["OPENAI_API_KEY"] = "bad-key"

#     temporary_anthropic_key = os.environ["ANTHROPIC_API_KEY"]
#     os.environ["ANTHROPIC_API_KEY"] = "bad-key"
#     # Redirect stdout
#     old_stdout = sys.stdout
#     sys.stdout = new_stdout = io.StringIO()

#     try:
#         response = completion(model="claude-instant-1", messages=messages)
#     except AuthenticationError:
#         pass

#     if "Logging Details Pre-API Call" not in output:
#         raise Exception("Required log message not found!")
#     elif "Logging Details Post-API Call" not in output:
#         raise Exception("Required log message not found!")
#     elif "Logging Details LiteLLM-Failure Call" not in output:
#         raise Exception("Required log message not found!")
#     os.environ["OPENAI_API_KEY"] = temporary_oai_key
#     os.environ["ANTHROPIC_API_KEY"] = temporary_anthropic_key
#     score += 1
# except Exception as e:
#     print(f"exception type: {type(e).__name__}")
#     # Restore stdout
#     sys.stdout = old_stdout
#     output = new_stdout.getvalue().strip()

#     print(output)
#     pytest.fail(f"Error occurred: {e}")


# # streaming completion
# ## test on openai completion call
# try:
#     temporary_oai_key = os.environ["OPENAI_API_KEY"]
#     os.environ["OPENAI_API_KEY"] = "bad-key"

#     temporary_anthropic_key = os.environ["ANTHROPIC_API_KEY"]
#     os.environ["ANTHROPIC_API_KEY"] = "bad-key"
#     # Redirect stdout
#     old_stdout = sys.stdout
#     sys.stdout = new_stdout = io.StringIO()

#     try:
#         response = completion(model="gpt-3.5-turbo", messages=messages)
#     except AuthenticationError:
#         pass

#     # Restore stdout
#     sys.stdout = old_stdout
#     output = new_stdout.getvalue().strip()

#     print(output)

#     if "Logging Details Pre-API Call" not in output:
#         raise Exception("Required log message not found!")
#     elif "Logging Details Post-API Call" not in output:
#         raise Exception("Required log message not found!")
#     elif "Logging Details LiteLLM-Failure Call" not in output:
#         raise Exception("Required log message not found!")
    
#     os.environ["OPENAI_API_KEY"] = temporary_oai_key
#     os.environ["ANTHROPIC_API_KEY"] = temporary_anthropic_key
#     score += 1
# except Exception as e:
#     print(f"exception type: {type(e).__name__}")
#     pytest.fail(f"Error occurred: {e}")

# ## test on non-openai completion call
# try:
#     temporary_oai_key = os.environ["OPENAI_API_KEY"]
#     os.environ["OPENAI_API_KEY"] = "bad-key"

#     temporary_anthropic_key = os.environ["ANTHROPIC_API_KEY"]
#     os.environ["ANTHROPIC_API_KEY"] = "bad-key"
#     # Redirect stdout
#     old_stdout = sys.stdout
#     sys.stdout = new_stdout = io.StringIO()

#     try:
#         response = completion(model="claude-instant-1", messages=messages)
#     except AuthenticationError:
#         pass
    
#     # Restore stdout
#     sys.stdout = old_stdout
#     output = new_stdout.getvalue().strip()

#     print(output)

#     if "Logging Details Pre-API Call" not in output:
#         raise Exception("Required log message not found!")
#     elif "Logging Details Post-API Call" not in output:
#         raise Exception("Required log message not found!")
#     elif "Logging Details LiteLLM-Failure Call" not in output:
#         raise Exception("Required log message not found!")
#     score += 1
# except Exception as e:
#     print(f"exception type: {type(e).__name__}")
#     pytest.fail(f"Error occurred: {e}")

# # embedding

# try:
#     temporary_oai_key = os.environ["OPENAI_API_KEY"]
#     os.environ["OPENAI_API_KEY"] = "bad-key"

#     temporary_anthropic_key = os.environ["ANTHROPIC_API_KEY"]
#     os.environ["ANTHROPIC_API_KEY"] = "bad-key"
#     # Redirect stdout
#     old_stdout = sys.stdout
#     sys.stdout = new_stdout = io.StringIO()

#     try:
#         response = embedding(model="text-embedding-ada-002", input=["good morning from litellm"])
#     except AuthenticationError:
#         pass

#     # Restore stdout
#     sys.stdout = old_stdout
#     output = new_stdout.getvalue().strip()

#     print(output)

#     if "Logging Details Pre-API Call" not in output:
#         raise Exception("Required log message not found!")
#     elif "Logging Details Post-API Call" not in output:
#         raise Exception("Required log message not found!")
#     elif "Logging Details LiteLLM-Failure Call" not in output:
#         raise Exception("Required log message not found!")
# except Exception as e:
#     print(f"exception type: {type(e).__name__}")
#     pytest.fail(f"Error occurred: {e}")