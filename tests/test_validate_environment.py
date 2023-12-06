#### What this tests ####
#    This tests the validate environment function

import sys, os
import traceback

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import time
import litellm

print(litellm.validate_environment("openai/gpt-3.5-turbo"))