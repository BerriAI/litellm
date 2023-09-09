#### What this tests ####
#    This tests the validate environment function

import sys, os
import traceback

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import time
import litellm

api_key = litellm.validate_environment()