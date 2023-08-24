#### What this tests ####
#    This tests the model alias mapping - if user passes in an alias, and has set an alias, set it to the actual value

import sys, os
import traceback

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import embedding, completion

litellm.set_verbose = True

# Test: Check if the alias created via LiteDebugger is mapped correctly
print(completion("wizard-lm", messages=[{"role": "user", "content": "Hey, how's it going?"}]))