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
{
    "top_p": 0.75,
    "prompt": "What's the meaning of life?",
    "num_beams": 4,
    "temperature": 0.1,
}
print(
    completion(
        "llama2",
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
        top_p=0.1,
        temperature=0,
        num_beams=4,
        max_tokens=60,
    )
)
