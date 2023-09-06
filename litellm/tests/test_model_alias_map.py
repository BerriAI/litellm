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

model_alias_map = {
    "llama2": "replicate/llama-2-70b-chat:2796ee9483c3fd7aa2e171d38f4ca12251a30609463dcfd4cd76703f22e96cdf"
}

litellm.model_alias_map = model_alias_map

try:
    completion(
        "llama2",
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
        top_p=0.1,
        temperature=0.01,
        num_beams=4,
        max_tokens=60,
    )
except Exception as e:
    print(e)
