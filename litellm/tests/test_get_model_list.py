import os, sys, traceback

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import get_model_list

print(get_model_list())
print(get_model_list())
# print(litellm.model_list)
