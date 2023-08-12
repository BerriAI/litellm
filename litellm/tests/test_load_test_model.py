import sys, os
import traceback
sys.path.insert(0, os.path.abspath('../..'))  # Adds the parent directory to the system path
import litellm
from litellm import load_test_model

result = load_test_model(model="gpt-3.5-turbo", num_calls=5)
print(result)