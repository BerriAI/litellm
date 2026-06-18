"""
Conftest for TokenHub tests.
Ensures LITELLM_LOCAL_MODEL_COST_MAP is set before litellm is imported.
"""

import os

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
