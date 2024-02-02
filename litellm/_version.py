import importlib_metadata
import traceback

try:
    version = importlib_metadata.version("litellm")
except Exception as e:
    traceback.print_exc()
    raise e
