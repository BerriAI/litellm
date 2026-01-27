import os

import litellm

try:
    # New and recommended way to access resources
    from importlib import resources

    filename = str(resources.files(litellm).joinpath("litellm_core_utils/tokenizers"))
except (ImportError, AttributeError):
    # Old way to access resources, which setuptools deprecated some time ago
    import pkg_resources  # type: ignore

    filename = pkg_resources.resource_filename(
        __name__, "litellm_core_utils/tokenizers"
    )

# Check if the directory is writable. If not, use /tmp as a fallback.
# This is especially important for non-root Docker environments where the package directory is read-only.
is_non_root = os.getenv("LITELLM_NON_ROOT", "").lower() == "true"
if not os.access(filename, os.W_OK) and is_non_root:
    filename = "/tmp/tiktoken_cache"
    os.makedirs(filename, exist_ok=True)

os.environ["TIKTOKEN_CACHE_DIR"] = os.getenv(
    "CUSTOM_TIKTOKEN_CACHE_DIR", filename
)  # use local copy of tiktoken b/c of - https://github.com/BerriAI/litellm/issues/1071
import tiktoken
import time
import random

# Retry logic to handle race conditions when multiple processes try to create
# the tiktoken cache file simultaneously (common in parallel test execution on Windows)
_max_retries = 5
_retry_delay = 0.1  # Start with 100ms

for attempt in range(_max_retries):
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        break
    except (FileExistsError, OSError):
        if attempt == _max_retries - 1:
            # Last attempt, re-raise the exception
            raise
        # Exponential backoff with jitter to reduce collision probability
        delay = _retry_delay * (2**attempt) + random.uniform(0, 0.1)
        time.sleep(delay)
