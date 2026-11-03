import os
import tempfile
from typing import Final

import litellm

try:
    # New and recommended way to access resources
    from importlib import resources

    filename = str(resources.files(litellm).joinpath("litellm_core_utils/tokenizers"))
except (ImportError, AttributeError):
    # Old way to access resources, which setuptools deprecated some time ago
    import pkg_resources

    filename = pkg_resources.resource_filename(__name__, "litellm_core_utils/tokenizers")

# Always default TIKTOKEN_CACHE_DIR to the bundled tokenizers directory
# unless the user explicitly overrides it via CUSTOM_TIKTOKEN_CACHE_DIR.
# This keeps tiktoken fully offline-capable by default (see #1071).
custom_cache_dir: Final = os.getenv("CUSTOM_TIKTOKEN_CACHE_DIR")
if custom_cache_dir:
    cache_dir = custom_cache_dir
else:
    cache_dir = filename

# Ensure cache_dir exists and is writable.
# If the target directory cannot be created or is read-only (e.g. read-only site-packages),
# fall back to a user-writable temporary directory to avoid FileNotFoundError / PermissionError.
try:
    os.makedirs(cache_dir, exist_ok=True)
    probe_file = os.path.join(cache_dir, f".probe_{os.getpid()}")
    try:
        with open(probe_file, "wb") as f:
            f.write(b"")
        os.remove(probe_file)
    except (OSError, PermissionError):
        cache_dir = os.path.join(tempfile.gettempdir(), "litellm_tokenizers")
        os.makedirs(cache_dir, exist_ok=True)
except Exception:
    cache_dir = os.path.join(tempfile.gettempdir(), "litellm_tokenizers")
    os.makedirs(cache_dir, exist_ok=True)

os.environ["TIKTOKEN_CACHE_DIR"] = (
    cache_dir  # use local copy of tiktoken b/c of - https://github.com/BerriAI/litellm/issues/1071
)

import random
import time

import tiktoken

# Retry logic to handle race conditions when multiple processes try to create
# the tiktoken cache file simultaneously (common in parallel test execution on Windows)
_max_retries: Final = 5
_retry_delay: Final = 0.1  # Start with 100ms

for attempt in range(_max_retries):
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        break
    except (FileExistsError, OSError):
        # On early failures, ensure fallback cache directory is active if original cache directory had issues
        fallback_dir: Final = os.path.join(tempfile.gettempdir(), "litellm_tokenizers")
        if os.environ.get("TIKTOKEN_CACHE_DIR") != fallback_dir:
            try:
                os.makedirs(fallback_dir, exist_ok=True)
                os.environ["TIKTOKEN_CACHE_DIR"] = fallback_dir
            except Exception:
                pass
        if attempt == _max_retries - 1:
            # Last attempt, re-raise the exception
            raise
        # Exponential backoff with jitter to reduce collision probability
        delay = _retry_delay * (2.0**attempt) + random.uniform(0, 0.1)
        time.sleep(delay)
