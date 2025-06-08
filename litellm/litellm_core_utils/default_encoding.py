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

os.environ["TIKTOKEN_CACHE_DIR"] = os.getenv(
    "CUSTOM_TIKTOKEN_CACHE_DIR", filename
)  # use local copy of tiktoken b/c of - https://github.com/BerriAI/litellm/issues/1071
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")
