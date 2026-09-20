from pathlib import Path
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

CL100K_BASE_RANK_FILE: Final = "9b5ad71b2ce5302211f9c61530b329a4922fc6a4"
O200K_BASE_RANK_FILE: Final = "fb374d419588a4632f3f557e76b4b70aebbca790"


def cl100k_base_rank_file() -> str:
    """The vendored tiktoken `cl100k_base` rank file (`base64(token) rank` lines)."""
    return Path(filename, CL100K_BASE_RANK_FILE).read_text(encoding="ascii")


def o200k_base_rank_file() -> str:
    """The vendored tiktoken `o200k_base` rank file (`base64(token) rank` lines)."""
    return Path(filename, O200K_BASE_RANK_FILE).read_text(encoding="ascii")


from litellm.litellm_core_utils.tokenizer import OpenAIEncoding as Tokenizer

encoding: Final = Tokenizer.from_tiktoken("cl100k_base")
