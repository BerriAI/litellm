import hashlib
from pathlib import Path

import pytest

import litellm

TOKENIZERS = Path(litellm.__file__).parent / "litellm_core_utils" / "tokenizers"

# The cl100k_base and o200k_base rank files, and the other bundled encoding,
# named by the hash of their URL as tiktoken's cache expects.
RANK_FILES = [
    "9b5ad71b2ce5302211f9c61530b329a4922fc6a4",
    "fb374d419588a4632f3f557e76b4b70aebbca790",
    "ec7223a39ce59f226a68acc30dc1af2788490e15",
]


@pytest.mark.parametrize("name", RANK_FILES)
def test_bundled_rank_file_has_unix_line_endings(name):
    # A wheel built on Windows with line-ending conversion changes these bytes:
    # tiktoken then fails its hash check, deletes the file and downloads it, so
    # the proxy cannot start offline (#50748).
    assert b"\r" not in (TOKENIZERS / name).read_bytes()


def test_bundled_cl100k_file_matches_the_hash_tiktoken_expects():
    data = (TOKENIZERS / RANK_FILES[0]).read_bytes()
    assert (
        hashlib.sha256(data).hexdigest()
        == "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
    )
