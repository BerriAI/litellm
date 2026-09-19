import importlib.util
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import tiktoken.load
from tiktoken_ext import openai_public

REPO_ROOT = Path(__file__).resolve().parents[2]
TOKENIZERS = Path("litellm", "litellm_core_utils", "tokenizers")
TIKTOKEN_CACHE_FILE_NAME = re.compile(r"[0-9a-f]{40}")


def _tokenizers_dir_without_importing_litellm() -> Path:
    spec = importlib.util.find_spec("litellm")
    assert spec is not None and spec.submodule_search_locations
    return Path(next(iter(spec.submodule_search_locations))).joinpath(*TOKENIZERS.parts[1:])


@pytest.mark.parametrize(
    "load_encoding",
    [openai_public.cl100k_base, openai_public.o200k_base, openai_public.p50k_base],
    ids=["cl100k_base", "o200k_base", "p50k_base"],
)
def test_bundled_tiktoken_cache_loads_without_the_network(load_encoding, tmp_path, monkeypatch):
    cache_copy = tmp_path / "tokenizers"
    shutil.copytree(_tokenizers_dir_without_importing_litellm(), cache_copy)
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(cache_copy))

    def refuse_download(url):
        raise AssertionError(f"tiktoken rejected the bundled cache and tried to download {url}")

    monkeypatch.setattr(tiktoken.load, "read_file", refuse_download)

    assert load_encoding()["mergeable_ranks"]


def test_gitattributes_keeps_the_bundled_caches_byte_exact():
    caches = sorted(
        (TOKENIZERS / p.name).as_posix()
        for p in (REPO_ROOT / TOKENIZERS).iterdir()
        if TIKTOKEN_CACHE_FILE_NAME.fullmatch(p.name)
    )
    assert caches, f"no tiktoken cache files in {REPO_ROOT / TOKENIZERS}"
    try:
        out = subprocess.run(
            ["git", "check-attr", "text", "--", *caches],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")

    assert out.splitlines() == [f"{cache}: text: unset" for cache in caches]
