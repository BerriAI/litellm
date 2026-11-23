import os
import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest

import litellm
from tests.unit.litellm_core_utils.test_decode_special_tokens import TOKENIZER_JSON


@pytest.mark.parametrize("offline", ("0", "1"))
def test_hub_loader_preserves_environment_auth_cache_and_offline(tmp_path: Path, offline: str) -> None:
    script: Final = """
import json
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import httpx
import huggingface_hub
from huggingface_hub.errors import LocalEntryNotFoundError
import litellm
payload = sys.argv[2].encode()
offline = sys.argv[3] == "1"
observed = []
def handle(request):
    assert not offline, "offline loading issued a request"
    if request.url.path.endswith("/tokenizer.json"):
        observed.append(request.headers.get("authorization"))
        if request.headers.get("authorization") != "Bearer audit-fixture-token":
            return httpx.Response(401)
    return httpx.Response(200, headers={"content-length": str(len(payload)), "etag": '"fixture"', "x-repo-commit": "a" * 40}, content=payload if request.method == "GET" else b"")
if not offline:
    huggingface_hub.set_client_factory(lambda: httpx.Client(transport=httpx.MockTransport(handle)))
try:
    tokenizer = litellm.create_pretrained_tokenizer("test-fixture/tokenizer")["tokenizer"]
except LocalEntryNotFoundError:
    assert offline
    assert observed == []
else:
    assert not offline
    assert "Bearer audit-fixture-token" in observed
    assert tokenizer.decode(tokenizer.encode("Hello World").ids) == "Hello World"
    assert tuple(Path(sys.argv[4]).rglob("tokenizer.json"))
print("compatible")
"""
    result: Final = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            str(Path(litellm.__file__).parent.parent),
            TOKENIZER_JSON,
            offline,
            str(tmp_path / "cache"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        env={
            **os.environ,
            "HF_HOME": str(tmp_path / "home"),
            "HF_HUB_CACHE": str(tmp_path / "cache"),
            "HF_ENDPOINT": "http://127.0.0.1:9",
            "HF_TOKEN": "audit-fixture-token",
            "HF_HUB_OFFLINE": offline,
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "0",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "compatible"


@pytest.mark.parametrize("rust", (None, "0", "1"))
def test_tokenization_stays_offline(tmp_path: Path, rust: str | None) -> None:
    script: Final = """
import sys
sys.path.insert(0, sys.argv[1])
def reject_network(event, args):
    if event == "socket.connect":
        raise AssertionError("tokenizer attempted a network connection")
sys.addaudithook(reject_network)
import litellm
from litellm.rust_bridge.tokenizer import get_encoding
import tiktoken
from tokenizers import Tokenizer
assert isinstance(litellm.encoding, tiktoken.Encoding)
for name in ("cl100k_base", "o200k_base", "o200k_harmony", "p50k_base", "p50k_edit"):
    encoding = get_encoding(name)
    text = "offline café 漢字 🙂" + " " * 64
    assert encoding.decode(encoding.encode(text)) == text
ids = litellm.encode(text="hello world")
assert litellm.decode(tokens=ids) == "hello world"
assert litellm.token_counter(model=None, text="hello world") == len(ids)
custom = litellm.create_tokenizer(sys.argv[2])
assert isinstance(custom["tokenizer"], Tokenizer)
custom["tokenizer"].enable_padding(pad_id=0, pad_token="[UNK]")
assert litellm.decode(tokens=litellm.encode(text="Hello World", custom_tokenizer=custom), custom_tokenizer=custom) == "Hello World"
print("compatible")
"""
    result: Final = subprocess.run(
        [sys.executable, "-I", "-c", script, str(Path(litellm.__file__).parent.parent), TOKENIZER_JSON],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=tmp_path,
        env={
            **{key: value for key, value in os.environ.items() if key != "LITELLM_RUST"},
            **({"LITELLM_RUST": rust} if rust is not None else {}),
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "TIKTOKEN_CACHE_DIR": str(tmp_path / "unused-tokenizer-cache"),
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "compatible"
    assert not (tmp_path / "unused-tokenizer-cache").exists()
