#### What this tests ####
#    This tests litellm.token_counter.token_counter() function
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Final
import tiktoken
from tokenizers import Regex, Tokenizer, models, pre_tokenizers

from unittest.mock import AsyncMock

import litellm
from litellm import decode, encode
import litellm.constants
from litellm.types.utils import SelectTokenizerResponse


## Choose which token_counter the test will use.


# test_token_counter_normal_plus_function_calling()


# test_tokenizers()


def test_encoding_and_decoding(tmp_path: Path):
    sample_text = "Hellö World, this is my input string!"

    # openai encoding + decoding
    openai_tokens = encode(model="gpt-3.5-turbo", text=sample_text)
    openai_text = decode(model="gpt-3.5-turbo", tokens=openai_tokens)

    assert openai_text == sample_text

    # claude encoding + decoding
    claude_tokens = encode(model="claude-3-5-haiku-20241022", text=sample_text)

    claude_text = decode(model="claude-3-5-haiku-20241022", tokens=claude_tokens)

    assert claude_text == sample_text

    # cohere encoding + decoding
    cohere_tokens = encode(model="command-nightly", text=sample_text)
    cohere_text = decode(model="command-nightly", tokens=cohere_tokens)

    assert cohere_text == sample_text

    # llama2 encoding + decoding
    words = sample_text.split()
    result = _run_in_memory_hub(
        HUB_ROUND_TRIP_SCRIPT,
        {
            "hf-internal-testing/llama-tokenizer": _word_level_tokenizer_json(
                pre_tokenizers.WhitespaceSplit(),
                vocab={"[UNK]": 0, **{word: i + 1 for i, word in enumerate(words)}},
            )
        },
        sample_text,
        tmp_path,
    )

    assert result["decoded"] == sample_text
    assert result["requested"] == ["hf-internal-testing/llama-tokenizer"]
    assert len(result["tokens"]) == len(words)
    assert len(result["tokens"]) != len(encode(model="gpt-3.5-turbo", text=sample_text))


# test_encoding_and_decoding()


# test_gpt_vision_token_counting()

from litellm.utils import _HUGGINGFACE_TOKENIZER_LOAD_OUTCOMES, claude_json_str

# Clear the cache at module load to ensure clean state
_HUGGINGFACE_TOKENIZER_LOAD_OUTCOMES.flush_cache()


HUB_SETUP_SCRIPT: Final = """
import json
import sys
sys.path.insert(0, sys.argv[1])
import httpx
import huggingface_hub
import litellm
served = json.loads(sys.argv[2])
text = sys.argv[3]
requested = []
def handle(request):
    repo = request.url.path.lstrip("/").split("/resolve/")[0]
    if repo not in served or not request.url.path.endswith("/tokenizer.json"):
        return httpx.Response(404)
    requested.append(repo)
    payload = served[repo].encode()
    headers = {"content-length": str(len(payload)), "etag": '"fixture"', "x-repo-commit": "a" * 40}
    return httpx.Response(200, headers=headers, content=payload if request.method == "GET" else b"")
huggingface_hub.set_client_factory(lambda: httpx.Client(transport=httpx.MockTransport(handle)))
"""

HUB_TOKENIZER_SCRIPT: Final = HUB_SETUP_SCRIPT + """
litellm.cohere_models = {"command-r-v1"}
litellm.anthropic_models = {"claude-2"}
custom = litellm.create_pretrained_tokenizer("Xenova/llama-3-tokenizer")
print(json.dumps({
    "llama2": litellm.token_counter(model="meta-llama/Llama-2-7b-chat", text=text),
    "llama3": litellm.token_counter(model="meta-llama/llama-3-70b-instruct", text=text),
    "cohere": litellm.token_counter(model="command-r-v1", text=text),
    "anthropic": litellm.token_counter(model="claude-2", text=text),
    "custom": litellm.token_counter(custom_tokenizer=custom, text=text),
    "requested": sorted(set(requested)),
}))
"""

HUB_ROUND_TRIP_SCRIPT: Final = HUB_SETUP_SCRIPT + """
tokens = litellm.encode(model="meta-llama/Llama-2-7b-chat", text=text)
print(json.dumps({"tokens": tokens, "decoded": litellm.decode(model="meta-llama/Llama-2-7b-chat", tokens=tokens), "requested": sorted(set(requested))}))
"""

HUB_RETRY_DURING_EXIT_SCRIPT: Final = HUB_SETUP_SCRIPT + """
import threading
import time
from litellm.utils import _HuggingFaceTokenizerLoadOutcomes, _return_huggingface_tokenizer
retry_reached_hub = threading.Event()
def answer_once_the_process_is_exiting(request):
    retry_reached_hub.set()
    while threading.main_thread().is_alive():
        time.sleep(0.01)
    time.sleep(0.5)
    return handle(request)
now = [0.0]
outcomes = _HuggingFaceTokenizerLoadOutcomes(clock=lambda: now[0])
model = "meta-llama/llama-3-70b-instruct"
_return_huggingface_tokenizer(model, outcomes)
huggingface_hub.set_client_factory(lambda: httpx.Client(transport=httpx.MockTransport(answer_once_the_process_is_exiting)))
now[0] = 3600.0
_return_huggingface_tokenizer(model, outcomes)
print(json.dumps({"retry_reached_hub": retry_reached_hub.wait(timeout=60)}))
"""


def _word_level_tokenizer_json(
    pre_tokenizer: pre_tokenizers.PreTokenizer, vocab: Mapping[str, int] | None = None
) -> str:
    tokenizer: Final = Tokenizer(
        models.WordLevel(vocab=dict(vocab) if vocab is not None else {"[UNK]": 0}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = pre_tokenizer
    return tokenizer.to_str()


def _run_in_memory_hub(script: str, served: dict[str, str], text: str, tmp_path: Path) -> dict:
    result: Final = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            str(Path(litellm.__file__).parent.parent),
            json.dumps(served),
            text,
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            **os.environ,
            "HF_HOME": str(tmp_path / "home"),
            "HF_HUB_CACHE": str(tmp_path / "cache"),
            "HF_ENDPOINT": "http://127.0.0.1:9",
            "HF_HUB_OFFLINE": "0",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        },
    )

    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_token_counter_uses_the_tokenizer_of_each_model_family_and_of_a_custom_tokenizer(tmp_path: Path) -> None:
    sample: Final = "Tokenizers disagree: anthropic, tiktoken; llama-2 & llama-3!"
    served: Final = {
        "hf-internal-testing/llama-tokenizer": _word_level_tokenizer_json(pre_tokenizers.WhitespaceSplit()),
        "Xenova/llama-3-tokenizer": _word_level_tokenizer_json(pre_tokenizers.Split(Regex("."), "isolated")),
        "Xenova/c4ai-command-r-v01-tokenizer": _word_level_tokenizer_json(pre_tokenizers.Whitespace()),
    }
    expected: Final = {repo: len(Tokenizer.from_str(payload).encode(sample).ids) for repo, payload in served.items()}
    anthropic_count: Final = len(Tokenizer.from_str(claude_json_str).encode(sample).ids)
    tiktoken_count: Final = litellm.token_counter(model="gpt-3.5-turbo", text=sample)
    assert len({*expected.values(), anthropic_count, tiktoken_count}) == len(expected) + 2

    counts: Final = _run_in_memory_hub(HUB_TOKENIZER_SCRIPT, served, sample, tmp_path)
    assert counts == {
        "llama2": expected["hf-internal-testing/llama-tokenizer"],
        "llama3": expected["Xenova/llama-3-tokenizer"],
        "cohere": expected["Xenova/c4ai-command-r-v01-tokenizer"],
        "anthropic": anthropic_count,
        "custom": expected["Xenova/llama-3-tokenizer"],
        "requested": sorted(served),
    }


def test_process_exits_cleanly_while_a_background_tokenizer_retry_is_downloading(tmp_path: Path) -> None:
    assert _run_in_memory_hub(HUB_RETRY_DURING_EXIT_SCRIPT, {}, "", tmp_path) == {"retry_reached_hub": True}
