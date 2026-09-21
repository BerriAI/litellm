import copy
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Final, Literal

import pytest
import tiktoken
from tokenizers import Tokenizer as ReferenceTokenizer

import litellm
from litellm.caching._embedding_router import truncate_embedding_input
from litellm.litellm_core_utils.tokenizer import HuggingFaceTokenizer, OpenAIEncoding
from litellm.utils import claude_json_str
from tests.test_litellm.litellm_core_utils.test_decode_special_tokens import TOKENIZER_JSON


@pytest.mark.parametrize(
    "name", ("cl100k_base", "o200k_base", "p50k_base", "p50k_edit", "r50k_base", "gpt2", "o200k_harmony")
)
@pytest.mark.parametrize(
    "text", ("hello world", "café 漢字 🙂", "", "a\ud800b", "\ud83d\ude42", "🙂\ud83d\ude42\udfff", " " * 64)
)
def test_openai_encoding_matches_python_unicode_and_batches(name: str, text: str) -> None:
    reference: Final = tiktoken.get_encoding(name)
    encoding: Final = OpenAIEncoding.from_tiktoken(name)
    expected: Final = reference.encode(text)

    assert encoding.encode(text) == expected
    assert encoding.count(text) == len(expected)
    assert encoding.encode_batch([text], num_threads=2) == reference.encode_batch([text], num_threads=2)
    assert encoding.encode_ordinary_batch([text]) == reference.encode_ordinary_batch([text])
    assert encoding.decode_batch([expected]) == reference.decode_batch([expected])
    assert encoding.decode_bytes_batch([expected]) == reference.decode_bytes_batch([expected])


@pytest.mark.parametrize("allowed", (frozenset(), frozenset({"<|endoftext|>"}), "all"))
@pytest.mark.parametrize("disallowed", (frozenset(), frozenset({"<|fim_prefix|>"}), "all"))
def test_openai_special_token_options_match_python(
    allowed: frozenset[str] | Literal["all"], disallowed: frozenset[str] | Literal["all"]
) -> None:
    reference: Final = tiktoken.get_encoding("cl100k_base")
    encoding: Final = OpenAIEncoding.from_tiktoken(reference.name)
    text: Final = "hello<|endoftext|><|fim_prefix|>world"
    allowed_set: Final = reference.special_tokens_set if allowed == "all" else allowed
    disallowed_set: Final = reference.special_tokens_set - allowed_set if disallowed == "all" else disallowed
    if any(token in text for token in disallowed_set):
        with pytest.raises(ValueError, match="disallowed special token"):
            encoding.encode(text, allowed_special=allowed, disallowed_special=disallowed)
        return
    assert encoding.encode(text, allowed_special=allowed, disallowed_special=disallowed) == reference.encode(
        text, allowed_special=allowed, disallowed_special=disallowed
    )
    assert encoding.special_tokens_set == reference.special_tokens_set
    assert encoding.eot_token == reference.eot_token


@pytest.mark.parametrize("errors", ("replace", "ignore", "backslashreplace", "strict"))
def test_openai_partial_token_decoding_preserves_error_policy(errors: str) -> None:
    reference: Final = tiktoken.get_encoding("cl100k_base")
    encoding: Final = OpenAIEncoding.from_tiktoken(reference.name)
    tokens: Final = reference.encode("🙂")[:1]
    assert encoding.decode_bytes(tokens) == reference.decode_bytes(tokens)
    if errors == "strict":
        with pytest.raises(UnicodeDecodeError):
            encoding.decode(tokens, errors=errors)
        return
    assert encoding.decode(tokens, errors=errors) == reference.decode(tokens, errors=errors)
    assert encoding.decode_tokens_bytes(tokens) == reference.decode_tokens_bytes(tokens)


def test_public_encoding_and_semantic_cache_preserve_truncated_unicode() -> None:
    reference: Final = tiktoken.get_encoding(litellm.encoding.name)
    text: Final = "🙂"
    tokens: Final = reference.encode(text)

    assert litellm.encoding.encode(text, disallowed_special=()) == tokens
    assert litellm.encoding.encode_batch([text]) == [tokens]
    assert litellm.decode(tokens=tokens[:1]) == reference.decode(tokens[:1])
    assert truncate_embedding_input(text, "", 1) == reference.decode(tokens[:1])


@pytest.mark.parametrize("add_special_tokens", (True, False))
def test_huggingface_encoding_preserves_result_fields_and_serialization(add_special_tokens: bool) -> None:
    reference: Final = ReferenceTokenizer.from_str(TOKENIZER_JSON)
    tokenizer: Final = HuggingFaceTokenizer.from_str(TOKENIZER_JSON)
    expected: Final = reference.encode("Hello World", add_special_tokens=add_special_tokens)
    actual: Final = tokenizer.encode("Hello World", add_special_tokens=add_special_tokens)

    assert (actual.ids, actual.tokens, actual.type_ids, actual.offsets, actual.word_ids, actual.sequence_ids) == (
        expected.ids,
        expected.tokens,
        expected.type_ids,
        expected.offsets,
        expected.word_ids,
        expected.sequence_ids,
    )
    assert (actual.attention_mask, actual.special_tokens_mask, actual.n_sequences, len(actual)) == (
        expected.attention_mask,
        expected.special_tokens_mask,
        expected.n_sequences,
        len(expected),
    )
    assert copy.deepcopy(actual).ids == expected.ids
    assert pickle.loads(pickle.dumps(actual)).offsets == expected.offsets
    assert tokenizer.decode(actual.ids, skip_special_tokens=False) == reference.decode(
        expected.ids, skip_special_tokens=False
    )


def test_huggingface_character_offsets_and_pretokenized_pairs_match_python() -> None:
    reference: Final = ReferenceTokenizer.from_str(claude_json_str)
    tokenizer: Final = HuggingFaceTokenizer.from_str(claude_json_str)
    text: Final = "café 漢字 🙂"
    actual: Final = tokenizer.encode(text)
    expected: Final = reference.encode(text)

    assert actual.offsets == expected.offsets
    assert actual.ids == expected.ids
    assert (
        tokenizer.encode(["hello", "world"], ["again"], is_pretokenized=True).ids
        == reference.encode(["hello", "world"], ["again"], is_pretokenized=True).ids
    )


def test_huggingface_batches_apply_padding_across_inputs() -> None:
    reference: Final = ReferenceTokenizer.from_str(TOKENIZER_JSON)
    reference.enable_padding(pad_id=0, pad_token="[UNK]")
    tokenizer: Final = HuggingFaceTokenizer.from_str(reference.to_str())
    inputs: Final = ["Hello", ("Hello World", "World")]
    expected: Final = reference.encode_batch(inputs)
    actual: Final = tokenizer.encode_batch(inputs)
    fast: Final = tokenizer.encode_batch_fast(inputs)

    assert [(item.ids, item.attention_mask, item.offsets) for item in actual] == [
        (item.ids, item.attention_mask, item.offsets) for item in expected
    ]
    assert [item.ids for item in fast] == [item.ids for item in expected]
    assert tokenizer.decode_batch([item.ids for item in actual]) == reference.decode_batch(
        [item.ids for item in expected]
    )


def test_caller_supplied_huggingface_tokenizer_preserves_public_encode_and_count() -> None:
    tokenizer: Final = ReferenceTokenizer.from_str(TOKENIZER_JSON)
    custom: Final = {"type": "huggingface_tokenizer", "tokenizer": tokenizer}
    expected: Final = tokenizer.encode("Hello World").ids

    assert litellm.encode(text="Hello World", custom_tokenizer=custom) == expected
    assert litellm.token_counter(text="Hello World", custom_tokenizer=custom) == len(expected)
    assert litellm.decode(tokens=expected, custom_tokenizer=custom) == "Hello World"


def test_caller_supplied_tiktoken_treats_special_spellings_as_text() -> None:
    tokenizer: Final = tiktoken.get_encoding("cl100k_base")
    custom: Final = {"type": "openai_tokenizer", "tokenizer": tokenizer}
    text: Final = "<|endoftext|>"

    assert litellm.encode(text=text, custom_tokenizer=custom) == tokenizer.encode(text, disallowed_special=())


def test_public_tokenizer_objects_survive_pickle_and_deepcopy(tmp_path: Path) -> None:
    custom: Final = litellm.create_tokenizer(TOKENIZER_JSON)
    tokenizer: Final = custom["tokenizer"]
    path: Final = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))

    assert copy.deepcopy(custom)["tokenizer"].encode("Hello World").ids == tokenizer.encode("Hello World").ids
    assert (
        pickle.loads(pickle.dumps(custom))["tokenizer"].encode("Hello World").ids == tokenizer.encode("Hello World").ids
    )
    assert HuggingFaceTokenizer.from_file(str(path)).encode("Hello World").ids == tokenizer.encode("Hello World").ids
    assert copy.deepcopy(litellm.encoding).encode("hello") == litellm.encoding.encode("hello")
    assert pickle.loads(pickle.dumps(litellm.encoding)).encode("hello") == litellm.encoding.encode("hello")


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


def test_installed_tokenization_does_not_import_python_tokenizer_packages(tmp_path: Path) -> None:
    script: Final = """
import importlib.abc
import sys
sys.path.insert(0, sys.argv[1])
def reject_network(event, args):
    if event == "socket.connect":
        raise AssertionError("tokenizer attempted a network connection")
sys.addaudithook(reject_network)
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"tiktoken", "tokenizers"}:
            raise AssertionError("unexpected runtime dependency: " + fullname)
sys.meta_path.insert(0, Block())
import litellm
from litellm.litellm_core_utils.tokenizer import OpenAIEncoding
for name in ("cl100k_base", "o200k_base", "o200k_harmony", "p50k_base", "p50k_edit", "r50k_base", "gpt2"):
    encoding = OpenAIEncoding.from_tiktoken(name)
    text = "offline café 漢字 🙂" + " " * 64
    assert encoding.decode(encoding.encode(text)) == text
ids = litellm.encode(text="hello world")
assert litellm.decode(tokens=ids) == "hello world"
assert litellm.token_counter(model=None, text="hello world") == len(ids)
custom = litellm.create_tokenizer(sys.argv[2])
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
            **os.environ,
            "LITELLM_RUST": "0",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "TIKTOKEN_CACHE_DIR": str(tmp_path / "unused-tokenizer-cache"),
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "compatible"
    assert not (tmp_path / "unused-tokenizer-cache").exists()


@pytest.mark.parametrize("is_pretokenized", (False, True))
def test_huggingface_batch_sequence_containers_match_python(is_pretokenized: bool) -> None:
    reference: Final = ReferenceTokenizer.from_str(TOKENIZER_JSON)
    tokenizer: Final = HuggingFaceTokenizer.from_str(TOKENIZER_JSON)
    inputs: Final = [["Hello", "World"], ("Hello", "World")]
    actual: Final = tokenizer.encode_batch(inputs, is_pretokenized=is_pretokenized)
    expected: Final = reference.encode_batch(inputs, is_pretokenized=is_pretokenized)
    assert [(item.ids, item.type_ids, item.sequence_ids) for item in actual] == [
        (item.ids, item.type_ids, item.sequence_ids) for item in expected
    ]
