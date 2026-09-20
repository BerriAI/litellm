from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence, Set
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Final, Literal, Protocol, TypeAlias, runtime_checkable

from litellm.rust_bridge._native import HuggingFaceEncoding
from litellm.rust_bridge._native import Tokenizer as NativeTokenizer

SpecialTokens: TypeAlias = Literal["all"] | Collection[str]
HuggingFaceInput: TypeAlias = str | list[str] | tuple[str, ...]
HuggingFaceBatchInput: TypeAlias = HuggingFaceInput | tuple[HuggingFaceInput, HuggingFaceInput] | list[HuggingFaceInput]


@dataclass(frozen=True, slots=True)
class OpenAIEncoding:
    name: str
    _native: NativeTokenizer
    _special_tokens: frozenset[str]

    @staticmethod
    def from_tiktoken(encoding: str) -> OpenAIEncoding:
        native: Final = NativeTokenizer.from_tiktoken(encoding)
        return OpenAIEncoding(encoding, native, frozenset(native.special_tokens()))

    @property
    def special_tokens_set(
        self,
    ) -> set[str]:  # mutable-ok: [LIT001, LIT002] SDK return type
        return self._native.special_tokens()

    @property
    def eot_token(self) -> int:
        return self.encode("<|endoftext|>", allowed_special="all")[0]

    def encode(
        self,
        text: str,
        *,
        allowed_special: Literal["all"] | Set[str] = frozenset(),
        disallowed_special: SpecialTokens = "all",
    ) -> list[int]:  # mutable-ok: [LIT001, LIT002] SDK return type
        allowed: Final = self._special_tokens if allowed_special == "all" else frozenset(allowed_special)
        disallowed: Final = (
            self._special_tokens - allowed if disallowed_special == "all" else frozenset(disallowed_special)
        )
        for special in disallowed:
            if special in text:
                raise ValueError(f"Encountered text corresponding to disallowed special token {special!r}")
        if not allowed:
            return self.encode_ordinary(text)
        return self._native.encode_special(text, tuple(allowed))

    def encode_ordinary(self, text: str) -> list[int]:  # mutable-ok: [LIT001, LIT002] SDK return type
        return self._native.encode(text)

    def count(self, text: str) -> int:
        return self._native.count(text)

    def encode_batch(
        self,
        text: Sequence[str],
        *,
        num_threads: int = 8,
        allowed_special: Literal["all"] | Set[str] = frozenset(),
        disallowed_special: SpecialTokens = "all",
    ) -> list[list[int]]:  # mutable-ok: [LIT001, LIT002] SDK return type
        encode: Final = partial(self.encode, allowed_special=allowed_special, disallowed_special=disallowed_special)
        with ThreadPoolExecutor(num_threads) as executor:
            return list(  # mutable-ok: [LIT002] SDK returns a list
                executor.map(encode, text)
            )

    def encode_ordinary_batch(
        self, text: Sequence[str], *, num_threads: int = 8
    ) -> list[list[int]]:  # mutable-ok: [LIT001, LIT002] SDK return type
        with ThreadPoolExecutor(num_threads) as executor:
            return list(  # mutable-ok: [LIT002] SDK returns a list
                executor.map(self.encode_ordinary, text)
            )

    def decode_bytes(self, tokens: Sequence[int]) -> bytes:
        return self._native.decode_bytes(tokens)

    def decode(self, tokens: Sequence[int], errors: str = "replace") -> str:
        return self.decode_bytes(tokens).decode("utf-8", errors=errors)

    def decode_single_token_bytes(self, token: int) -> bytes:
        return self.decode_bytes((token,))

    def decode_tokens_bytes(self, tokens: Sequence[int]) -> list[bytes]:  # mutable-ok: [LIT001, LIT002] SDK return type
        return [  # mutable-ok: [LIT002] SDK returns a list
            self.decode_single_token_bytes(token) for token in tokens
        ]

    def decode_batch(
        self, batch: Sequence[Sequence[int]], *, errors: str = "replace", num_threads: int = 8
    ) -> list[str]:  # mutable-ok: [LIT001, LIT002] SDK return type
        with ThreadPoolExecutor(num_threads) as executor:
            return list(  # mutable-ok: [LIT002] SDK returns a list
                executor.map(partial(self.decode, errors=errors), batch)
            )

    def decode_bytes_batch(
        self, batch: Sequence[Sequence[int]], *, num_threads: int = 8
    ) -> list[bytes]:  # mutable-ok: [LIT001, LIT002] SDK return type
        with ThreadPoolExecutor(num_threads) as executor:
            return list(  # mutable-ok: [LIT002] SDK returns a list
                executor.map(self.decode_bytes, batch)
            )

    def __reduce__(self) -> tuple[Callable[[str], OpenAIEncoding], tuple[str]]:
        return (OpenAIEncoding.from_tiktoken, (self.name,))


@dataclass(frozen=True, slots=True)
class HuggingFaceTokenizer:
    _native: NativeTokenizer

    @staticmethod
    def from_str(json: str) -> HuggingFaceTokenizer:
        return HuggingFaceTokenizer(NativeTokenizer.from_json(json))

    from_json = from_str

    @staticmethod
    def from_file(path: str) -> HuggingFaceTokenizer:
        return HuggingFaceTokenizer.from_str(Path(path).read_text(encoding="utf-8"))

    @staticmethod
    def from_pretrained(identifier: str, revision: str = "main", token: str | None = None) -> HuggingFaceTokenizer:
        return HuggingFaceTokenizer(NativeTokenizer.from_pretrained(identifier, revision=revision, token=token))

    def to_str(self, pretty: bool = False) -> str:
        return self._native.to_json(pretty)

    def save(self, path: str, pretty: bool = True) -> None:
        Path(path).write_text(self.to_str(pretty), encoding="utf-8")

    @property
    def name(self) -> str:
        return self._native.name

    def encode(
        self,
        sequence: HuggingFaceInput,
        pair: HuggingFaceInput | None = None,
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
    ) -> HuggingFaceEncoding:
        return self._native.encode_huggingface(sequence, pair, is_pretokenized, add_special_tokens)

    def encode_batch(
        self,
        input: Sequence[HuggingFaceBatchInput],
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
    ) -> list[HuggingFaceEncoding]:  # mutable-ok: [LIT001, LIT002] SDK return type
        return self._encode_batch(input, is_pretokenized, add_special_tokens, fast=False)

    def encode_batch_fast(
        self,
        input: Sequence[HuggingFaceBatchInput],
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
    ) -> list[HuggingFaceEncoding]:  # mutable-ok: [LIT001, LIT002] SDK return type
        return self._encode_batch(input, is_pretokenized, add_special_tokens, fast=True)

    def _encode_batch(
        self, input: Sequence[HuggingFaceBatchInput], is_pretokenized: bool, add_special_tokens: bool, fast: bool
    ) -> list[HuggingFaceEncoding]:  # mutable-ok: [LIT001, LIT002] SDK return type
        sequences: Final = tuple(_batch_input(item, is_pretokenized) for item in input)
        return self._native.encode_batch_huggingface(sequences, is_pretokenized, add_special_tokens, fast)

    def count(self, text: str) -> int:
        return self._native.count(text)

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return self._native.decode(ids, skip_special_tokens=skip_special_tokens)

    def decode_batch(
        self, sequences: Sequence[Sequence[int]], skip_special_tokens: bool = True
    ) -> list[str]:  # mutable-ok: [LIT001, LIT002] SDK return type
        return [  # mutable-ok: [LIT002] SDK returns a list
            self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in sequences
        ]

    def __reduce__(self) -> tuple[Callable[[str], HuggingFaceTokenizer], tuple[str]]:
        return (HuggingFaceTokenizer.from_str, (self.to_str(),))


def _batch_input(
    item: HuggingFaceBatchInput, is_pretokenized: bool
) -> tuple[HuggingFaceInput, HuggingFaceInput | None]:
    if isinstance(item, str):
        return (item, None)
    if is_pretokenized and all(isinstance(word, str) for word in item):
        return (tuple(word for word in item if isinstance(word, str)), None)
    if len(item) != 2:
        raise TypeError("batch input must be a sequence or a pair of sequences")
    return (item[0], item[1])


Tokenizer: TypeAlias = OpenAIEncoding | HuggingFaceTokenizer


class _AddedToken(Protocol):
    @property
    def special(self) -> bool: ...


@runtime_checkable
class _AddedTokenDecoder(Protocol):
    def get_added_tokens_decoder(self) -> Mapping[int, _AddedToken]: ...


def strip_special_tokens(tokenizer: object, tokens: Sequence[int]) -> Sequence[int]:
    if not isinstance(tokenizer, _AddedTokenDecoder):
        return tokens
    try:
        added: Final = tokenizer.get_added_tokens_decoder()
    except Exception:  # noqa: BLE001  # optional metadata failures historically fall back to decoding
        return tokens
    special_ids: Final = frozenset(token_id for token_id, token in added.items() if token.special)
    return tuple(token for token in tokens if token not in special_ids)
