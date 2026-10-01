"""Python faces of the Rust text codecs.

``OpenAIEncoding`` mirrors ``tiktoken.Encoding`` and ``HuggingFaceTokenizer`` mirrors
``tokenizers.Tokenizer``, so a caller holding ``litellm.encoding`` or the object returned by
``litellm.create_tokenizer`` sees the same read-only surface whichever backend the Rust catalog
selected. Both wrappers are immutable: ``tokenizers`` mutators (``enable_padding``,
``enable_truncation``, ``add_tokens``) stay on the Python tokenizer.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence, Set
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, Protocol, TypeAlias, runtime_checkable

import tiktoken
from tokenizers import AddedToken
from tokenizers import Tokenizer as PythonHuggingFaceTokenizer

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from litellm.rust_bridge._native import HuggingFaceEncoding
    from litellm.rust_bridge._native import Tokenizer as NativeTokenizer

SpecialTokens: TypeAlias = Literal["all"] | Collection[str]
AllowedSpecial: TypeAlias = Literal["all"] | Set[str]
HuggingFaceInput: TypeAlias = str | list[str] | tuple[str, ...]
HuggingFaceBatchInput: TypeAlias = HuggingFaceInput | tuple[HuggingFaceInput, HuggingFaceInput] | list[HuggingFaceInput]


@dataclass(frozen=True, slots=True)
class OpenAIEncoding:
    """``tiktoken.Encoding`` over the Rust tiktoken codec."""

    _native: NativeTokenizer
    _special_tokens: Mapping[str, int]

    @staticmethod
    def wrap(native: NativeTokenizer) -> OpenAIEncoding:
        return OpenAIEncoding(native, MappingProxyType(native.special_tokens()))

    @staticmethod
    def from_tiktoken(encoding: str) -> OpenAIEncoding:
        from litellm.rust_bridge._native import Tokenizer as NativeTokenizer

        return OpenAIEncoding.wrap(NativeTokenizer.from_tiktoken(encoding))

    def __repr__(self) -> str:
        return f"<Encoding {self.name!r}>"

    @property
    def name(self) -> str:
        return self._native.name

    @property
    def max_token_value(self) -> int:
        return self._native.max_token_value()

    @property
    def n_vocab(self) -> int:
        """For backwards compatibility. Prefer to use `enc.max_token_value + 1`."""
        return self.max_token_value + 1

    @property
    def eot_token(self) -> int:
        return self._special_tokens["<|endoftext|>"]

    @property
    def special_tokens_set(self) -> set[str]:  # mutable-ok: [LIT001, LIT002] SDK return type
        return set(self._special_tokens)

    def is_special_token(self, token: int) -> bool:
        return self._native.is_special_token(token)

    # ---- encoding -------------------------------------------------------------------------

    def encode_ordinary(self, text: str) -> list[int]:  # mutable-ok: [LIT001, LIT002] SDK return type
        return self._native.encode(text)

    def encode(
        self,
        text: str,
        *,
        allowed_special: AllowedSpecial = frozenset(),
        disallowed_special: SpecialTokens = "all",
    ) -> list[int]:  # mutable-ok: [LIT001, LIT002] SDK return type
        allowed: Final = self._allowed(text, allowed_special, disallowed_special)
        if not allowed:
            return self.encode_ordinary(text)
        return self._native.encode_special(text, tuple(allowed))

    def encode_to_numpy(
        self,
        text: str,
        *,
        allowed_special: AllowedSpecial = frozenset(),
        disallowed_special: SpecialTokens = "all",
    ) -> npt.NDArray[np.uint32]:
        import numpy

        return numpy.asarray(
            self.encode(text, allowed_special=allowed_special, disallowed_special=disallowed_special),
            dtype=numpy.uint32,
        )

    def encode_ordinary_batch(
        self, text: Sequence[str], *, num_threads: int = 8
    ) -> list[list[int]]:  # mutable-ok: [LIT001, LIT002] SDK return type
        with ThreadPoolExecutor(num_threads) as executor:
            return list(  # mutable-ok: [LIT002] SDK returns a list
                executor.map(self.encode_ordinary, text)
            )

    def encode_batch(
        self,
        text: Sequence[str],
        *,
        num_threads: int = 8,
        allowed_special: AllowedSpecial = frozenset(),
        disallowed_special: SpecialTokens = "all",
    ) -> list[list[int]]:  # mutable-ok: [LIT001, LIT002] SDK return type
        encode: Final = partial(self.encode, allowed_special=allowed_special, disallowed_special=disallowed_special)
        with ThreadPoolExecutor(num_threads) as executor:
            return list(  # mutable-ok: [LIT002] SDK returns a list
                executor.map(encode, text)
            )

    def encode_with_unstable(
        self,
        text: str,
        *,
        allowed_special: AllowedSpecial = frozenset(),
        disallowed_special: SpecialTokens = "all",
    ) -> tuple[list[int], list[list[int]]]:  # mutable-ok: [LIT001, LIT002] SDK return type
        """The stable tokens of `text` and every completion its unstable tail could become.

        Completions come back sorted; tiktoken returns them in hash order."""
        allowed: Final = self._allowed(text, allowed_special, disallowed_special)
        return self._native.encode_with_unstable(text, tuple(allowed))

    def encode_single_token(self, text_or_bytes: str | bytes) -> int:
        """The token of one whole piece, special tokens included. Raises `KeyError` otherwise."""
        piece: Final = text_or_bytes.encode("utf-8") if isinstance(text_or_bytes, str) else text_or_bytes
        return self._native.encode_single_token(piece)

    def count(self, text: str, fast: bool = False) -> int:
        """Count ordinary text; `fast` accelerates supported encodings and otherwise counts normally."""
        return self._native.count(text, fast)

    # ---- decoding -------------------------------------------------------------------------

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

    def decode_with_offsets(
        self, tokens: Sequence[int]
    ) -> tuple[str, list[int]]:  # mutable-ok: [LIT001, LIT002] SDK return type
        """The decoded text and, per token, the index of the first character holding its bytes.

        Like tiktoken, raises `UnicodeDecodeError` when the tokens do not decode to valid UTF-8."""
        token_bytes: Final = self.decode_tokens_bytes(tokens)
        text_len = 0
        offsets: Final[list[int]] = []  # mutable-ok: [LIT001] local accumulator
        for token in token_bytes:
            offsets.append(max(0, text_len - (0x80 <= token[0] < 0xC0)))
            text_len += sum(1 for c in token if not 0x80 <= c < 0xC0)
        return b"".join(token_bytes).decode("utf-8", errors="strict"), offsets

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

    def token_byte_values(self) -> list[bytes]:  # mutable-ok: [LIT001, LIT002] SDK return type
        return self._native.token_byte_values()

    def __reduce__(self) -> tuple[Callable[[str], OpenAIEncoding], tuple[str]]:
        return (OpenAIEncoding.from_tiktoken, (self.name,))

    # ---- private --------------------------------------------------------------------------

    def _allowed(self, text: str, allowed_special: AllowedSpecial, disallowed_special: SpecialTokens) -> frozenset[str]:
        """tiktoken's special-token policy: which specials `text` may encode, after rejecting
        any it must not contain."""
        allowed: Final = frozenset(self._special_tokens) if allowed_special == "all" else frozenset(allowed_special)
        disallowed: Final = (
            frozenset(self._special_tokens) - allowed if disallowed_special == "all" else frozenset(disallowed_special)
        )
        for token in disallowed:
            if token in text:
                raise ValueError(
                    f"Encountered text corresponding to disallowed special token {token!r}.\n"
                    "If you want this text to be encoded as a special token, "
                    f"pass it to `allowed_special`, e.g. `allowed_special={{{token!r}, ...}}`.\n"
                    "If you want this text to be encoded as normal text, disable the check for this token "
                    f"by passing `disallowed_special=(enc.special_tokens_set - {{{token!r}}})`.\n"
                    "To disable this check for all special tokens, pass `disallowed_special=()`.\n"
                )
        return allowed


@dataclass(frozen=True, slots=True)
class HuggingFaceTokenizer:
    """The read-only ``tokenizers.Tokenizer`` surface over the Rust Hugging Face codec."""

    _native: NativeTokenizer

    @staticmethod
    def from_str(json: str) -> HuggingFaceTokenizer:
        from litellm.rust_bridge._native import Tokenizer as NativeTokenizer

        return HuggingFaceTokenizer(NativeTokenizer.from_json(json))

    from_json = from_str

    @staticmethod
    def from_buffer(buffer: bytes) -> HuggingFaceTokenizer:
        return HuggingFaceTokenizer.from_str(buffer.decode("utf-8"))

    @staticmethod
    def from_file(path: str) -> HuggingFaceTokenizer:
        return HuggingFaceTokenizer.from_str(Path(path).read_text(encoding="utf-8"))

    @staticmethod
    def from_pretrained(identifier: str, revision: str = "main", token: str | None = None) -> HuggingFaceTokenizer:
        from litellm.rust_bridge._native import Tokenizer as NativeTokenizer

        return HuggingFaceTokenizer(NativeTokenizer.from_pretrained(identifier, revision=revision, token=token))

    def to_str(self, pretty: bool = False) -> str:
        return self._native.to_json(pretty)

    def save(self, path: str, pretty: bool = True) -> None:
        Path(path).write_text(self.to_str(pretty), encoding="utf-8")

    @property
    def name(self) -> str:
        return self._native.name

    # ---- vocabulary -----------------------------------------------------------------------

    def token_to_id(self, token: str) -> int | None:
        return self._native.token_to_id(token)

    def id_to_token(self, id: int) -> str | None:
        return self._native.id_to_token(id)

    def get_vocab(
        self, with_added_tokens: bool = True
    ) -> dict[str, int]:  # mutable-ok: [LIT001, LIT002] SDK return type
        return self._native.get_vocab(with_added_tokens)

    def get_vocab_size(self, with_added_tokens: bool = True) -> int:
        return self._native.get_vocab_size(with_added_tokens)

    def get_added_tokens_decoder(self) -> dict[int, AddedToken]:  # mutable-ok: [LIT001, LIT002] SDK return type
        return {  # mutable-ok: [LIT002] SDK returns a dict
            token_id: AddedToken(
                content, single_word=single_word, lstrip=lstrip, rstrip=rstrip, normalized=normalized, special=special
            )
            for token_id, (
                content,
                single_word,
                lstrip,
                rstrip,
                normalized,
                special,
            ) in self._native.added_tokens_decoder()
        }

    def num_special_tokens_to_add(self, is_pair: bool) -> int:
        return self._native.num_special_tokens_to_add(is_pair)

    @property
    def padding(self) -> dict[str, object] | None:  # mutable-ok: [LIT001, LIT002] SDK return type
        return self._native.padding()

    @property
    def truncation(self) -> dict[str, object] | None:  # mutable-ok: [LIT001, LIT002] SDK return type
        return self._native.truncation()

    @property
    def encode_special_tokens(self) -> bool:
        return self._native.encode_special_tokens()

    # ---- encoding and decoding ------------------------------------------------------------

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

    def count(self, text: str, fast: bool = False) -> int:
        """Count with this tokenizer's configuration; `fast` uses acceleration where supported."""
        return self._native.count(text, fast)

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


Encoding: TypeAlias = tiktoken.Encoding | OpenAIEncoding
HuggingFace: TypeAlias = PythonHuggingFaceTokenizer | HuggingFaceTokenizer
Tokenizer: TypeAlias = Encoding | HuggingFace


class _AddedToken(Protocol):
    @property
    def special(self) -> bool: ...


@runtime_checkable
class _AddedTokenDecoder(Protocol):
    def get_added_tokens_decoder(self) -> Mapping[int, _AddedToken]: ...


def strip_special_tokens(tokenizer: object, tokens: Sequence[int]) -> Sequence[int]:
    """Drop the special added tokens before a Python `tokenizers` decode; the Rust codec's
    `decode(skip_special_tokens=True)` already does this itself."""
    if isinstance(tokenizer, HuggingFaceTokenizer) or not isinstance(tokenizer, _AddedTokenDecoder):
        return tokens
    try:
        added: Final = tokenizer.get_added_tokens_decoder()
    except Exception:  # noqa: BLE001  # optional metadata failures historically fall back to decoding
        return tokens
    special_ids: Final = frozenset(token_id for token_id, token in added.items() if token.special)
    return tuple(token for token in tokens if token not in special_ids)
