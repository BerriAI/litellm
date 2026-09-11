"""Pin tiktoken cl100k_base reference counts for the Rust parity tests.

Run from the repository root with the project environment:

    uv run --no-sync python litellm-rust/crates/token-counter/tests/fixtures/cl100k/generate.py

`texts.jsonl` holds `{"text", "tokens", "pieces"}` lines: `tokens` counted with
`tiktoken.get_encoding("cl100k_base").encode(text, disallowed_special=())`, the
same call `litellm.token_counter` makes, and `pieces` the installed encoding's
split pattern applied with the `regex` module tiktoken itself uses, so a scanner
that splits differently fails even where BPE would count the same. `requests.jsonl` holds
`{"body", "input_tokens"}` lines, `body` being the exact request bytes as a JSON
string, counted with the proxy's admission counter
(`_count_input_tokens(body, "gpt-4")`). Every message in the 50k-token body is
shorter than the Python chunk size so the chunked Python count equals the exact
whole-text tiktoken count the Rust counter produces.
"""

import itertools
import json
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import regex
import tiktoken

from litellm.constants import TIKTOKEN_ENCODE_CHUNK_SIZE_CHARS
from litellm.proxy.spend_tracking.budget_reservation import _count_input_tokens

HERE: Final = Path(__file__).resolve().parent
ENCODING: Final = tiktoken.get_encoding("cl100k_base")
SPLIT_PATTERN: Final = regex.compile(ENCODING._pat_str)  # pyright: ignore[reportPrivateUsage]  # tiktoken has no public accessor

# Mirrors ALPHABET in src/byte_level.rs, plus the pieces the cl100k pattern treats differently.
ALPHABET: Final = (
    "a",
    "Z",
    "e",
    "s",
    "t",
    "d",
    "m",
    "'",
    "'s",
    "'re",
    "'ll",
    "'S",
    "0",
    "9",
    " ",
    "  ",
    "\t",
    "\n",
    "\r\n",
    "\x0b",
    ".",
    ",",
    "!",
    "-",
    "(",
    '"',
    "\xa0",
    "\x85",
    "\u2028",
    "\u3000",
    "\u200b",
    "\u200d",
    "é",
    "e\u0301",
    "ß",
    "漢",
    "字",
    "ع",
    "३",
    "½",
    "Ⅳ",
    "🙂",
    "👍🏽",
    "Ａ",
    "ﬁ",
    "㍿",
    "㋿",
    "ꟲ",
    "𐞁",
    "a\u030a",
    "\u1e0b\u0323",
    "<",
    ">",
    "EOT",
    "<EOT>",
    "<META_START>",
    "'D",
    "'M",
    "'T",
    "'VE",
    "'Re",
    "'ſ",
    "ſ",
    "12345678",
    "٣٤٥٦",
    "<|endoftext|>",
    "<|fim_prefix|>",
    "\r",
    "\r\n\r\n",
    "   \n",
    "!!",
    "#$%",
    "\u00ad",
    "\u0301",
    "\U0001f600\U0001f3fd",
    "İ",
    "ǅ",
)

CORPUS: Final = (
    "",
    "Hello, how are you today?",
    "I'm sure they're right, we'll see. WE'LL SEE, I'M SURE THEY'RE RIGHT, IT'S HERS AND IT'D BE 'D",
    "don't Don'T DON'T won'T i've I'VE i'Ve you'RE 'S 'T 'M 'D 'LL 'VE 'RE 'ſ 'x",
    "1234567890 123 12 1 0000000 ٣٤٥٦٧٨ ३४५६ 1,234,567.89 2026-09-11T18:00:00Z",
    "$abc %def &ghi @jkl _mno #pqr ~stu ^vwx |yz \\a /b :c ;d ?e !f (g )h [i ]j {k }l <m >n =o +p *q",
    "foo   bar  baz \t qux\t\tquux \n\nline\r\nline\r\n\r\n   \n\t\r\n  x  ",
    "trailing spaces   ",
    "trailing tabs\t\t",
    "trailing newline\n",
    "\n\n\n",
    "\r\n\r\n\r\n",
    "   ",
    "😀😃😄 👍🏽 🇺🇸 👨‍👩‍👧‍👦 ✈️ ❤️‍🔥 ٭ ※ ⌘ ⏎",
    "漢字かな交じり文、東京都千代田区。日本語のテキストです。中文测试。한국어 텍스트",
    "مرحبا بالعالم، هذا نص عربي مع أرقام ١٢٣٤٥٦٧ و علامات ترقيم!",
    "Zürich, façade, naïve, Ærøskøbing, Ελληνικά, Русский текст, עברית, हिन्दी, ไทย",
    "e\u0301 a\u030a \u1e0b\u0323 \u0301\u0301 combining\u0308 marks\u0301!",
    "ΣΊΣΥΦΟΣ ǅungla İstanbul ﬁle ﬂow Ａｂｃ ㍿ ㋿ ꟲ 𐞁",
    "<|endoftext|> <|fim_prefix|>code<|fim_middle|>more<|fim_suffix|> <|endofprompt|> <|im_start|>",
    "<EOT> <META_START> <s> </s> [INST] [/INST] <<SYS>>",
    "def f(x):\n    return {'a': x ** 2, \"b\": [1, 2, 3]}  # comment\n\nprint(f(10))\n",
    '{"model":"gpt-4","messages":[{"role":"user","content":"hi\\n"}],"temperature":0.7}',
    "https://example.com/path?query=1&other=two#fragment user@example.com 192.168.0.1",
    "a" * 3000,
    " " * 3000,
    "." * 3000,
    "ab" * 1500,
    "\n" * 3000,
    "0" * 3000,
    "!" * 3000,
    "😀" * 1000,
    "漢" * 1000,
    "\u00a0abc\u00a0! \u2028x \u3000y \u200bz \u200d\u200d q",
    "x\u0085y \x0b\x0c z",
    "\x00\x01\x02 \x7f \ufffd",
    "tab\tseparated\tvalues\n1\t2\t3\n",
    "MiXeD cAsE wOrDs AND ACRONYMS like NASA, HTTP/2, gRPC, iOS, macOS",
    "snake_case_identifier camelCaseIdentifier PascalCaseIdentifier SCREAMING_SNAKE_CASE kebab-case",
    "x'sy x'ty x'rey x'vey x'my x'lly x'dy x'S x'T x'RE x'VE x'M x'LL x'D x'sS x'llL",
    "IT'SOK it'Dbe x'Sy x'Ty x'My x'Dy x'LLy x'VEy x'REy x'Ly x'Vy x'Ry 'Sx'Tx'Mx'LLx'VEx'REx'Dx",
    "'s't're've'm'll'd 'S'T'RE'VE'M'LL'D ''s '''s",
    "9'9 9's a'9 '9 ' 's' ' 's",
    "١٢٣٤ ½⅓¼ ⅣⅤ 𝟘𝟙𝟚𝟛𝟜𝟝𝟞𝟟𝟠𝟡 ①②③",
)

WORDS: Final = (
    "the",
    "quick",
    "brown",
    "fox",
    "jumps",
    "over",
    "lazy",
    "dog",
    "while",
    "counting",
    "tokens",
    "for",
    "budget",
    "reservation",
    "before",
    "admission",
    "on",
    "the",
    "gateway",
    "and",
    "every",
    "request",
    "body",
    "is",
    "scanned",
    "exactly",
    "once",
    "with",
    "a",
    "hand",
    "written",
    "piece",
    "scanner",
    "that",
    "mirrors",
    "tiktoken's",
    "regex",
    "boundaries",
    "It's",
    "faster",
    "because",
    "there's",
    "no",
    "backtracking",
    "engine",
    "involved",
    "so",
    "we'll",
    "keep",
    "it",
    "that",
    "way",
    "Zürich",
    "café",
    "naïve",
    "東京",
    "مرحبا",
    "🙂",
    "42",
    "1999",
    "3.14159",
    "$1,234.56",
    "100%",
    "user@example.com",
    "https://example.com/a/b?c=d",
    "C++",
    "F#",
    "node.js",
    "v1.2.3",
    "(parens)",
    "[brackets]",
    "{braces}",
    "<tags>",
    '"quotes"',
    "'single'",
    "don't",
    "WON'T",
    "I'M",
    "They'RE",
)


def random_text(rng: random.Random) -> str:
    return "".join(rng.choice(ALPHABET) for _ in range(rng.randrange(0, 40)))


def paragraph(rng: random.Random, words: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(words))


def short_paragraphs(rng: random.Random) -> Iterator[str]:
    while True:
        content = paragraph(rng, rng.randrange(60, 140))
        if len(content) < TIKTOKEN_ENCODE_CHUNK_SIZE_CHARS:
            yield content


def chat_body(rng: random.Random, target_tokens: int) -> dict[str, object]:
    candidates: Final = tuple(itertools.islice(short_paragraphs(rng), 2000))
    running: Final = tuple(itertools.accumulate(len(ENCODING.encode(content)) + 3 for content in candidates))
    turns: Final = next(index for index, total in enumerate(running) if total >= target_tokens) + 1
    contents: Final = candidates[: turns + (turns % 2)]
    return {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Answer precisely and cite sources."},
            *(
                {"role": "user" if index % 2 == 0 else "assistant", "content": content}
                for index, content in enumerate(contents)
            ),
            {"role": "user", "content": "Summarise the conversation so far in three sentences."},
        ],
    }


TOOLS: Final = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    "days": {"type": "integer"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "opts": {
                        "type": "object",
                        "properties": {"verbose": {"type": "boolean"}, "level": {"type": "integer", "enum": [1, 2]}},
                        "required": ["verbose"],
                    },
                    "anything": {},
                },
                "required": ["location"],
            },
        },
    },
    {"type": "function", "function": {"name": "noop"}},
]

SMALL_REQUESTS: Final = (
    {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello, how are you today?"}]},
    {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are a terse assistant."},
            {
                "role": "user",
                "name": "alice",
                "content": [
                    {"type": "text", "text": "Summarise this paragraph about ships and harbours."},
                    "plain string item",
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Sure."}]},
        ],
    },
    {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": TOOLS,
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
    },
    {
        "model": "gpt-4",
        "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "weather?"}],
        "tools": TOOLS,
        "tool_choice": "none",
    },
    {"model": "gpt-4", "prompt": "Write a haiku about ships."},
    {"model": "gpt-4", "prompt": ["first prompt", "second prompt"]},
    {
        "model": "gpt-4",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": 'Summarise caf\u00e9 menus, na\u00efve \u2014 ok? "quoted"\n'}
                ],
            },
            {"role": "assistant", "content": "Sure."},
        ],
        "instructions": "be terse",
    },
    {"model": "gpt-4", "input": [[101, 2023, 5], [7]], "encoding_format": "float"},
    {
        "model": "gpt-4",
        "query": "best harbour",
        "documents": [
            "doc one",
            {"text": "doc two", "title": "T", "n": 3, "ok": True, "none": None, "tags": ["a", "b"]},
        ],
    },
)


def main() -> None:
    rng: Final = random.Random(2026)
    texts: Final = tuple(CORPUS) + tuple(random_text(rng) for _ in range(3000))
    with (HERE / "texts.jsonl").open("w", encoding="utf-8") as handle:
        for text in texts:
            tokens = len(ENCODING.encode(text, disallowed_special=()))
            pieces = SPLIT_PATTERN.findall(text)
            handle.write(json.dumps({"text": text, "tokens": tokens, "pieces": pieces}, ensure_ascii=False) + "\n")
    bodies: Final = tuple(SMALL_REQUESTS) + (chat_body(rng, 50_000),)
    with (HERE / "requests.jsonl").open("w", encoding="utf-8") as handle:
        for body in bodies:
            input_tokens = _count_input_tokens(dict(body), "gpt-4")
            assert input_tokens is not None
            handle.write(json.dumps({"body": json.dumps(body), "input_tokens": input_tokens}) + "\n")


if __name__ == "__main__":
    main()
