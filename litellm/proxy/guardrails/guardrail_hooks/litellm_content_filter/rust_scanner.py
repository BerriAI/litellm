"""Rust-accelerated screening for the content filter hot path.

The Python content filter screens every request by running ``re.search`` once
per keyword and once per regex pattern, under the GIL. This wraps the Rust
``ContentScanner`` (one Aho-Corasick pass over the literal keywords plus one
``RegexSet`` pass over the patterns, with the GIL released) and maps the
matches back to the keyword/pattern keys the guardrail already uses.
"""

import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple


def _load_scanner_factory() -> Callable[[str], object]:
    """Return the compiled ``ContentScanner`` constructor, raising ImportError
    when the optional ``litellm_python_bridge`` extension is not installed so the
    caller can fall back to pure-Python screening."""
    import litellm_python_bridge

    return litellm_python_bridge.ContentScanner


@dataclass
class ScanResult:
    """Which terms the scanner found in one text."""

    category_keywords_present: Set[str] = field(default_factory=set)
    always_block_present: Set[str] = field(default_factory=set)
    blocked_words_present: Set[str] = field(default_factory=set)
    # compiled_patterns index -> match spans (byte offsets)
    pattern_spans: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)


# Namespace tags so a single integer id space maps back to the right table.
_KIND_CATEGORY = "category"
_KIND_ALWAYS_BLOCK = "always_block"
_KIND_BLOCKED = "blocked"
_KIND_PATTERN = "pattern"


class ContentFilterScanner:
    def __init__(
        self,
        scanner: object,
        id_lookup: Dict[int, Tuple[str, object]],
        fallback_pattern_indexes: Set[int],
    ):
        self._scanner = scanner
        self._id_lookup = id_lookup
        # compiled_patterns indexes the scanner could NOT take (contextual extra
        # config, or a regex Rust could not compile); the caller screens these
        # itself. This is a per-pattern correctness fallback, not a fallback for
        # the Rust package being absent.
        self.fallback_pattern_indexes = fallback_pattern_indexes

    @classmethod
    def build(
        cls,
        category_keywords: Dict[str, object],
        always_block_keywords: Dict[str, object],
        blocked_words: Dict[str, object],
        compiled_patterns: List[Dict[str, object]],
        keyword_to_regex: Callable[[str], str],
        scanner_factory: Optional[Callable[[str], object]] = None,
    ) -> "ContentFilterScanner":
        """Build a scanner from the guardrail's compiled config.

        scanner_factory is injectable for tests; it defaults to the Rust wheel's
        ContentScanner constructor (a JSON string in, a scanner out).
        """
        if scanner_factory is None:
            scanner_factory = _load_scanner_factory()
        literals: List[dict] = []
        regexes: List[dict] = []
        id_lookup: Dict[int, Tuple[str, object]] = {}
        next_id = 0

        def add_keyword(keyword: str, kind: str, key: object) -> None:
            nonlocal next_id
            term_id = next_id
            next_id += 1
            id_lookup[term_id] = (kind, key)
            multi_word = " " in keyword
            if "*" in keyword:
                # Wildcard keyword is a pattern, not a literal.
                core = keyword_to_regex(keyword)
                pattern = core if multi_word else rf"\b{core}\b"
                regexes.append({"id": term_id, "pattern": pattern})
            else:
                literals.append(
                    {
                        "id": term_id,
                        "text": keyword,
                        "word_boundary": not multi_word,
                    }
                )

        for keyword in category_keywords:
            add_keyword(keyword, _KIND_CATEGORY, keyword)
        for keyword in always_block_keywords:
            add_keyword(keyword, _KIND_ALWAYS_BLOCK, keyword)
        for keyword in blocked_words:
            # The hot-path blocked-word loop uses re.search(escape(kw)) with no
            # word boundary, i.e. substring matching.
            add_keyword(keyword, _KIND_BLOCKED, keyword)

        fallback_pattern_indexes: Set[int] = set()
        for index, entry in enumerate(compiled_patterns):
            if entry.get("keyword_regex") is not None or entry.get(
                "allow_word_numbers"
            ):
                # Contextual extras are not modeled by the scanner; screen in Python.
                fallback_pattern_indexes.add(index)
                continue
            term_id = next_id
            next_id += 1
            id_lookup[term_id] = (_KIND_PATTERN, index)
            regexes.append({"id": term_id, "pattern": entry["regex"].pattern})

        scanner = scanner_factory(
            json.dumps({"literals": literals, "regexes": regexes})
        )

        # Any regex Rust could not compile (lookaround, backrefs) falls back to
        # Python screening for that pattern only.
        for err in json.loads(getattr(scanner, "compile_errors", "[]")):
            kind_key = id_lookup.get(err["id"])
            if kind_key is not None and kind_key[0] == _KIND_PATTERN:
                fallback_pattern_indexes.add(kind_key[1])

        return cls(scanner, id_lookup, fallback_pattern_indexes)

    def scan(self, text: str) -> ScanResult:
        result = ScanResult()
        for term_id, start, end in self._scanner.scan(text):
            kind_key = self._id_lookup.get(term_id)
            if kind_key is None:
                continue
            kind, key = kind_key
            if kind == _KIND_CATEGORY:
                result.category_keywords_present.add(key)
            elif kind == _KIND_ALWAYS_BLOCK:
                result.always_block_present.add(key)
            elif kind == _KIND_BLOCKED:
                result.blocked_words_present.add(key)
            elif kind == _KIND_PATTERN:
                result.pattern_spans.setdefault(key, []).append((start, end))
        return result
