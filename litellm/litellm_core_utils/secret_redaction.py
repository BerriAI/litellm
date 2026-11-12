"""
Credential/secret redaction utilities.

This module owns the compiled regex and the public `redact_string` helper so
that any part of the codebase (logging, exception mapping, etc.) can scrub
secrets from strings without depending on the logging-configuration module.
"""

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from litellm.constants import MINIMUM_CUSTOM_KEY_LENGTH
from litellm.rust_bridge import diagnostics

REDACTED: Final = "REDACTED"

_PEM_PRIVATE_KEY_BLOCK: Final = r"-----BEGIN[A-Z \-]*PRIVATE KEY-----[\s\S]*?-----END[A-Z \-]*PRIVATE KEY-----"


def _build_secret_patterns() -> "re.Pattern[str]":
    patterns: Final[list[str]] = [
        # PEM private key / certificate blocks
        _PEM_PRIVATE_KEY_BLOCK,
        # GCP OAuth2 access tokens (ya29.*)
        r"\bya29\.[A-Za-z0-9_.~+/-]+",
        # Credential %s formatting (space separator, no key= prefix)
        r"(?:client_secret|azure_password|azure_username)\s+[^\s,'\"})\]{}>]+",
        # AWS access key IDs
        r"(?:AKIA|ASIA)[0-9A-Z]{16}",
        # Bearer tokens (OAuth, JWT, etc.)
        r"Bearer\s+[A-Za-z0-9\-._~+/]{10,}=*",
        # Basic auth headers
        r"Basic\s+[A-Za-z0-9+/]{10,}={0,2}",
        # OpenAI / Anthropic sk- prefixed keys
        rf"sk-[A-Za-z0-9\-_]{{{MINIMUM_CUSTOM_KEY_LENGTH - len('sk-')},}}",
        # Credentials passed as URL query params. Terminated by "&" like the key=
        # and sig= patterns below, so the rest of the request line survives in an
        # access log. Must precede the generic patterns to win at the same position.
        r"(?<=[?&])(?:api[_-]?key|\w*(?:token|password|passwd|client_secret|secret_key|_secret))"
        r"=[^\s&'\"]+",
        # Generic api_key / api-key / apikey (handles 'key': 'value' dict repr)
        r"(?:api[_-]?key)['\"]?\s*[:=]\s*['\"]?[^\s,'\"})\]{}>]{8,}",
        # x-api-key / api-key header values (handles 'key': 'value' dict repr)
        r"(?:x-api-key|api-key)['\"]?\s*[:=]\s*['\"]?[^\s,'\"})\]{}>]+",
        # Anthropic internal header keys
        r"x-ak-[A-Za-z0-9\-_]{20,}",
        # Google API keys (bare key value)
        r"AIza[0-9A-Za-z\-_]{35}",
        # URL query-param key=VALUE (e.g. ?key=AIza... or &key=...) — catches the
        # full "key=<secret>" fragment so the value is redacted regardless of format.
        r"(?<=[?&])key=[^\s&'\"]{8,}",
        # Password / secret params (handles key=value and 'key': 'value')
        # Word boundary prevents O(n^2) backtracking on long word-char runs.
        r"(?:^|(?<=\W))\w*(?:password|passwd|client_secret|secret_key|_secret)"
        r"['\"]?\s*[:=]\s*['\"]?[^\s,'\"})\]{}>]+",
        # Database connection string credentials (scheme://user:pass@host).
        # The user half stops at the ":" separator and both halves are length-capped,
        # so a long attacker-supplied URL cannot backtrack quadratically.
        r"(?<=://)[^\s'\":]{0,4096}:[^\s'\"]{1,4096}(?=@)",
        # Databricks personal access tokens
        r"dapi[0-9a-f]{32}",
        # Module-level provider keys logged as litellm.<provider>_key=<value>
        r"litellm\.[A-Za-z0-9_]*_key['\"]?\s*[:=]\s*['\"]?[^\s,'\"})\]{}>]+",
        # ── Key-name-based redaction ──
        # Catches secrets inside dicts/config dumps by matching on the KEY name
        # regardless of what the value looks like.
        # e.g. 'master_key': 'any-value-here', "database_url": "postgres://..."
        # private_key with PEM-aware value capture
        r"""private_key['\"]?\s*[:=]\s*['\"]?(?:-----BEGIN[A-Z \-]*PRIVATE KEY-----[\s\S]*?-----END[A-Z \-]*PRIVATE KEY-----|[^\s,'\"})\]{}>]+)""",
        r"(?:master_key|xai_key|database_url|db_url|connection_string|"
        r"aws_secret_access_key|aws_session_token|aws_access_key_id|s3_secret_access_key|s3_access_key_id|"
        r"signing_key|encryption_key|"
        r"auth_token|access_token|refresh_token|"
        r"slack_webhook_url|webhook_url|"
        r"database_connection_string|"
        r"huggingface_token|jwt_secret)"
        r"""['\"]?\s*[:=]\s*['\"]?[^\s,'\"})\]{}>]+""",
        # Raw JWTs (without Bearer prefix)
        r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*",
        # Azure SAS tokens in URLs. The delimiter is a lookbehind, like the
        # `key=` pattern above, so the `?` or `&` survives and the redacted URL
        # stays well formed (this string is often a request line in a log).
        r"(?<=[?&])sig=[A-Za-z0-9%+/=]+",
        # Full JSON service-account blobs (single-line and multi-line)
        r'\{[^{}]*"type"\s*:\s*"service_account"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
    ]
    return re.compile("|".join(patterns), re.IGNORECASE)


_SECRET_RE: Final = _build_secret_patterns()


def python_redact_string(value: str) -> str:
    return _SECRET_RE.sub(REDACTED, value)


_python_redact_string = python_redact_string


def redact_string(value: str) -> str:
    """Scrub known secret/credential patterns from *value* and return the result."""
    return diagnostics.run(lambda native: native.redact_text(value), lambda: python_redact_string(value))


_TOKEN_SHAPES: Final = (
    r"ya29\.[A-Za-z0-9_.~+/-]{20,}",
    r"(?:AKIA|ASIA)[0-9A-Z]{16}\b",
    rf"sk-[A-Za-z0-9\-_]{{{MINIMUM_CUSTOM_KEY_LENGTH - len('sk-')},}}",
    r"(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{24,}",
    r"AIza[0-9A-Za-z\-_]{35}",
    r"dapi[0-9a-f]{32}\b",
    r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}",
    r"github_pat_[A-Za-z0-9_]{82}",
    r"glpat-[A-Za-z0-9\-_]{20,}",
    r"lin_api_[A-Za-z0-9]{40}\b",
    r"xox[abposr]-[A-Za-z0-9-]{10,}",
    r"hf_[A-Za-z0-9]{34,}",
    r"npm_[A-Za-z0-9]{36}\b",
    r"pypi-AgEIcHlwaS5vcmc[A-Za-z0-9\-_]{50,}",
    r"SG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}\b",
    r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]+",
)
# Tokens start at a word boundary or right after a JSON escape like "\n", since tool output is often serialized JSON
_TOKEN_START: Final = r"(?:(?<!\w)|(?<=\\[nrt]))"
TOKEN_REDACTION_WINDOW_CHARS: Final = 16_384
_TOKEN_PREFIX_OVERLAP_CHARS: Final = 128
_TOKEN_RULE_RE: Final = re.compile(r"(.*)(\[[^\]]+\])(\{[0-9]+(?:,[0-9]*)?\}|\+)(\\b)?")
_TOKEN_WORD_RE: Final = re.compile(r"\w")
_JWT_RUN_RE: Final = re.compile(r"[A-Za-z0-9_-]+")
_PEM_LABEL_RE: Final = re.compile(r"[A-Z \-]+")
_PEM_FOOTER_LABEL_RE: Final = re.compile(r"[A-Z -]+")
_PEM_KEY_SUFFIX: Final = "PRIVATE KEY-----"
_PEM_END_PREFIX: Final = "-----END"


@dataclass(frozen=True, slots=True)
class _TokenRule:
    prefix: str
    body: re.Pattern[str]
    minimum: int
    maximum: int | None
    boundary: bool


def _token_rule(shape: str) -> _TokenRule:
    parsed: Final = _TOKEN_RULE_RE.fullmatch(shape)
    assert parsed is not None
    prefix, alphabet, quantifier, boundary = parsed.groups()
    counts: Final = quantifier.strip("{}").split(",")
    return _TokenRule(
        prefix=prefix,
        body=re.compile(f"{alphabet}+"),
        minimum=1 if quantifier == "+" else int(counts[0]),
        maximum=None if quantifier == "+" or counts[-1] == "" else int(counts[-1]),
        boundary=boundary is not None,
    )


_TOKEN_RULES: Final = tuple(_token_rule(shape) for shape in _TOKEN_SHAPES[:-1])
_TOKEN_RE: Final = re.compile(
    r"(?P<pem>-----BEGIN)|"
    + _TOKEN_START
    + "(?:"
    + "|".join(f"(?P<t{index}>{rule.prefix})" for index, rule in enumerate(_TOKEN_RULES))
    + r"|(?P<jwt>eyJ))"
)


def _find_marker(value: str, marker: str, start: int, stop: int, *, reverse: bool = False) -> int:
    step: Final = TOKEN_REDACTION_WINDOW_CHARS - len(marker) + 1
    windows: Final = range(stop, start, -step) if reverse else range(start, stop, step)
    for edge in windows:
        if reverse:
            if (found := value.rfind(marker, max(start, edge - TOKEN_REDACTION_WINDOW_CHARS), edge)) >= 0:
                return found
        elif (found := value.find(marker, edge, min(stop, edge + TOKEN_REDACTION_WINDOW_CHARS))) >= 0:
            return found
    return -1


class _TokenScanner:
    def __init__(self, value: str) -> None:
        self.value = value
        self._runs: Mapping[re.Pattern[str], tuple[tuple[int, int], ...]] = MappingProxyType({})
        self._pem_header_suffix: tuple[int, int] | None = None
        self._pem_footer_suffix: tuple[int, int] | None = None
        self._pem_first_header: tuple[int, int] | None = None
        self._pem_inner_footer: tuple[int, int] | None = None
        self._missing_pem_footer_from: int | None = None

    def _run_end(self, pattern: re.Pattern[str], start: int) -> int:
        cached: Final = next((end for begin, end in self._runs.get(pattern, ()) if begin <= start <= end), None)
        if cached is not None:
            return cached
        end: int = start  # rebind-ok: cursor advances through bounded regex windows
        while end < len(self.value):
            if (
                match := pattern.match(self.value, end, min(len(self.value), end + TOKEN_REDACTION_WINDOW_CHARS))
            ) is None:
                break
            end = match.end()
        self._runs = {**self._runs, pattern: (*self._runs.get(pattern, ()), (start, end))[-3:]}
        return end

    def _pem_suffix_end(self, start: int, stop: int, *, header: bool = False) -> int | None:
        cached: Final = self._pem_header_suffix if header else self._pem_footer_suffix
        suffix_start: Final = (
            cached[1]
            if cached is not None and cached[0] == stop
            else _find_marker(self.value, _PEM_KEY_SUFFIX, start, stop, reverse=True)
        )
        if header:
            self._pem_header_suffix = (stop, suffix_start)
        else:
            self._pem_footer_suffix = (stop, suffix_start)
        return suffix_start + len(_PEM_KEY_SUFFIX) if suffix_start >= start else None

    def _pem_footer_end(self, start: int) -> int | None:
        if self._missing_pem_footer_from is not None and start >= self._missing_pem_footer_from:
            return None
        cursor: int = start  # rebind-ok: cursor advances past each inspected footer label
        while (marker := _find_marker(self.value, _PEM_END_PREFIX, cursor, len(self.value))) >= 0:
            if (
                end := self._pem_suffix_end(
                    marker + len(_PEM_END_PREFIX), self._run_end(_PEM_FOOTER_LABEL_RE, marker + len(_PEM_END_PREFIX))
                )
            ) is not None:
                return end
            cursor = self._run_end(_PEM_FOOTER_LABEL_RE, marker + len(_PEM_END_PREFIX))
        self._missing_pem_footer_from = start
        return None

    def _pem_end(self, start: int) -> int | None:
        header_end: Final = self._pem_suffix_end(start, self._run_end(_PEM_LABEL_RE, start), header=True)
        if header_end is None:
            return None
        footer_end: Final = self._pem_footer_end(header_end)
        if footer_end is not None:
            return footer_end
        first_header: Final = (
            self._pem_first_header[1]
            if self._pem_first_header is not None
            and self._pem_first_header[0] == header_end
            and self._pem_first_header[1] >= start
            else _find_marker(self.value, _PEM_KEY_SUFFIX, start, header_end)
        )
        self._pem_first_header = (header_end, first_header)
        inner_footer: Final = (
            self._pem_inner_footer[1]
            if self._pem_inner_footer is not None and self._pem_inner_footer[0] == header_end
            else _find_marker(self.value, _PEM_END_PREFIX, start, header_end - len(_PEM_KEY_SUFFIX), reverse=True)
        )
        self._pem_inner_footer = (header_end, inner_footer)
        return header_end if inner_footer >= first_header + len(_PEM_KEY_SUFFIX) else None

    def _jwt_end(self, start: int) -> int | None:
        header_end: Final = self._run_end(_JWT_RUN_RE, start)
        if header_end - start < 10 or not self.value.startswith(".eyJ", header_end):
            return None
        payload_start: Final = header_end + len(".eyJ")
        payload_end: Final = self._run_end(_JWT_RUN_RE, payload_start)
        if payload_end - payload_start < 10 or not self.value.startswith(".", payload_end):
            return None
        signature_start: Final = payload_end + 1
        signature_end: Final = self._run_end(_JWT_RUN_RE, signature_start)
        return signature_end if signature_end > signature_start else None

    def _token_end(self, match: re.Match[str]) -> int | None:
        if match.lastgroup == "pem":
            return self._pem_end(match.end())
        if match.lastgroup == "jwt":
            return self._jwt_end(match.end())
        assert match.lastgroup is not None
        rule: Final = _TOKEN_RULES[int(match.lastgroup[1:])]
        body: Final = (
            rule.body.match(self.value, match.end(), min(len(self.value), match.end() + rule.maximum))
            if rule.maximum is not None
            else None
        )
        end: Final = body.end() if body is not None else self._run_end(rule.body, match.end())
        if end - match.end() < rule.minimum or (rule.maximum is not None and end - match.end() != rule.maximum):
            return None
        if rule.boundary and bool(_TOKEN_WORD_RE.match(self.value, end - 1, end)) == bool(
            _TOKEN_WORD_RE.match(self.value, end, min(len(self.value), end + 1))
        ):
            return None
        return end

    def _next_token(self, start: int) -> re.Match[str] | None:
        for left in range(start, len(self.value), TOKEN_REDACTION_WINDOW_CHARS - _TOKEN_PREFIX_OVERLAP_CHARS):
            if (
                match := _TOKEN_RE.search(self.value, left, min(len(self.value), left + TOKEN_REDACTION_WINDOW_CHARS))
            ) is not None:
                return match
        return None

    def _text(self, start: int, stop: int) -> Iterator[str]:
        for left in range(start, stop, TOKEN_REDACTION_WINDOW_CHARS):
            yield self.value[left : min(stop, left + TOKEN_REDACTION_WINDOW_CHARS)]

    def pieces(self, replacement: str) -> Iterator[str]:
        cursor: int = 0  # rebind-ok: scan cursor advances independently of the last replacement
        consumed: int = 0  # rebind-ok: output cursor advances after each replacement
        while (match := self._next_token(cursor)) is not None:
            if (end := self._token_end(match)) is None:
                cursor = match.start() + 1
                continue
            yield from self._text(consumed, match.start())
            yield replacement
            cursor = end
            consumed = end
        yield from self._text(consumed, len(self.value))


def redact_tokens(value: str, replacement: str = REDACTED) -> str:
    """Replace substrings shaped like a credential (provider key prefixes, JWTs, PEM private keys).

    Unlike redact_string() this never matches on surrounding words such as ``password:`` or ``Basic``,
    so it is safe to run over free-form prompt and tool content.
    """
    expanded_replacement: Final = re.sub("", replacement, "", count=1)
    return "".join(_TokenScanner(value).pieces(expanded_replacement))


_UNIX_SYSTEM_PATH: Final = r"/(?:etc|var|opt|usr|home|root|private|Users|tmp|mnt|srv)/[^\s'\"\)\]}>,]+"
_WINDOWS_DRIVE_PATH: Final = r"[A-Za-z]:\\[^\s'\"\)\]}>,]+"
_PRIVATE_OR_LOOPBACK_IPV4: Final = (
    r"\b(?:10(?:\.\d{1,3}){3}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}|192\.168(?:\.\d{1,3}){2}|127(?:\.\d{1,3}){3})\b"
)
_INTERNAL_SUFFIX_HOSTNAME: Final = r"\b[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.(?:internal|local|corp|lan|intra|private)\b"
_INTERNAL_DETAIL_RE: Final = re.compile(
    "|".join((_UNIX_SYSTEM_PATH, _WINDOWS_DRIVE_PATH, _PRIVATE_OR_LOOPBACK_IPV4, _INTERNAL_SUFFIX_HOSTNAME)),
    re.IGNORECASE,
)
_TRACEBACK_MARKER: Final = "Traceback (most recent call last):"


def _python_redact_internal_details(value: str) -> str:
    """Drop an embedded traceback and scrub filesystem paths and internal hostnames,
    on top of redact_string(). For client-facing messages only: server logs keep this detail."""
    marker_index: Final = value.find(_TRACEBACK_MARKER)
    without_traceback: Final = value[:marker_index].rstrip() if marker_index != -1 else value
    return _INTERNAL_DETAIL_RE.sub(REDACTED, python_redact_string(without_traceback))


def redact_internal_details(value: str) -> str:
    return diagnostics.run(
        lambda native: native.redact_client_message(value), lambda: _python_redact_internal_details(value)
    )


def python_redact_structured_value(key: str | None, value: str) -> str:
    """Scrub *value* as it appeared under *key* inside a structured record.

    redact_string() replaces a whole ``key: value`` span with REDACTED, which is
    fine inside free text but destroys the surrounding syntax when the span is a
    JSON member rather than message content. This renders the pair the way a dict
    repr would, so the key-name patterns still fire, but collapses only the value
    so the caller's structure survives.
    """
    scrubbed: Final = python_redact_string(value)
    if scrubbed != value or key is None:
        return scrubbed
    rendered: Final = f"'{key}': '{value}'"
    return REDACTED if python_redact_string(rendered) != rendered else value


_python_redact_structured_value = python_redact_structured_value


def redact_structured_value(key: str | None, value: str) -> str:
    return diagnostics.run(
        lambda native: native.redact_structured_text(key, value),
        lambda: python_redact_structured_value(key, value),
    )
