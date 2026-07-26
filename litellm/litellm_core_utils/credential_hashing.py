import base64
import hashlib
import json
import re

_SHA256_HEX_RE = re.compile(r"[a-fA-F0-9]{64}")
_BEARER_PREFIX = "bearer "


def is_valid_sha256_hash(value: str) -> bool:
    return bool(_SHA256_HEX_RE.fullmatch(value))


def _is_jwt(value: str) -> bool:
    header, _, rest = value.partition(".")
    payload, _, signature = rest.partition(".")
    if not header or not payload or not signature or "." in signature:
        return False
    try:
        decoded_header = json.loads(base64.urlsafe_b64decode(header + "=" * (-len(header) % 4)))
    except Exception:
        return False
    return isinstance(decoded_header, dict) and "alg" in decoded_header


def sanitize_key_hash(value: str | None) -> str | None:
    """
    Guarantee that a value destined for a `user_api_key_hash` field carries no
    credential material, since those fields land in durable sinks (spend logs,
    S3 request logs, Prometheus labels).

    Virtual keys (`sk-...`, optionally still `Bearer `-prefixed) and JWTs are
    replaced with their sha256 digest, matching the digest virtual keys are
    already stored and looked up under. Values that are already a digest, and
    non-credential identifiers such as the master key alias, pass through
    untouched so downstream grouping stays stable.
    """
    if value is None:
        return None
    credential = value[len(_BEARER_PREFIX) :] if value[: len(_BEARER_PREFIX)].lower() == _BEARER_PREFIX else value
    if is_valid_sha256_hash(credential):
        return credential
    if credential.startswith("sk-") or _is_jwt(credential):
        return hashlib.sha256(credential.encode()).hexdigest()
    return value
