import hashlib
from collections.abc import MutableMapping
from typing import Final


def enforce_safety_identifier(
    *,
    data: MutableMapping[str, object],  # mutable-ok: trusted enforcement rewrites the request payload in place
    user_id: str | None,
    enabled: bool,
) -> bool:
    if not enabled:
        return False
    if user_id:
        safety_identifier: Final = hashlib.sha256(user_id.encode("utf-8")).hexdigest()
        if data.get("safety_identifier") == safety_identifier:
            return False
        data["safety_identifier"] = safety_identifier  # rebind-ok: enforce the trusted request identity in place
        return True
    if "safety_identifier" not in data:
        return False
    data.pop("safety_identifier", None)  # rebind-ok: remove the untrusted client value when no identity exists
    return True
