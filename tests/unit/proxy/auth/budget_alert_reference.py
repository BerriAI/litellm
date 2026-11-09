from collections.abc import Mapping

from pydantic import TypeAdapter, ValidationError

from litellm._auth.budget_alerts import Facts


def reference_recipients(raw: object) -> list[str]:
    if isinstance(raw, list):
        return [email.strip() for email in raw if isinstance(email, str) and email.strip()]
    if isinstance(raw, str):
        return [email.strip() for email in raw.split(",") if email.strip()]
    return []


def reference_alert(facts: Facts, global_config: object, local_config: object) -> tuple[bool, object]:
    if not facts.member and not (facts.limit is not None and facts.spend is not None and facts.spend > 0):
        return False, None
    if facts.member:
        try:
            config = TypeAdapter(Mapping[str, object]).validate_python(local_config)
        except ValidationError:
            config = {}
        local_config = {
            key: value for key, value in config.items() if key.isdigit() and len(key) <= 3 and 1 <= int(key) <= 100
        }
    global_entries = {key: reference_recipients(value) for key, value in (global_config or {}).items()}
    local_entries = {key: reference_recipients(value) for key, value in (local_config or {}).items()}
    merged = {
        key: list(dict.fromkeys(global_entries.get(key, []) + local_entries.get(key, [])))
        for key in set(global_entries) | set(local_entries)
    }
    if merged:
        if facts.member and facts.spend <= 0:
            return False, None
        minimum = min((int(key) for key in merged if key.isdigit()), default=None)
        if minimum is None or facts.spend < facts.limit * (minimum / 100.0):
            return False, None
        return True, tuple((key, tuple(value)) for key, value in merged.items())
    return (
        not facts.member and facts.spend >= facts.limit * facts.default_fraction and facts.spend < facts.limit
    ), None
