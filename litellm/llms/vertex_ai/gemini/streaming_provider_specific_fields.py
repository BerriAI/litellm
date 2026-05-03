from typing import Any, Dict, List, Optional


def _merge_server_side_tool_invocations(existing: Optional[Any], incoming: Any) -> Any:
    if not isinstance(incoming, list):
        return incoming

    if not isinstance(existing, list):
        existing = []

    merged_invocations: List[Any] = []
    invocations_by_id: Dict[str, Dict[str, Any]] = {}

    for invocation in existing:
        if not isinstance(invocation, dict):
            merged_invocations.append(invocation)
            continue

        invocation_copy = dict(invocation)
        invocation_id = invocation_copy.get("id")
        if isinstance(invocation_id, str) and invocation_id:
            invocations_by_id[invocation_id] = invocation_copy
        merged_invocations.append(invocation_copy)

    for invocation in incoming:
        if not isinstance(invocation, dict):
            merged_invocations.append(invocation)
            continue

        invocation_id = invocation.get("id")
        if isinstance(invocation_id, str) and invocation_id in invocations_by_id:
            existing_invocation = invocations_by_id[invocation_id]
            for key, value in invocation.items():
                if key not in existing_invocation or existing_invocation[key] is None:
                    existing_invocation[key] = value
            continue

        invocation_copy = dict(invocation)
        if isinstance(invocation_id, str) and invocation_id:
            invocations_by_id[invocation_id] = invocation_copy
        merged_invocations.append(invocation_copy)

    return merged_invocations


def merge_gemini_streaming_provider_specific_field(
    combined_provider_fields: Dict[str, Any], key: str, value: Any
) -> bool:
    if key == "server_side_tool_invocations":
        combined_provider_fields[key] = _merge_server_side_tool_invocations(
            combined_provider_fields.get(key), value
        )
        return True

    if key == "thought_signatures" and isinstance(value, list):
        existing = combined_provider_fields.get(key)
        if isinstance(existing, list):
            existing.extend(value)
        else:
            combined_provider_fields[key] = list(value)
        return True

    return False
