from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from litellm.constants import MCP_DATA_BOUNDARY_VIOLATION_CODE

DataBoundaryPolicySource: TypeAlias = Literal["key", "team", "user"]
DataBoundaryPolicy: TypeAlias = tuple[DataBoundaryPolicySource, Sequence[str] | None]


def normalize_data_boundary(value: str | None) -> str | None:
    if value is None:
        return None
    normalized: Final = value.strip().lower()
    return normalized or None


@dataclass(frozen=True)
class DataBoundaryViolation:
    server_name: str
    server_data_boundary: str | None
    allowed_data_boundaries: tuple[str, ...]
    policy_source: DataBoundaryPolicySource

    @property
    def reason(self) -> str:
        location: Final = (
            f"data boundary '{self.server_data_boundary}'"
            if self.server_data_boundary is not None
            else "no declared data boundary"
        )
        allowed: Final = ", ".join(f"'{boundary}'" for boundary in self.allowed_data_boundaries)
        return (
            f"MCP data boundary violation: server '{self.server_name}' is in {location}, "
            f"but the {self.policy_source} policy only permits data boundaries [{allowed}]. "
            "Contact proxy admin to change the data boundary policy."
        )

    def to_detail(self) -> Mapping[str, str | tuple[str, ...] | None]:
        return {
            "error": self.reason,
            "code": MCP_DATA_BOUNDARY_VIOLATION_CODE,
            "server_name": self.server_name,
            "server_data_boundary": self.server_data_boundary,
            "allowed_data_boundaries": self.allowed_data_boundaries,
            "policy_source": self.policy_source,
        }


def find_data_boundary_violation(
    server_name: str,
    server_data_boundary: str | None,
    policies: Sequence[DataBoundaryPolicy],
) -> DataBoundaryViolation | None:
    normalized_server_boundary: Final = normalize_data_boundary(server_data_boundary)
    for policy_source, allowed_boundaries in policies:
        if not allowed_boundaries:
            continue
        normalized_allowed: frozenset[str] = frozenset(
            boundary for boundary in map(normalize_data_boundary, allowed_boundaries) if boundary is not None
        )
        if normalized_server_boundary is None or normalized_server_boundary not in normalized_allowed:
            return DataBoundaryViolation(
                server_name=server_name,
                server_data_boundary=normalized_server_boundary,
                allowed_data_boundaries=tuple(sorted(normalized_allowed)),
                policy_source=policy_source,
            )
    return None


def violation_unless_any_source_admits(
    source_results: Sequence[DataBoundaryViolation | None],
) -> DataBoundaryViolation | None:
    if not source_results or any(result is None for result in source_results):
        return None
    return source_results[0]


def data_boundary_policy_unavailable_detail(server_name: str) -> Mapping[str, str]:
    return {
        "error": (
            f"MCP data boundary policy for server '{server_name}' could not be loaded, so the request is denied. "
            "Contact proxy admin to check the data boundary policy."
        ),
        "code": MCP_DATA_BOUNDARY_VIOLATION_CODE,
        "server_name": server_name,
    }
