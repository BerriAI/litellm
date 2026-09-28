import pytest
from pydantic import ValidationError

from litellm.types.proxy.management_endpoints.scim_agent_provisioning import SCIM_AGENT_USER_SCHEMA
from litellm.types.proxy.management_endpoints.scim_v2 import SCIMUser


@pytest.mark.parametrize(
    "value,expected",
    [
        ("ABCDEF00-1234-4234-9234-123456789ABC", "abcdef00-1234-4234-9234-123456789abc"),
        ("opaque-directory-id", "opaque-directory-id"),
    ],
)
def test_directory_ids_normalize_uuids_without_rewriting_opaque_ids(value: str, expected: str) -> None:
    from litellm.types.proxy.management_endpoints.scim_agent_provisioning import canonical_directory_id

    assert canonical_directory_id(value) == expected


def test_agent_schema_without_parent_is_not_a_human() -> None:
    with pytest.raises(ValidationError, match="identityParentId"):
        SCIMUser.model_validate({"schemas": [SCIM_AGENT_USER_SCHEMA], "userName": "agent@example.com"})

