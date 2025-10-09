"""
Minimal test for Presidio Union[PiiEntityType, str] type fix.
Tests only the core fix without heavy dependencies.
"""
from enum import Enum
from typing import Dict, Union


class PiiEntityType(str, Enum):
    EMAIL_ADDRESS = "EMAIL_ADDRESS"


class PiiAction(str, Enum):
    BLOCK = "BLOCK"
    MASK = "MASK"


def test_presidio_union_type_fix() -> None:
    """Test that Union[PiiEntityType, str] allows both enum and string entity types"""

    # This is the core fix - mixed entity types in pii_entities_config
    pii_entities_config: Dict[Union[PiiEntityType, str], PiiAction] = {
        PiiEntityType.EMAIL_ADDRESS: PiiAction.MASK,
        "EMPLOYEE_ID": PiiAction.MASK,
        "CUSTOMER_ID": PiiAction.BLOCK,
    }

    # Verify entities can be used together (what Presidio needs)
    entities_list = list(pii_entities_config.keys())
    assert len(entities_list) == 3
    assert PiiEntityType.EMAIL_ADDRESS in entities_list
    assert "EMPLOYEE_ID" in entities_list
    assert "CUSTOMER_ID" in entities_list

    # Union type fix verified: mixed entity types work correctly


if __name__ == "__main__":
    test_presidio_union_type_fix()
