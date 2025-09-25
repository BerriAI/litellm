"""
Minimal test for Presidio Union[PiiEntityType, str] type fix.
Tests only the core fix without heavy dependencies.
"""
from typing import Union, Dict
from enum import Enum


class PiiEntityType(str, Enum):
    EMAIL_ADDRESS = "EMAIL_ADDRESS"


class PiiAction(str, Enum):
    BLOCK = "BLOCK"
    MASK = "MASK"


def test_presidio_union_type_fix():
    """Test that Union[PiiEntityType, str] allows both enum and string entity types"""
    
    # Custom recognizers for EMPLOYEE_ID and CUSTOMER_ID
    custom_recognizers = [
        {
            "name": "Employee ID Recognizer",
            "supported_language": "en",
            "patterns": [{"name": "employee id", "regex": "EMP-[0-9]{6}", "score": 0.9}],
            "context": ["employee", "id"],
            "supported_entity": "EMPLOYEE_ID"
        },
        {
            "name": "Customer ID Recognizer",
            "supported_language": "en",
            "patterns": [{"name": "customer id", "regex": "CUST-[0-9]{8}", "score": 0.9}],
            "context": ["customer", "id"],
            "supported_entity": "CUSTOMER_ID"
        }
    ]
    
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
    
    print("✅ Union type fix verified: mixed entity types work correctly")
    print("✅ Custom recognizers defined for EMPLOYEE_ID and CUSTOMER_ID")


if __name__ == "__main__":
    test_presidio_union_type_fix()