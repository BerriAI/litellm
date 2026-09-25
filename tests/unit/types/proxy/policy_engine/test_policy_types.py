import pytest
from pydantic import ValidationError

from litellm.types.proxy.policy_engine.policy_types import PolicyAttachment


@pytest.mark.parametrize("priority", [-2147483648, 2147483647])
def test_policy_attachment_accepts_int32_priority(priority: int):
    assert PolicyAttachment(policy="p", priority=priority).priority == priority


@pytest.mark.parametrize("priority", [-2147483649, 2147483648])
def test_policy_attachment_rejects_priority_outside_int32(priority: int):
    with pytest.raises(ValidationError):
        PolicyAttachment(policy="p", priority=priority)
