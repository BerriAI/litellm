from typing import Final

import pytest

from litellm._auth.failure import NativeAuthRuntimeError
from litellm.proxy.auth.model_policy_read import require_native_fact


@pytest.mark.parametrize("value", (False, 0, "", (), object()))
def test_read_precondition_preserves_falsy_and_opaque_facts(value: object) -> None:
    assert require_native_fact(value) is value


def test_missing_read_fact_is_terminal() -> None:
    with pytest.raises(NativeAuthRuntimeError) as raised:
        require_native_fact(None)
    error: Final = raised.value
    assert isinstance(error, RuntimeError)
