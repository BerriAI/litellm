import importlib.util
from pathlib import Path
from typing import Final

import pytest

GUARD_PATH: Final = Path(__file__).resolve().parent / "check_no_none_string_error_fields.py"
_spec: Final = importlib.util.spec_from_file_location("check_no_none_string_error_fields", GUARD_PATH)
assert _spec is not None and _spec.loader is not None
guard: Final = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)


@pytest.mark.parametrize(
    "source",
    [
        pytest.param('raise ProxyException(message="m", type="None", param=None, code=500)\n', id="type_keyword"),
        pytest.param('raise ProxyException(message="m", type="x", param="None", code=500)\n', id="param_keyword"),
        pytest.param('ProxyException(message="m", type=getattr(e, "type", "None"), param=None)\n', id="type_getattr"),
        pytest.param('ProxyException(message="m", type="x", param=getattr(e, "param", "None"))\n', id="param_getattr"),
        pytest.param('error = {"message": "m", "type": "x", "param": "None"}\n', id="dict_literal"),
    ],
)
def test_a_none_string_error_field_is_reported_with_its_line(source: str) -> None:
    assert tuple(guard.violations("x = 1\n" + source, "proxy.py")) == ("proxy.py:2",)


@pytest.mark.parametrize(
    "source",
    [
        pytest.param('ProxyException(message="m", type="auth_error", param=None, code=401)\n', id="real_none_param"),
        pytest.param('ProxyException(message="None", type="x", param=getattr(e, "param", None))\n', id="none_message"),
        pytest.param('label = getattr(e, "provider", "None")\n', id="other_attribute"),
        pytest.param('row = {"model": "None"}\n', id="other_dict_key"),
    ],
)
def test_a_real_none_or_an_unrelated_none_string_passes(source: str) -> None:
    assert tuple(guard.violations(source, "proxy.py")) == ()
