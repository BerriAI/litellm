from typing import Any, Callable

import pytest

import litellm


@pytest.mark.parametrize(
    ("helper", "expected"),
    [
        (litellm.get_supported_openai_params, None),
        (litellm.supports_function_calling, False),
        (litellm.supports_reasoning, False),
        (litellm.supports_response_schema, False),
    ],
)
def test_should_keep_quiet_for_unmapped_model_helpers(
    capsys: pytest.CaptureFixture[str],
    helper: Callable[..., Any],
    expected: Any,
) -> None:
    assert helper(model="hhh") is expected

    captured = capsys.readouterr()
    assert captured.out == ""
