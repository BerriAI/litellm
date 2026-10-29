from typing import Final

import pytest

from litellm.llms.eu_jev.common_utils import eu_jev_scoped_model, eu_jev_upstream_body


@pytest.mark.parametrize("model", [7, ["eu-jeff-latest"], False])
def test_a_non_string_client_model_reaches_upstream_unchanged(model: int | list[str] | bool) -> None:
    client_body: Final = {"model": model, "state": "x"}
    hook_body: Final = {"model": eu_jev_scoped_model(client_body), "state": "x"}

    assert eu_jev_upstream_body(hook_body, client_body) == client_body
