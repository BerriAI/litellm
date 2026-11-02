from typing import cast

import httpx
import pytest

from litellm.llms.base_llm.decisions.transformation import (
    BaseDecisionsConfig,
    DecisionsResponseValidationError,
)
from litellm.types.decisions import DecisionsIRRequest


def test_transform_decisions_response_wraps_non_json_body():
    """Regression for #48660: a non-JSON response body raised an unhandled
    pydantic ValidationError because `validate_json` ran outside the
    try/except. It must surface as DecisionsResponseValidationError instead."""
    config = BaseDecisionsConfig()
    raw_response = httpx.Response(status_code=502, content=b"<html>Bad Gateway</html>")

    with pytest.raises(DecisionsResponseValidationError):
        config.transform_decisions_response(
            model="jev-latest",
            custom_llm_provider="systemone",
            raw_response=raw_response,
            request=cast(DecisionsIRRequest, None),
        )
