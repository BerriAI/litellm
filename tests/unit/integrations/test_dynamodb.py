from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Final
from unittest.mock import MagicMock

import pytest

import litellm
from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY
from litellm.integrations.dynamodb import DyanmoDBLogger
from litellm.litellm_core_utils.internal_call_metadata import BACKGROUND_RESPONSE_COST_REQUEST_ID_SUFFIX
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    SETTLES_BACKGROUND_RESPONSE_KWARG,
    BackgroundResponseSettlement,
)
from litellm.types.utils import BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN

BACKGROUND_COST_POLL_KWARGS: Final[Mapping[str, object]] = {
    "call_type": "aget_responses",
    "litellm_params": {"metadata": {INTERNAL_CALL_ORIGIN_METADATA_KEY: BACKGROUND_RESPONSE_COST_POLL_CALL_ORIGIN}},
}
PASSTHROUGH_SETTLEMENT_KWARGS: Final[Mapping[str, object]] = {
    "call_type": "pass_through_endpoint",
    "litellm_params": {"metadata": {}},
    SETTLES_BACKGROUND_RESPONSE_KWARG: BackgroundResponseSettlement.SETTLES,
}


@pytest.mark.parametrize(
    "cost_kwargs",
    (
        pytest.param(BACKGROUND_COST_POLL_KWARGS, id="cost-poll"),
        pytest.param(PASSTHROUGH_SETTLEMENT_KWARGS, id="passthrough-settlement"),
    ),
)
@pytest.mark.usefixtures("sdk_only_install")
def test_background_cost_event_keeps_its_own_item_beside_its_create(
    monkeypatch: pytest.MonkeyPatch, cost_kwargs: Mapping[str, object]
) -> None:
    monkeypatch.setattr(litellm, "dynamodb_table_name", "litellm-logs")
    dynamodb: Final = MagicMock()
    logger: Final = DyanmoDBLogger(dynamodb=dynamodb)
    response_obj: Final = {"id": "resp_YWJjZA", "usage": {"total_tokens": 3}}
    now: Final = datetime.now(timezone.utc)
    create_kwargs: Final[Mapping[str, object]] = {"call_type": "aresponses", "litellm_params": {"metadata": {}}}

    for kwargs in (create_kwargs, cost_kwargs):
        logger.log_event(kwargs, response_obj, now, now, print_verbose=print)

    dynamodb.Table.assert_called_with("litellm-logs")
    written_ids: Final = [call.kwargs["Item"]["id"] for call in dynamodb.Table.return_value.put_item.call_args_list]
    assert written_ids == ["resp_YWJjZA", f"resp_YWJjZA{BACKGROUND_RESPONSE_COST_REQUEST_ID_SUFFIX}"]
