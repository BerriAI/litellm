"""The usage export carries the stored key and user metadata objects.

A customer charging usage back by cost center keeps that cost center in the
virtual key's metadata and the owner's user metadata. Before LIT-8524 the
Usage page export only carried the key alias, user id and email, so every
exported row needed a /key/info and /user/info round trip to recover the
metadata. The export now joins both objects onto each daily_with_keys row.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Final

import pytest
from e2e_config import provider_edge_base, unique_marker
from e2e_metadata import Domain, Mode, Provider, Subject, meta
from lifecycle import ResourceManager
from models import (
    ChatBody,
    ChatMessage,
    KeyGenerateBody,
    KeyMetadata,
    LiteLLMParamsBody,
    UserNewBody,
    UserNewResponse,
)
from pydantic import BaseModel, TypeAdapter
from spend_e2e_client import SpendClient, unwrap
from spend_reconciliation import BACKEND, INPUT_RATE, OUTPUT_RATE

pytestmark = pytest.mark.e2e

EXPORT_ROUTE: Final = "/user/daily/activity/export"


class UsageExportParams(BaseModel):
    start_date: str
    end_date: str
    api_key: str
    export_type: str = "daily_with_keys"
    format: str = "json"


class UsageExportKeyMetadata(BaseModel):
    tag: str | None = None


class UsageExportUserMetadata(BaseModel):
    department: str | None = None


class UsageExportRow(BaseModel):
    api_key: str | None = None
    user_id: str | None = None
    key_metadata: UsageExportKeyMetadata
    user_metadata: UsageExportUserMetadata


_EXPORT_ROWS: Final = TypeAdapter(list[UsageExportRow])


class TestUsageExportMetadata:
    @pytest.mark.replayable
    @pytest.mark.covers("quota_management.spend_tracking.usage_export.carries_key_and_user_metadata")
    @meta(
        Subject(
            domain=Domain.SPEND_BUDGETS,
            providers=(Provider.OPENAI,),
            models=(BACKEND,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_daily_with_keys_export_row_carries_key_and_user_metadata(
        self, client: SpendClient, resources: ResourceManager
    ) -> None:
        started: Final = datetime.now(timezone.utc)
        marker: Final = unique_marker()
        base: Final = provider_edge_base("openai")
        model: Final = f"e2e-usage-export-{marker}"
        model_id: Final = client.proxy.create_model(
            model,
            LiteLLMParamsBody(
                model=BACKEND,
                api_key="os.environ/OPENAI_API_KEY",
                api_base=None if base is None else f"{base}/v1",
                input_cost_per_token=INPUT_RATE,
                output_cost_per_token=OUTPUT_RATE,
            ),
        )
        resources.defer(lambda: client.proxy.delete_model(model_id))
        user_id: Final = unwrap(
            client.proxy.transport.post(
                "/user/new",
                headers=client.proxy.transport.master,
                json=UserNewBody(
                    user_email=f"e2e-usage-export-{marker}@example.com",
                    user_role="internal_user",
                    user_id=f"e2e-usage-export-{marker}",
                    metadata={"department": f"research-{marker}"},
                ),
                response_type=UserNewResponse,
            )
        ).user_id
        resources.defer(lambda: client.delete_user(user_id))
        record: Final = client.generate_key_record(
            KeyGenerateBody(
                user_id=user_id,
                models=[model],
                key_alias=f"e2e-usage-export-{marker}",
                metadata=KeyMetadata(tag=f"cost-center-{marker}"),
            )
        )
        resources.defer(lambda: client.proxy.delete_key(record.key))
        assert record.token, "/key/generate answered without the key's token hash"
        token: Final = record.token

        response: Final = unwrap(
            client.proxy.chat(
                record.key,
                ChatBody(
                    model=model,
                    messages=[ChatMessage(role="user", content=f"Reply with one word. {marker}")],
                    max_completion_tokens=32,
                ),
            )
        )
        assert response.id, "successful response must have an ID"
        end: Final = datetime.now(timezone.utc) + timedelta(days=1)
        activity: Final = client.poll_daily_activity_for_key(
            token, start=started - timedelta(days=1), end=end, min_requests=1
        )
        assert activity is not None, f"/user/daily/activity never listed key {token} under api_keys"

        exported: Final = client.proxy.transport.probe(
            EXPORT_ROUTE,
            params=UsageExportParams(
                start_date=(started - timedelta(days=1)).strftime("%Y-%m-%d"),
                end_date=end.strftime("%Y-%m-%d"),
                api_key=token,
            ),
        )
        assert exported.status_code == 200, f"{EXPORT_ROUTE} answered {exported.status_code}: {exported.body}"
        rows: Final = _EXPORT_ROWS.validate_json(exported.body)
        key_rows: Final = tuple(row for row in rows if row.api_key == token)
        assert key_rows, f"{EXPORT_ROUTE} returned no daily_with_keys row for {token}: {exported.body}"
        for row in key_rows:
            assert row.user_id == user_id, f"export row for {token} is owned by {row.user_id}, not {user_id}"
            assert row.key_metadata.tag == f"cost-center-{marker}", f"key metadata not exported: {row.key_metadata}"
            assert row.user_metadata.department == f"research-{marker}", (
                f"user metadata not exported: {row.user_metadata}"
            )
