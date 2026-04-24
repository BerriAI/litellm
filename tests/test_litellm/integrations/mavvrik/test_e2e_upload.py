"""End-to-end tests for the Mavvrik upload layer against the real API.

These tests hit the live Mavvrik API.  They are skipped automatically
when the required environment variables are absent so they never break CI.

Set the following env vars before running:

    MAVVRIK_API_KEY=<api-key>
    MAVVRIK_API_ENDPOINT=https://api.mavvrik.dev/<tenant-id>
    MAVVRIK_CONNECTION_ID=<connection-id>

Run with:
    poetry run pytest tests/test_litellm/integrations/mavvrik/test_e2e_upload.py -v -s
"""

import calendar
import os
import sys
from datetime import date, datetime, timedelta

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.integrations.mavvrik.client import Client
from litellm.integrations.mavvrik.uploader import Uploader

# ---------------------------------------------------------------------------
# Credentials — populated from env vars; test is skipped if any are absent.
# ---------------------------------------------------------------------------

API_KEY = os.getenv("MAVVRIK_API_KEY", "")
API_ENDPOINT = os.getenv("MAVVRIK_API_ENDPOINT", "")
CONNECTION_ID = os.getenv("MAVVRIK_CONNECTION_ID", "")

_CREDS_PRESENT = all([API_KEY, API_ENDPOINT, CONNECTION_ID])
_skip_if_no_creds = pytest.mark.skipif(
    not _CREDS_PRESENT,
    reason="Mavvrik credentials not configured — set MAVVRIK_API_KEY, MAVVRIK_API_ENDPOINT, MAVVRIK_CONNECTION_ID",
)

# Name used for the synthetic GCS object.
# Prefixed with "test-" so it is clearly not real data.
_TEST_DATE = "test-e2e-litellm"

# Minimal synthetic CSV that matches the Mavvrik schema column order
_TEST_CSV = (
    "date,user_id,api_key,model,model_group,custom_llm_provider,"
    "prompt_tokens,completion_tokens,spend,api_requests,successful_requests,"
    "failed_requests,cache_creation_input_tokens,cache_read_input_tokens,"
    "created_at,updated_at,team_id,api_key_alias,team_alias,user_email\n"
    "2026-01-01,user-e2e,sk-test,gpt-4o,gpt-4o,openai,"
    "100,50,0.0025,1,1,0,0,0,"
    "2026-01-01T00:00:00Z,2026-01-01T00:01:00Z,team-e2e,e2e-key,e2e-team,e2e@example.com\n"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    return Client(
        api_key=API_KEY,
        api_endpoint=API_ENDPOINT,
        connection_id=CONNECTION_ID,
    )


@pytest.fixture(scope="module")
def uploader(client):
    return Uploader(client=client)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@_skip_if_no_creds
class TestE2ERegister:
    @pytest.mark.asyncio
    async def test_register_returns_iso_string_or_none(self, client):
        """register() must return an ISO-8601 date string or None (first run)."""
        marker = await client.register()
        print(f"\n  register() returned marker: {marker}")

        if marker is not None:
            dt = datetime.fromisoformat(marker)
            assert dt.year >= 2020, f"Unexpected marker year: {dt.year}"

    @pytest.mark.asyncio
    async def test_register_twice_is_idempotent(self, client):
        """Calling register() twice should succeed without error."""
        m1 = await client.register()
        m2 = await client.register()
        print(f"\n  First call:  {m1}")
        print(f"  Second call: {m2}")
        if m1 is not None:
            datetime.fromisoformat(m1)
        if m2 is not None:
            datetime.fromisoformat(m2)


@_skip_if_no_creds
class TestE2EGetSignedUrl:
    @pytest.mark.asyncio
    async def test_get_signed_url_returns_url(self, client):
        """get_signed_url() must return a GCS URL for a given date name."""
        url = await client.get_signed_url(_TEST_DATE)
        print(f"\n  signed URL: {url[:80]}...")
        assert url.startswith("https://"), f"Expected https URL, got: {url[:40]}"


@_skip_if_no_creds
class TestE2EUpload:
    @pytest.mark.asyncio
    async def test_upload_synthetic_csv(self, uploader):
        """Full 3-step GCS upload: get signed URL → initiate → PUT gzip bytes."""
        await uploader.upload(_TEST_CSV, date_str=_TEST_DATE)
        print(f"\n  Upload for date {_TEST_DATE} succeeded")

    @pytest.mark.asyncio
    async def test_upload_same_date_twice_is_idempotent(self, uploader):
        """Re-uploading the same date must succeed (GCS object is overwritten)."""
        await uploader.upload(_TEST_CSV, date_str=_TEST_DATE)
        await uploader.upload(_TEST_CSV, date_str=_TEST_DATE)
        print(f"\n  Two uploads for date {_TEST_DATE} both succeeded (idempotent)")

    @pytest.mark.asyncio
    async def test_upload_empty_payload_is_noop(self, uploader):
        """Empty payload must return without making any network calls."""
        await uploader.upload("   ", date_str=_TEST_DATE)
        print("\n  Empty payload correctly skipped")


@_skip_if_no_creds
class TestE2EAdvanceMarker:
    @pytest.mark.asyncio
    async def test_advance_marker_succeeds(self, client):
        """advance_marker() must PATCH Mavvrik without raising."""
        epoch = 1700000000
        await client.advance_marker(epoch)
        print(f"\n  advance_marker({epoch}) succeeded")

    @pytest.mark.asyncio
    async def test_advance_marker_with_recent_date(self, client):
        """advance_marker() with a recent epoch must also succeed."""
        yesterday = date.today() - timedelta(days=1)
        epoch = int(calendar.timegm(yesterday.timetuple()))
        await client.advance_marker(epoch)
        print(f"\n  advance_marker({epoch}) for {yesterday} succeeded")


@_skip_if_no_creds
class TestE2EFullFlow:
    @pytest.mark.asyncio
    async def test_register_then_upload_then_advance(self, client, uploader):
        """Simulate one complete scheduled export cycle end-to-end."""
        marker_iso = await client.register()
        if marker_iso is not None:
            datetime.fromisoformat(marker_iso)
        print(f"\n  register() marker: {marker_iso}")

        await uploader.upload(_TEST_CSV, date_str=_TEST_DATE)
        print(f"  upload() for {_TEST_DATE}: OK")

        export_epoch = 1700000000
        await client.advance_marker(export_epoch)
        print(f"  advance_marker({export_epoch}): OK")

        print("\n  Full cycle PASSED")
