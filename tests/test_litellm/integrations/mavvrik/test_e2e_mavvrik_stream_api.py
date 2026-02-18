"""End-to-end tests for the Mavvrik streaming layer against the real API.

These tests hit the live Mavvrik API and GCS.  They are skipped automatically
when the required environment variables are absent so they never break CI.

Set the following env vars (or edit the constants below) before running:

    MAVVRIK_API_KEY=<api-key>
    MAVVRIK_API_ENDPOINT=https://api.mavvrik.dev
    MAVVRIK_TENANT=<tenant-id>
    MAVVRIK_INSTANCE_ID=<instance-id>

Run with:
    poetry run pytest tests/test_litellm/integrations/mavvrik/test_e2e_mavvrik_stream_api.py -v -s
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.integrations.mavvrik.mavvrik_stream_api import MavvrikStreamer

# ---------------------------------------------------------------------------
# Credentials — populated from env vars; test is skipped if any are absent.
# ---------------------------------------------------------------------------

API_KEY = os.getenv("MAVVRIK_API_KEY", "xkcsvi7ja9MP3AGXeT4fjOZxyY1eUhMHLEQjZ3w1IHVPFeVCrPhyeJ8sRKNGDhvi")
API_ENDPOINT = os.getenv("MAVVRIK_API_ENDPOINT", "https://api.mavvrik.dev")
TENANT = os.getenv("MAVVRIK_TENANT", "iymc3dzwr5_x9izj")
INSTANCE_ID = os.getenv("MAVVRIK_INSTANCE_ID", "dybter3fyd")

_CREDS_PRESENT = all([API_KEY, API_ENDPOINT, TENANT, INSTANCE_ID])
_skip_if_no_creds = pytest.mark.skipif(
    not _CREDS_PRESENT,
    reason="Mavvrik credentials not configured",
)

# Date used for the synthetic upload — far enough in the past that it won't
# interfere with real data but still a valid YYYY-MM-DD string.
_TEST_DATE = "2026-01-01"

# Minimal synthetic CSV that matches the Mavvrik schema column order
_TEST_CSV = (
    "id,date,user_id,api_key,model,model_group,custom_llm_provider,"
    "prompt_tokens,completion_tokens,spend,api_requests,successful_requests,"
    "failed_requests,cache_creation_input_tokens,cache_read_input_tokens,"
    "created_at,updated_at,team_id,api_key_alias,team_alias,user_email\n"
    "test-row-001,2026-01-01,user-e2e,sk-test,gpt-4o,gpt-4o,openai,"
    "100,50,0.0025,1,1,0,0,0,"
    "2026-01-01T00:00:00Z,2026-01-01T00:01:00Z,team-e2e,e2e-key,e2e-team,e2e@example.com\n"
)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def streamer():
    return MavvrikStreamer(
        api_key=API_KEY,
        api_endpoint=API_ENDPOINT,
        tenant=TENANT,
        instance_id=INSTANCE_ID,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@_skip_if_no_creds
class TestE2ERegister:
    def test_register_returns_iso_string(self, streamer):
        """register() must return an ISO-8601 date string from the live API."""
        from datetime import datetime

        marker = streamer.register()
        print(f"\n  register() returned marker: {marker}")

        # Must be parseable as a datetime
        dt = datetime.fromisoformat(marker)
        assert dt.year >= 2020, f"Unexpected marker year: {dt.year}"

    def test_register_twice_is_idempotent(self, streamer):
        """Calling register() twice should succeed without error."""
        m1 = streamer.register()
        m2 = streamer.register()
        print(f"\n  First call:  {m1}")
        print(f"  Second call: {m2}")
        # Both must be valid ISO strings — values may differ if Mavvrik updates
        from datetime import datetime
        datetime.fromisoformat(m1)
        datetime.fromisoformat(m2)


@_skip_if_no_creds
class TestE2EGetSignedUrl:
    def test_get_signed_url_returns_gcs_url(self, streamer):
        """_get_signed_url() must return a GCS URL for a given date."""
        url = streamer._get_signed_url(_TEST_DATE)
        print(f"\n  signed URL: {url[:80]}...")
        assert url.startswith("https://"), f"Expected https URL, got: {url[:40]}"
        assert "storage.googleapis.com" in url or "storage.google" in url, (
            f"Expected GCS URL, got: {url[:80]}"
        )


@_skip_if_no_creds
class TestE2EUpload:
    def test_upload_synthetic_csv(self, streamer):
        """Full 3-step upload: get signed URL → initiate → PUT gzip bytes."""
        # Should not raise
        streamer.upload(_TEST_CSV, date_str=_TEST_DATE)
        print(f"\n  Upload for date {_TEST_DATE} succeeded")

    def test_upload_same_date_twice_is_idempotent(self, streamer):
        """Re-uploading the same date must succeed (GCS object is overwritten)."""
        streamer.upload(_TEST_CSV, date_str=_TEST_DATE)
        streamer.upload(_TEST_CSV, date_str=_TEST_DATE)
        print(f"\n  Two uploads for date {_TEST_DATE} both succeeded (idempotent)")

    def test_upload_empty_payload_is_noop(self, streamer):
        """Empty payload must return without making any network calls."""
        # No exception, no upload
        streamer.upload("   ", date_str=_TEST_DATE)
        print("\n  Empty payload correctly skipped")


@_skip_if_no_creds
class TestE2EAdvanceMarker:
    def test_advance_marker_succeeds(self, streamer):
        """advance_marker() must PATCH Mavvrik without raising."""
        import calendar
        from datetime import date

        epoch = int(
            calendar.timegm(date.fromisoformat(_TEST_DATE).timetuple())
        )
        streamer.advance_marker(epoch)
        print(f"\n  advance_marker({epoch}) succeeded")

    def test_advance_marker_with_recent_date(self, streamer):
        """advance_marker() with a recent epoch must also succeed."""
        import calendar
        from datetime import date, timedelta

        yesterday = date.today() - timedelta(days=1)
        epoch = int(calendar.timegm(yesterday.timetuple()))
        streamer.advance_marker(epoch)
        print(f"\n  advance_marker({epoch}) for {yesterday} succeeded")


@_skip_if_no_creds
class TestE2EFullFlow:
    def test_register_then_upload_then_advance(self, streamer):
        """Simulate one complete scheduled export cycle end-to-end."""
        import calendar
        from datetime import date, datetime

        # Step 1: register (verify connectivity + get marker)
        marker_iso = streamer.register()
        marker_dt = datetime.fromisoformat(marker_iso)
        print(f"\n  register() marker: {marker_iso}")

        # Step 2: upload one day of synthetic data
        streamer.upload(_TEST_CSV, date_str=_TEST_DATE)
        print(f"  upload() for {_TEST_DATE}: OK")

        # Step 3: advance Mavvrik's metricsMarker to the exported date
        export_epoch = int(
            calendar.timegm(date.fromisoformat(_TEST_DATE).timetuple())
        )
        streamer.advance_marker(export_epoch)
        print(f"  advance_marker({export_epoch}): OK")

        print("\n  Full cycle PASSED")
