# Mavvrik Hourly Export — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Export today's partial data to GCS every hour, overwriting the same date file for near-real-time visibility

**Architecture:** Add parallel hourly export alongside existing daily export. Hourly exports today's partial data, daily export handles complete historical days.

**Tech Stack:** Same stack (APScheduler, Polars, Mavvrik API, GCS)

---

## Current Behavior

- Scheduler runs every 60 minutes
- Exports only complete days (yesterday and earlier)
- Never exports today (rows still accumulating)
- Result: GCS always 1+ days behind

## Proposed Behavior

**Two separate jobs:**

1. **Daily export job** (existing) — runs every 60 min
   - Exports complete days from (marker + 1) to yesterday
   - Advances marker after each successful day
   - Never touches today

2. **Hourly export job** (new) — runs every 60 min
   - Exports today's partial data to GCS
   - Overwrites today's file each run
   - Does NOT advance marker (today isn't "complete")
   - Stops at midnight (daily job takes over)

**GCS behavior:**
- Today's file gets overwritten hourly with more complete data
- At midnight, today becomes yesterday, daily job exports the final version
- Hourly job starts exporting the new "today"

---

## Task 1: Add `export_today_usage_data()` method

**Files:**
- Modify: `litellm/integrations/mavvrik/mavvrik.py`

**Step 1: Write test for hourly export**

```python
# tests/test_litellm/integrations/mavvrik/test_mavvrik_logger.py
@pytest.mark.asyncio
async def test_export_today_usage_data(mock_prisma, mock_mavvrik_streamer):
    """Test hourly export of today's partial data."""
    logger = MavvrikLogger(...)

    # Mock today's data
    today_str = date.today().isoformat()

    # Call export_today_usage_data
    await logger.export_today_usage_data()

    # Verify:
    # 1. Query was for today's date
    # 2. Upload was called with today's date
    # 3. Marker was NOT advanced
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_litellm/integrations/mavvrik/test_mavvrik_logger.py::test_export_today_usage_data -v`
Expected: FAIL with "export_today_usage_data not defined"

**Step 3: Implement minimal export_today_usage_data**

```python
# litellm/integrations/mavvrik/mavvrik.py
async def export_today_usage_data(self, limit: int = 50000) -> None:
    """
    Export today's partial data to GCS (hourly job).

    Overwrites today's file in GCS each run. Does NOT advance marker.
    """
    try:
        today_str = date.today().isoformat()

        verbose_logger.info(f"Exporting today's partial data: {today_str}")

        # Query today's data
        db = LiteLLMDatabase()
        df = await db.get_usage_data(date_str=today_str, limit=limit)

        if df.is_empty():
            verbose_logger.info(f"No data for {today_str}, skipping")
            return

        # Transform to CSV
        csv_str = MavvrikTransformer.to_csv(df)

        # Upload to GCS (overwrites existing file for today)
        streamer = MavvrikStreamer(
            api_key=self.api_key,
            api_endpoint=self.api_endpoint,
            tenant=self.tenant,
            instance_id=self.instance_id,
        )
        await streamer.upload(csv_str, date_str=today_str)

        verbose_logger.info(f"✓ Today's partial data exported: {today_str}")

        # DO NOT advance marker — today is not complete

    except Exception as e:
        verbose_logger.error(f"Failed to export today's data: {e}", exc_info=True)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_litellm/integrations/mavvrik/test_mavvrik_logger.py::test_export_today_usage_data -v`
Expected: PASS

**Step 5: Commit**

```bash
git add litellm/integrations/mavvrik/mavvrik.py tests/test_litellm/integrations/mavvrik/test_mavvrik_logger.py
git commit -m "feat(mavvrik): add export_today_usage_data for hourly partial exports"
```

---

## Task 2: Add hourly scheduler job

**Files:**
- Modify: `litellm/integrations/mavvrik/mavvrik.py`

**Step 1: Write test for hourly job initialization**

```python
@pytest.mark.asyncio
async def test_initialize_hourly_export_job():
    """Test that hourly export job is scheduled."""
    logger = MavvrikLogger(...)
    scheduler = AsyncIOScheduler()

    await logger.initialize_mavvrik_hourly_export_job(scheduler)

    # Verify job is scheduled with 60 min interval
    jobs = scheduler.get_jobs()
    assert len(jobs) == 1
    assert jobs[0].trigger.interval.total_seconds() == 3600
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_litellm/integrations/mavvrik/test_mavvrik_logger.py::test_initialize_hourly_export_job -v`
Expected: FAIL

**Step 3: Implement initialize_mavvrik_hourly_export_job**

```python
# litellm/integrations/mavvrik/mavvrik.py
async def initialize_mavvrik_hourly_export_job(
    self, _scheduler: Optional[AsyncIOScheduler] = None
) -> None:
    """
    Initialize hourly export job for today's partial data.

    Runs every MAVVRIK_EXPORT_INTERVAL_MINUTES (default 60).
    Exports today's data to GCS (overwrites file each run).
    """
    from litellm.proxy.proxy_server import premium_user, prisma_client

    # Don't start if no settings
    if not await self._has_mavvrik_settings():
        verbose_logger.info("Mavvrik hourly job: no settings found, skipping")
        return

    interval_min = int(os.getenv("MAVVRIK_EXPORT_INTERVAL_MINUTES", "60"))

    verbose_logger.info(
        f"Starting Mavvrik hourly export job (interval: {interval_min} min)"
    )

    if _scheduler is None:
        from litellm.proxy.proxy_server import _scheduler_instance
        _scheduler = _scheduler_instance

    if _scheduler is None:
        verbose_logger.warning("Scheduler not available, hourly job not started")
        return

    _scheduler.add_job(
        self._hourly_export,
        "interval",
        minutes=interval_min,
        id="mavvrik_hourly_export",
        replace_existing=True,
    )

    verbose_logger.info("Mavvrik hourly export job started")
```

**Step 4: Implement _hourly_export wrapper**

```python
async def _hourly_export(self) -> None:
    """Wrapper for hourly export with pod lock."""
    from litellm.proxy.proxy_server import pod_lock_manager, prisma_client

    # Acquire lock if Redis available
    lock_acquired = False
    if pod_lock_manager is not None:
        lock_acquired = await pod_lock_manager.acquire_lock(
            lock_key="mavvrik_hourly_export"
        )
        if not lock_acquired:
            verbose_logger.info("Another pod is running hourly export, skipping")
            return

    try:
        await self.export_today_usage_data()
    finally:
        if lock_acquired and pod_lock_manager is not None:
            await pod_lock_manager.release_lock(lock_key="mavvrik_hourly_export")
```

**Step 5: Run test**

Run: `pytest tests/test_litellm/integrations/mavvrik/test_mavvrik_logger.py::test_initialize_hourly_export_job -v`
Expected: PASS

**Step 6: Commit**

```bash
git add litellm/integrations/mavvrik/mavvrik.py tests/test_litellm/integrations/mavvrik/test_mavvrik_logger.py
git commit -m "feat(mavvrik): add hourly scheduler job for today's partial data"
```

---

## Task 3: Wire up hourly job to proxy startup

**Files:**
- Modify: `litellm/integrations/mavvrik/mavvrik.py` (async_log_success_event)

**Step 1: Add hourly job initialization in async_log_success_event**

```python
# litellm/integrations/mavvrik/mavvrik.py
async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
    """
    Called on every successful API call.

    Initializes both daily and hourly export jobs on first call.
    """
    global _MAVVRIK_DAILY_JOB_STARTED, _MAVVRIK_HOURLY_JOB_STARTED

    # Initialize daily export job (existing)
    if not _MAVVRIK_DAILY_JOB_STARTED:
        _MAVVRIK_DAILY_JOB_STARTED = True
        asyncio.create_task(self.initialize_mavvrik_export_job())

    # Initialize hourly export job (new)
    if not _MAVVRIK_HOURLY_JOB_STARTED:
        _MAVVRIK_HOURLY_JOB_STARTED = True
        asyncio.create_task(self.initialize_mavvrik_hourly_export_job())
```

**Step 2: Add global flag**

```python
# Top of mavvrik.py
_MAVVRIK_DAILY_JOB_STARTED = False
_MAVVRIK_HOURLY_JOB_STARTED = False
```

**Step 3: Update health check**

```python
# litellm/proxy/health_endpoints/_health_endpoints.py
# Add mavvrik_hourly_scheduler_running to readiness response
```

**Step 4: Manual test**

```bash
# Start proxy with mavvrik config
litellm --config config/mavvrik-dev.yaml

# Wait for first API call to trigger job initialization

# Check health
curl http://localhost:4000/health/readiness | jq .mavvrik_hourly_scheduler_running
# Expected: true

# Check logs for hourly export
tail -f /tmp/litellm.log | grep "hourly"
# Expected: "Mavvrik hourly export job started"
# Expected: "Exporting today's partial data: 2026-02-20"
```

**Step 5: Commit**

```bash
git add litellm/integrations/mavvrik/mavvrik.py litellm/proxy/health_endpoints/_health_endpoints.py
git commit -m "feat(mavvrik): wire hourly job to proxy startup"
```

---

## Task 4: Update documentation

**Files:**
- Modify: `docs/mavvrik-data-flow.md`
- Modify: `docs/mavvrik-integration-design.md`
- Modify: `docs/mavvrik-customer-onboarding.md`

**Step 1: Update data flow doc**

Add section explaining hourly export:

```markdown
## Hourly Export (Today's Partial Data)

In addition to the daily export, Mavvrik runs an hourly job that exports today's
incomplete data to GCS. This provides near-real-time visibility into current spend.

**Behavior:**
- Runs every 60 minutes (same as daily job)
- Exports all rows where `date = today`
- Uploads to GCS with today's date as filename
- Overwrites the file each run (idempotent)
- Does NOT advance the marker

**At midnight:**
- Today becomes yesterday
- Daily export job exports the final version
- Hourly job starts exporting the new "today"
```

**Step 2: Update design doc**

Add to "Export Mechanism" section.

**Step 3: Update onboarding doc**

Update "What Happens Next" section:

```markdown
The integration runs two automatic jobs:

1. **Daily export** (every hour): exports complete historical days to GCS
2. **Hourly export** (every hour): exports today's partial data for real-time visibility

You do nothing. Both jobs run automatically.
```

**Step 4: Commit**

```bash
git add docs/mavvrik-*.md
git commit -m "docs(mavvrik): document hourly export of today's partial data"
```

---

## Task 5: Fix immediate issue - restart proxy with database

**Immediate action needed:**

```bash
# Kill current proxy
pkill -f "litellm.*proxy_server_config"

# Start with dev config (has database URL)
litellm --config config/mavvrik-dev.yaml --port 4000 > /tmp/litellm.log 2>&1 &

# Verify database connected
curl http://localhost:4000/mavvrik/settings -H "Authorization: Bearer sk-1234"
# Expected: { "marker": "2026-02-18", ... }

# Verify scheduler running
curl http://localhost:4000/health/readiness | jq .mavvrik_scheduler_running
# Expected: true

# Wait 5 minutes, check if Feb 19 exported
tail -f /tmp/litellm.log | grep "export"
```

---

## Testing Strategy

**Unit tests:**
- `test_export_today_usage_data()` — verify today's data exported
- `test_initialize_hourly_export_job()` — verify job scheduled
- `test_hourly_export_does_not_advance_marker()` — verify marker unchanged

**E2E test:**
- Run proxy with both jobs
- Trigger API calls to generate today's data
- Wait 5 minutes for hourly export
- Verify today's file in GCS
- Wait for another hour, verify file overwritten with more data

**Manual verification:**
- Check GCS for today's file (should appear within 60 min of first API call)
- Check marker not advanced by hourly job
- Check daily job still exports yesterday

---

## Rollout Plan

1. Implement hourly export (this plan)
2. Test in dev environment
3. Deploy to staging, verify both jobs running
4. Monitor GCS for duplicate exports (should not happen with pod lock)
5. Deploy to production
