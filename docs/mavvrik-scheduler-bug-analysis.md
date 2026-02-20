# Mavvrik Scheduler Not Starting - Root Cause Analysis

## Summary

The Mavvrik scheduler doesn't initialize at proxy startup because of an **initialization order bug**: the scheduler tries to find `MavvrikLogger` instances before they are created.

## Timeline of Events

**From logs:**
```
Line 116: INFO:     Application startup complete.
Line 133: Initialized Success Callbacks - ['mavvrik']
```

The callbacks are initialized AFTER application startup completes.

## Root Cause

**Initialization order:**

1. `proxy_startup_event()` runs (proxy_server.py:737)
2. `load_config()` is called (lines 766, 779, 789)
3. `initialize_scheduled_background_jobs()` is called (line 855)
4. Inside that, `MavvrikLogger.init_mavvrik_background_job()` is called (line 5631)
5. This tries to find MavvrikLogger instances: `get_custom_loggers_for_type(callback_type=MavvrikLogger)` (mavvrik.py:373-375)
6. **Returns empty list** because logger instances don't exist yet
7. Scheduler job is NOT added
8. Application startup completes (line 882: `yield`)
9. **THEN** callbacks are initialized and logger instances created

**The gap**: Steps 4-7 happen before step 9, so no logger instances exist when the scheduler tries to find them.

## Code Evidence

**mavvrik.py:369-387**
```python
@staticmethod
async def init_mavvrik_background_job(scheduler: AsyncIOScheduler):
    """Register the hourly export job with APScheduler."""
    loggers: List[CustomLogger] = litellm.logging_callback_manager.get_custom_loggers_for_type(
        callback_type=MavvrikLogger
    )
    verbose_logger.debug("MavvrikLogger: found %d logger instance(s)", len(loggers))
    if loggers:  # <-- This is False because loggers is empty!
        mavvrik_logger = cast(MavvrikLogger, loggers[0])
        scheduler.add_job(
            mavvrik_logger.initialize_mavvrik_export_job,
            "interval",
            minutes=MAVVRIK_EXPORT_INTERVAL_MINUTES,
        )
```

## Why CloudZero Works

CloudZero likely works because it has a different initialization pattern, or it's configured differently in the test setup.

## Solutions

### Option 1: Move scheduler init to after callback initialization (Recommended)

Add a hook that triggers after callbacks are initialized to call `init_mavvrik_background_job`.

**Where to add:**
After line 2675 in proxy_server.py (right after "Initialized Success Callbacks" is printed).

```python
# After callbacks are initialized
if "mavvrik" in litellm.success_callback:
    from litellm.integrations.mavvrik.mavvrik import MavvrikLogger
    if scheduler is not None:
        await MavvrikLogger.init_mavvrik_background_job(scheduler=scheduler)
```

### Option 2: Lazy initialization on first callback

Initialize the scheduler on the first successful API call instead of at startup.

**Implementation:**
Add to `MavvrikLogger` class:

```python
_scheduler_initialized = False

async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
    global _scheduler_initialized
    if not _scheduler_initialized:
        from litellm.proxy.proxy_server import _scheduler_instance
        if _scheduler_instance is not None:
            await self.__class__.init_mavvrik_background_job(scheduler=_scheduler_instance)
            _scheduler_initialized = True
```

### Option 3: Instantiate logger earlier

Ensure MavvrikLogger instance is created during `load_config` and added to the callback manager immediately.

## Recommendation

**Use Option 1** - it's the cleanest and most explicit. Move the scheduler initialization to happen after callbacks are initialized, right after the "Initialized Success Callbacks" message is printed.

## Testing the Fix

After implementing:

1. Restart proxy
2. Check logs for:
   - "MavvrikLogger: found X logger instance(s)" (should be > 0)
   - "MavvrikLogger: scheduling export job every 60 minutes"
3. Check health endpoint:
   ```bash
   curl http://localhost:4000/health/readiness | jq .mavvrik_scheduler_running
   # Should return: true
   ```
4. Wait 60 minutes or check logs for first export

## Files to Modify

- `litellm/proxy/proxy_server.py` (add scheduler init after callback init)
- OR `litellm/integrations/mavvrik/mavvrik.py` (add lazy init on first callback)
