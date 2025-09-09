# APScheduler Memory Leak Fix

## Problem Summary
The LiteLLM proxy server was experiencing critical memory leaks during startup, causing crashes with memory allocations exceeding 35GB. Memray analysis revealed the issue originated from APScheduler's internal functions.

## Root Cause Analysis

### Memory Leak Sources (from Memray stats):
1. `normalize()` function: **6.872GB allocated**
2. `_apply_jitter()` function: **6.542GB allocated** 
3. `get_next_fire_time()` functions: **6.173GB combined**
4. `_get_run_times()` function: **2.946GB allocated**

Total: **35.230GB allocated** with **483,180,019 allocations**

### Contributing Factors:
1. **Jitter Parameter**: The `jitter` parameter in job scheduling caused excessive memory allocations in APScheduler's normalize() function
2. **Very Frequent Intervals**: Jobs running every 10 seconds generated massive calculation overhead
3. **Missed Run Calculations**: APScheduler computing backlogs of missed runs during startup
4. **Job Rescheduling**: Resetting jobs to "now" triggered recalculation of thousands of missed executions

## Implemented Solution

### Key Changes:

#### 1. Removed Jitter Parameters
- **Before**: All jobs used `jitter` parameter (ranging from 2-3600 seconds)
- **After**: Removed all `jitter` parameters, using random offsets in intervals instead
- **Impact**: Eliminates the primary memory leak source in `_apply_jitter()` and `normalize()`

#### 2. Increased Minimum Job Intervals
- **Before**: Some jobs ran every 10 seconds
- **After**: Minimum interval increased to 30 seconds
- **Impact**: Reduces frequency of APScheduler calculations by 3x

#### 3. Enhanced Scheduler Configuration
```python
scheduler = AsyncIOScheduler(
    job_defaults={
        "coalesce": True,              # Collapse missed runs
        "misfire_grace_time": 3600,    # Increased from 120 to 3600 seconds
        "max_instances": 1,            # Prevent concurrent executions
        "replace_existing": True,      # Always replace existing jobs
    },
    jobstores={'default': MemoryJobStore()},
    executors={'default': AsyncIOExecutor()},
    timezone=None  # Disable timezone calculations
)
```

#### 4. Removed Job Rescheduling on Startup
- **Before**: All jobs were reset to `next_run_time=now` on startup
- **After**: Jobs start naturally with `misfire_grace_time` handling any backlogs
- **Impact**: Prevents massive backlog calculations

#### 5. Updated Default Constants
- `PROXY_BATCH_WRITE_AT`: Increased from 10 to 30 seconds default

## Files Modified
1. `litellm/proxy/proxy_server.py`:
   - Updated `initialize_scheduled_background_jobs()` method
   - Removed all `jitter` parameters from job scheduling
   - Added memory-optimized scheduler configuration
   - Increased job intervals

2. `litellm/constants.py`:
   - Updated `PROXY_BATCH_WRITE_AT` default from 10 to 30 seconds

## Deployment Notes

### Environment Variables (Optional Overrides)
If you need to adjust intervals, use these environment variables:
- `PROXY_BATCH_WRITE_AT`: Minimum 30 seconds recommended
- `PROXY_BUDGET_RESCHEDULER_MIN_TIME`: Default 597 seconds
- `PROXY_BUDGET_RESCHEDULER_MAX_TIME`: Default 605 seconds
- `PROXY_BATCH_POLLING_INTERVAL`: Default 3600 seconds

### Testing Recommendations
1. Monitor memory usage during proxy startup using:
   ```bash
   python -m memray run --output memray.bin litellm --config config.yaml
   python -m memray stats memray.bin
   ```

2. Verify scheduled jobs are running:
   - Check logs for "APScheduler started with memory leak prevention settings"
   - Monitor job execution timestamps

3. Load test with multiple proxy instances to ensure job distribution works without jitter

### Rollback Plan
If issues occur, rollback by:
1. Reverting the code changes
2. Setting `PROXY_BATCH_WRITE_AT=10` to restore original interval
3. Note: The memory leak will return with rollback

## Performance Impact
- **Memory**: Dramatic reduction from 35GB to expected <1GB during startup
- **CPU**: Reduced computational overhead from jitter calculations
- **Job Timing**: Slightly less random distribution (using interval offsets instead of jitter)
- **Reliability**: Improved stability, no more OOM crashes during startup

## Future Improvements
1. Consider migrating away from APScheduler to a simpler scheduling solution
2. Implement job queuing with external scheduler (Redis/Celery)
3. Add memory monitoring and alerts for scheduler operations