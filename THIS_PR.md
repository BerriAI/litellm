## Title

Fix TypeError in Scheduler When Priority Field Has Inconsistent Types

## Relevant issues

Fixes #14817

## Pre-Submission checklist

**Please complete all items before asking a LiteLLM maintainer to review your PR**

- [x] I have Added testing in the [`tests/litellm/`](https://github.com/BerriAI/litellm/tree/main/tests/litellm) directory, **Adding at least 1 test is a hard requirement** - [see details](https://docs.litellm.ai/docs/extras/contributing_code)
  - Added 3 new tests in `tests/local_testing/test_scheduler.py`
  - Added 5 new tests in `tests/test_litellm/caching/test_redis_cache.py`
- [x] I have added a screenshot of my new test passing locally 
  - See verification results below
- [x] My PR passes all unit tests on [`make test-unit`](https://docs.litellm.ai/docs/extras/contributing_code)
  - All 10 scheduler tests pass (7 existing + 3 new)
  - All Redis cache tests pass
- [x] My PR's scope is as isolated as possible, it only solves 1 specific problem
  - Focused solely on fixing TypeError in scheduler priority queue handling


## Type

🐛 Bug Fix

## Changes

### Problem
When processing requests with mixed priority field types (int, tuple, list), LiteLLM's scheduler crashes with:
```
TypeError: '<' not supported between instances of 'tuple' and 'list'
```

This occurs when Redis cache deserialization corrupts queue data types - `ast.literal_eval()` converts strings back to mixed types (tuples and lists), which then fail comparison in `heapq` operations.

### Root Cause
1. **Redis Serialization Issue**: When scheduler queue is stored in Redis and retrieved, deserialization can corrupt data types
2. **Heap Comparison Failure**: `heapq.heappush()` receives mixed types: `(int, str)` from fresh requests vs `(list, str)` from corrupted cache
3. **No Type Validation**: No validation of data types before heap operations

### Solution
Implemented **defense-in-depth** validation at three layers:

#### 1. Redis Cache Deserialization (`litellm/caching/redis_cache.py`)
Enhanced `_get_cache_logic()` method (lines 729-773):
- Validates and normalizes queue data during deserialization
- Ensures priority is always `int` and request_id is always `str`
- Gracefully skips invalid items with warning logs
- Returns empty list for completely failed deserializations

```python
# After ast.literal_eval, validate and normalize queue data
if isinstance(cached_response, list):
    normalized_queue = []
    for item in cached_response:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            try:
                priority = int(item[0]) if not isinstance(item[0], int) else item[0]
                request_id = str(item[1]) if not isinstance(item[1], str) else item[1]
                normalized_queue.append((priority, request_id))
            except (ValueError, TypeError):
                verbose_logger.warning(f"Skipping invalid queue item...")
```

#### 2. Scheduler Queue Retrieval (`litellm/scheduler.py`)
Enhanced `get_queue()` method (lines 116-144):
- Validates all queue items retrieved from cache
- Ensures each item is a tuple with `(int, str)` format
- Filters out invalid items
- Returns only validated queue items

#### 3. Request Addition (`litellm/scheduler.py`)
Enhanced `add_request()` method (lines 48-68):
- Validates priority is an integer before adding to heap
- Attempts to convert non-integer priorities to int
- Falls back to default priority of 0 if conversion fails
- Adds verbose logging for debugging

### Testing

#### New Tests Added

**Scheduler Tests** (`tests/local_testing/test_scheduler.py`):
1. `test_scheduler_priority_type_normalization` - Tests priority type normalization
2. `test_scheduler_redis_cache_deserialization` - Tests Redis cache with corrupted data
3. `test_scheduler_handles_invalid_queue_items` - Tests graceful handling of invalid items

**Redis Cache Tests** (`tests/test_litellm/caching/test_redis_cache.py`):
1. `test_get_cache_logic_normalizes_scheduler_queue` - Tests normalization of mixed tuple/list types
2. `test_get_cache_logic_handles_invalid_queue_items` - Tests handling of various invalid items
3. `test_get_cache_logic_handles_json_serialized_queue` - Tests normal JSON deserialization path
4. `test_get_cache_logic_returns_none_for_none_input` - Tests None input handling
5. `test_get_cache_logic_handles_non_queue_data` - Tests that non-queue data isn't affected

#### Test Results

**All tests pass on fix branch:**
```bash
$ pytest tests/local_testing/test_scheduler.py -v
======================== 10 passed, 1 warning in 4.04s =========================
```

**Verification of fix:**
```bash
# On fix branch (exit code 0 = success):
$ python verify_fix.py
RESULT: ✓ FIX VERIFIED - Scheduler handles corrupted data gracefully
Exit code 0: Fix is working correctly ✓

# On main branch (exit code 1 = bug exists):
$ python verify_fix.py
✗ TypeError occurred: '<' not supported between instances of 'tuple' and 'list'
RESULT: ✗ BUG EXISTS - TypeError occurred with corrupted data
Exit code 1: Bug is present ✗
```

### Files Modified

```
5 files changed, 533 insertions(+), 4 deletions(-)
- FIX_SUMMARY.md                                 | 171 lines (documentation)
- litellm/caching/redis_cache.py                 |  32 lines (fix)
- litellm/scheduler.py                           |  33 lines (fix)
- tests/local_testing/test_scheduler.py          | 123 lines (tests)
- tests/test_litellm/caching/test_redis_cache.py | 178 lines (tests)
```

### Impact

- ✅ **Prevents service crashes** from type comparison errors in production
- ✅ **Maintains data integrity** in priority queue
- ✅ **Backward compatible** - no breaking changes
- ✅ **Minimal performance impact** - only during cache operations
- ✅ **Graceful degradation** - invalid items skipped rather than causing crashes
- ✅ **Better debugging** - added verbose logging for troubleshooting

### Additional Documentation

- `FIX_SUMMARY.md` - Comprehensive documentation of the fix, root cause analysis, and solution architecture
- `verify_fix.py` - Verification script demonstrating the bug on main and fix on this branch
- `test_bug_reproduction.py` - Script to reproduce the exact bug from the issue

### Screenshots

**Test Results - All Passing:**
```
tests/local_testing/test_scheduler.py::test_scheduler_diff_model_names PASSED [ 10%]
tests/local_testing/test_scheduler.py::test_scheduler_handles_invalid_queue_items PASSED [ 20%]
tests/local_testing/test_scheduler.py::test_scheduler_prioritized_requests[healthy_deployments0-0-0] PASSED [ 30%]
tests/local_testing/test_scheduler.py::test_scheduler_prioritized_requests[healthy_deployments0-0-1] PASSED [ 40%]
tests/local_testing/test_scheduler.py::test_scheduler_prioritized_requests[healthy_deployments0-1-0] PASSED [ 50%]
tests/local_testing/test_scheduler.py::test_scheduler_prioritized_requests[healthy_deployments1-0-0] PASSED [ 60%]
tests/local_testing/test_scheduler.py::test_scheduler_prioritized_requests[healthy_deployments1-0-1] PASSED [ 70%]
tests/local_testing/test_scheduler.py::test_scheduler_prioritized_requests[healthy_deployments1-1-0] PASSED [ 80%]
tests/local_testing/test_scheduler.py::test_scheduler_priority_type_normalization PASSED [ 90%]
tests/local_testing/test_scheduler.py::test_scheduler_redis_cache_deserialization PASSED [100%]

======================== 10 passed, 1 warning in 4.04s =========================
```

**Verification - Bug Exists on Main:**
```
✗ TypeError occurred: '<' not supported between instances of 'tuple' and 'list'
This error indicates the bug from issue #14817 is present.
```

**Verification - Fix Works on This Branch:**
```
✓ Request added successfully!
✓ All queue items are properly normalized to (int, str) tuples
RESULT: ✓ FIX VERIFIED - Scheduler handles corrupted data gracefully
```

