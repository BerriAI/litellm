-- Apply a post-call actual-vs-estimated delta exactly once.
--
-- Deliberately has no window-reset branch: a slow stream completing after the
-- budget window rolls must not seed the new window with a stale delta. It only
-- dedups on request_id and applies the delta to the existing counter.
--
-- KEYS[1] = reconciled_key   (dedup marker, NX-set inside this EVAL)
-- KEYS[2] = counter_key
-- ARGV[1] = delta            (float; actual_cost - estimated_cost, may be < 0)
-- ARGV[2] = dedup_ttl_seconds (int)
--
-- Returns:
--   { 0, new_value }           applied
--   { 1, "duplicate" }         dedup key already present; delta dropped
--   { 4, "reconcile_against_evicted" } counter is gone; delta cannot be applied

local reconciled_key = KEYS[1]
local counter_key    = KEYS[2]
local delta          = tonumber(ARGV[1])
local dedup_ttl      = tonumber(ARGV[2])

local marked = redis.call('SET', reconciled_key, '1', 'NX', 'EX', dedup_ttl)
if not marked then
    return { 1, 'duplicate' }
end

if redis.call('EXISTS', counter_key) == 0 then
    return { 4, 'reconcile_against_evicted' }
end

local new_value = redis.call('INCRBYFLOAT', counter_key, delta)
return { 0, new_value }
