-- Atomic read-check-increment for budget and concurrent counters.
--
-- KEYS[1] = counter_key
-- KEYS[2] = window_key            (optional; absent for rolling-only counters)
-- ARGV[1] = limit                 (float; -1 disables the cap)
-- ARGV[2] = increment             (float)
-- ARGV[3] = window_period_seconds (int; 0 = no window-reset semantics)
-- ARGV[4] = counter_ttl_seconds   (int; 0 = no TTL change)
--
-- Returns (value and limit are bulk strings so float budgets are not truncated
-- by Redis' Lua-number-to-integer conversion):
--   { 0, new_value, limit, ttl_remaining }     admitted
--   { 1, current_value, limit, ttl_remaining } over-limit (NO mutation)
--   { 2, "<error tag>" }                        internal error (NO mutation)
--   { 3, "evicted_mid_window" }                 counter gone while window lives
--
-- Counter deletion is never performed here; counters age out via TTL only.

local time_reply = redis.call('TIME')
local now = tonumber(time_reply[1])

local limit         = tonumber(ARGV[1])
local increment     = tonumber(ARGV[2])
local window_period = tonumber(ARGV[3])
local ttl           = tonumber(ARGV[4])
local counter_key   = KEYS[1]

local raw_current = redis.call('GET', counter_key)
local current = tonumber(raw_current or '0')

if window_period > 0 and #KEYS >= 2 then
    local window_key = KEYS[2]
    local window_start = redis.call('GET', window_key)

    if window_start and (now - tonumber(window_start)) < window_period then
        -- Window is live. A missing counter here means the value was evicted
        -- mid-window (maxmemory-policy reclaimed it while window_key survived).
        -- Re-admitting from zero would silently leak budget, so signal
        -- degradation instead and let the engine's fail-mode decide.
        if not raw_current then
            return { 3, 'evicted_mid_window' }
        end
    else
        -- First write of the window, or the prior window elapsed. Start the
        -- counter at `increment` (not 0) and stamp a fresh window_start.
        if limit >= 0 and increment > limit then
            return { 1, '0', tostring(limit), window_period }
        end
        redis.call('SET', counter_key, increment)
        redis.call('SET', window_key, tostring(now))
        if ttl > 0 then
            redis.call('EXPIRE', counter_key, ttl)
            redis.call('EXPIRE', window_key, ttl)
        end
        return { 0, tostring(increment), tostring(limit), ttl }
    end
end

if limit >= 0 and (current + increment) > limit then
    return { 1, tostring(current), tostring(limit), redis.call('TTL', counter_key) }
end

local new_value = redis.call('INCRBYFLOAT', counter_key, increment)
local current_ttl = redis.call('TTL', counter_key)
if ttl > 0 and current_ttl == -1 then
    redis.call('EXPIRE', counter_key, ttl)
    current_ttl = ttl
end
return { 0, new_value, tostring(limit), current_ttl }
