from typing import Final

WINDOW_INCREMENT: Final = """
local results = {}
local now = tonumber(ARGV[1])
local window_size = tonumber(ARGV[2])

-- Process each window/counter pair
for i = 1, #KEYS, 2 do
    local window_key = KEYS[i]
    local counter_key = KEYS[i + 1]
    local increment_value = 1

    -- Check if window exists and is valid
    local window_start = redis.call('GET', window_key)
    if not window_start or (now - tonumber(window_start)) >= window_size then
        -- Reset window and counter
        local prefix = string.sub(window_key, 1, -(#':window') - 1)
        redis.call('DEL', prefix .. ':requests', prefix .. ':tokens')
        redis.call('SET', window_key, tostring(now))
        redis.call('SET', counter_key, increment_value)
        redis.call('EXPIRE', window_key, window_size)
        redis.call('EXPIRE', counter_key, window_size)
        table.insert(results, tostring(now)) -- window_start
        table.insert(results, increment_value) -- counter
    else
        local counter = redis.call('INCR', counter_key)
        -- This happens when window_key exists but counter_key doesn't (e.g., tokens key
        -- created after requests key when both share the same window_key)
        local current_ttl = redis.call('TTL', counter_key)
        if current_ttl == -1 then
            redis.call('EXPIRE', counter_key, window_size)
        end
        table.insert(results, window_start) -- window_start
        table.insert(results, counter) -- counter
    end
end

return results
"""

COUNTER_READ: Final = "return redis.call('MGET', unpack(KEYS))"

WINDOW_RESERVE: Final = """
-- Atomic check-and-increment-by-N across one or more descriptors.
-- All-or-nothing: if any descriptor would exceed its limit, no counter is
-- modified.
--
-- Uses Redis server time (`redis.call('TIME')`) instead of a client-supplied
-- timestamp so that window resets are deterministic across replicas with
-- skewed wall-clocks. This prevents a clock-skew-induced reopening of the
-- TOCTOU window across multi-replica deployments.
--
-- KEYS layout: pairs of (window_key, counter_key), one pair per descriptor.
-- ARGV layout: per-descriptor 4-tuple, starting at ARGV[1]:
--     ARGV[(i-1)*4 + 1] = limit
--     ARGV[(i-1)*4 + 2] = increment
--     ARGV[(i-1)*4 + 3] = ttl_seconds (counter TTL when window resets)
--     ARGV[(i-1)*4 + 4] = window_size_seconds (sliding-window length)
--
-- Return on success:
--     { 0, new_counter_1, window_start_1, new_counter_2, window_start_2, ... }
-- Return on over-limit: { 1, descriptor_index, current_counter, limit }
local time_reply = redis.call('TIME')
local now = tonumber(time_reply[1])
local descriptor_count = #KEYS / 2
local reset_windows = {}

-- Pass 1: read state, validate. Abort without writing if any over limit.
local descriptor_state = {}
for i = 1, descriptor_count do
    local window_key = KEYS[(i - 1) * 2 + 1]
    local counter_key = KEYS[(i - 1) * 2 + 2]
    local arg_base = (i - 1) * 4 + 1
    local limit = tonumber(ARGV[arg_base])
    local increment = tonumber(ARGV[arg_base + 1])
    local window_size = tonumber(ARGV[arg_base + 3])

    local window_start = redis.call('GET', window_key)
    local window_expired = (not window_start) or
        ((now - tonumber(window_start)) >= window_size)

    local current_counter
    if window_expired then
        current_counter = 0
    else
        current_counter = tonumber(redis.call('GET', counter_key) or 0)
    end

    local blocked
    if increment > 0 then
        blocked = current_counter + increment > limit
    else
        blocked = current_counter >= limit
    end
    if blocked then
        return { 1, i, current_counter, limit }
    end

    descriptor_state[i] = { window_expired, current_counter, window_start }
end

-- Pass 2: all checks passed. Apply increments.
local results = { 0 }
for i = 1, descriptor_count do
    local window_key = KEYS[(i - 1) * 2 + 1]
    local counter_key = KEYS[(i - 1) * 2 + 2]
    local arg_base = (i - 1) * 4 + 1
    local increment = tonumber(ARGV[arg_base + 1])
    local ttl = tonumber(ARGV[arg_base + 2])
    local window_size = tonumber(ARGV[arg_base + 3])

    local window_expired = descriptor_state[i][1]
    local active_window_start

    if window_expired then
        active_window_start = now
        if not reset_windows[window_key] then
            local prefix = string.sub(window_key, 1, -(#':window') - 1)
            redis.call('DEL', prefix .. ':requests', prefix .. ':tokens')
            reset_windows[window_key] = true
        end
        redis.call('SET', window_key, tostring(now))
        redis.call('SET', counter_key, increment)
        redis.call('EXPIRE', window_key, window_size)
        if ttl > 0 then
            redis.call('EXPIRE', counter_key, ttl)
        end
        table.insert(results, increment)
    else
        active_window_start = tonumber(descriptor_state[i][3])
        local new_counter = redis.call('INCRBY', counter_key, increment)
        local current_ttl = redis.call('TTL', counter_key)
        if current_ttl == -1 and ttl > 0 then
            redis.call('EXPIRE', counter_key, ttl)
        end
        table.insert(results, new_counter)
    end
    table.insert(results, active_window_start)
end

return results
"""

GUARDED_INCREMENT: Final = """
local results = {}
for i = 1, #KEYS, 2 do
    local window_key = KEYS[i]
    local counter_key = KEYS[i + 1]
    local arg_base = ((i - 1) / 2) * 3 + 1
    local expected_window_start = ARGV[arg_base]
    local increment = tonumber(ARGV[arg_base + 1])
    local ttl = tonumber(ARGV[arg_base + 2])
    local active_window_start = redis.call('GET', window_key)

    if active_window_start and active_window_start == expected_window_start then
        local new_counter = redis.call('INCRBY', counter_key, increment)
        local current_ttl = redis.call('TTL', counter_key)
        if current_ttl == -1 and ttl > 0 then
            redis.call('EXPIRE', counter_key, ttl)
        end
        table.insert(results, 1)
        table.insert(results, new_counter)
    else
        table.insert(results, 0)
        table.insert(results, tonumber(redis.call('GET', counter_key) or 0))
    end
end
return results
"""

SLOT_ACQUIRE: Final = """
-- Atomic check-and-acquire for the max_parallel_requests concurrency gauge.
-- Each gauge key is a sorted set of per-request slot ids scored by acquire
-- time (Redis server clock). In-flight requests are counted by ZCARD after
-- pruning slots older than the slot TTL, so unlike the windowed RPM/TPM
-- counters the gauge is never reset while requests are in flight, a
-- rejected request never occupies a slot, and a slot leaked by a crashed
-- worker self-heals after the slot TTL even under continuous traffic.
--
-- KEYS: one gauge zset key per descriptor.
-- ARGV: per-key triples (limit, slot_ttl_seconds, slot_id).
-- Success: { 0, in_flight_1, ... }. Over-limit: { 1, key_index, in_flight, limit }.
local time_reply = redis.call('TIME')
local now = tonumber(time_reply[1])
for i = 1, #KEYS do
    local limit = tonumber(ARGV[(i - 1) * 3 + 1])
    local slot_ttl = tonumber(ARGV[(i - 1) * 3 + 2])
    redis.call('ZREMRANGEBYSCORE', KEYS[i], '-inf', now - slot_ttl)
    local in_flight = redis.call('ZCARD', KEYS[i])
    if in_flight + 1 > limit then
        return { 1, i, in_flight, limit }
    end
end
local results = { 0 }
for i = 1, #KEYS do
    local slot_ttl = tonumber(ARGV[(i - 1) * 3 + 2])
    local slot_id = ARGV[(i - 1) * 3 + 3]
    redis.call('ZADD', KEYS[i], now, slot_id)
    redis.call('EXPIRE', KEYS[i], slot_ttl)
    table.insert(results, redis.call('ZCARD', KEYS[i]))
end
return results
"""

SLOT_RELEASE: Final = """
-- Release one slot per gauge key by removing this request's slot id.
-- ZREM of an absent member (or key) is a no-op, so a release without a
-- matching acquire (proxy-side rejection, double-fired callback, slot
-- already expired) can never free a slot owned by another request.
-- KEYS: gauge zset keys. ARGV: per-key slot_id.
-- Returns the remaining in-flight count per key.
local results = {}
for i = 1, #KEYS do
    redis.call('ZREM', KEYS[i], ARGV[i])
    table.insert(results, redis.call('ZCARD', KEYS[i]))
end
return results
"""

SLOT_COUNT: Final = """
-- Read the current in-flight count per gauge key (prunes expired slots
-- first so leaked slots do not inflate the reading).
-- KEYS: gauge zset keys. ARGV: per-key slot_ttl_seconds.
local time_reply = redis.call('TIME')
local now = tonumber(time_reply[1])
local results = {}
for i = 1, #KEYS do
    redis.call('ZREMRANGEBYSCORE', KEYS[i], '-inf', now - tonumber(ARGV[i]))
    table.insert(results, redis.call('ZCARD', KEYS[i]))
end
return results
"""

COUNTER_INCREMENT: Final = """
local results = {}

-- Process each key/increment_value/ttl triplet
for i = 1, #KEYS do
    local key = KEYS[i]
    local increment_value = tonumber(ARGV[i * 2 - 1])
    local ttl_seconds = tonumber(ARGV[i * 2])

    -- Increment the value
    local new_value = redis.call('INCRBYFLOAT', key, increment_value)

    -- Handle TTL: only set expire if ttl_seconds > 0 and key has no current TTL
    -- ttl_seconds can be 0 (no TTL) or positive (set TTL)
    if ttl_seconds and ttl_seconds > 0 then
        local current_ttl = redis.call('TTL', key)
        if current_ttl == -1 then
            redis.call('EXPIRE', key, ttl_seconds)
        end
    end

    table.insert(results, new_value)
end

return results
"""

LIMITED_INCREMENT: Final = """
local amount = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local current = tonumber(redis.call('GET', KEYS[1]) or '0')
if current + amount > limit then
    return {0, current}
end
local updated = redis.call('INCRBY', KEYS[1], amount)
redis.call('EXPIRE', KEYS[1], ttl)
return {1, updated}
"""

REFUND: Final = """
local updated = redis.call('DECRBY', KEYS[1], tonumber(ARGV[1]))
if updated <= 0 then
    redis.call('DEL', KEYS[1])
end
return 1
"""

SAVE: Final = """
redis.call('SET', KEYS[1], ARGV[1], 'EX', tonumber(ARGV[2]))
return 1
"""

POP: Final = """
local value = redis.call('GET', KEYS[1])
if value and value ~= '' then
    redis.call('SET', KEYS[1], '', 'EX', tonumber(ARGV[1]))
end
return value
"""

ITERATION_INCREMENT: Final = """
local key = KEYS[1]
local ttl = tonumber(ARGV[1])

local current = redis.call('INCR', key)
if current == 1 then
    redis.call('EXPIRE', key, ttl)
end

return current
"""

SPEND_INCREMENT: Final = """
local key = KEYS[1]
local amount = ARGV[1]
local ttl = tonumber(ARGV[2])

local existed = redis.call('EXISTS', key)
local new_val = redis.call('INCRBYFLOAT', key, amount)
if existed == 0 then
    redis.call('EXPIRE', key, ttl)
end

return new_val
"""

CLAIM: Final = """
local current = redis.call('GET', KEYS[1])
if current == false then
  redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2])
  return ARGV[1]
end
if ARGV[3] then
  local decoded, stored = pcall(cjson.decode, current)
  if ARGV[4] then
    local legacy = current
    if decoded and type(stored) == 'string' then legacy = stored end
    for _, eligible in ipairs(cjson.decode(ARGV[3])) do
      if eligible['model_id'] == legacy then
        redis.call('EXPIRE', KEYS[1], ARGV[2])
        return current
      end
    end
  end
  if decoded and type(stored) == 'table' then
    for _, eligible in ipairs(cjson.decode(ARGV[3])) do
      local matches = true
      for key, value in pairs(eligible) do
        if stored[key] ~= value then matches = false; break end
      end
      for key, _ in pairs(stored) do
        if eligible[key] == nil then matches = false; break end
      end
      if matches then
        redis.call('EXPIRE', KEYS[1], ARGV[2])
        return current
      end
    end
  end
  redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2])
  return ARGV[1]
end
if current == ARGV[1] then
  redis.call('EXPIRE', KEYS[1], ARGV[2])
end
return current
"""

BLOCK_TTLS: Final = "return {redis.call('TTL', KEYS[2]), redis.call('TTL', KEYS[4])}"

RECORD_FAILURE: Final = (
    "local function bump(count_key, block_key, limit) "
    "local blocked = redis.call('TTL', block_key) "
    "if blocked > 0 then return blocked end "
    "local count = redis.call('INCR', count_key) "
    "if redis.call('TTL', count_key) < 0 then redis.call('EXPIRE', count_key, ARGV[3]) end "
    "if count > limit then redis.call('SET', block_key, '1', 'EX', ARGV[4]) return tonumber(ARGV[4]) end "
    "return 0 end "
    "local user_block = bump(KEYS[1], KEYS[2], tonumber(ARGV[1])) "
    "local source_block = 0 "
    "if tonumber(ARGV[2]) > 0 and user_block == 0 then "
    "source_block = bump(KEYS[3], KEYS[4], tonumber(ARGV[2])) end "
    "return {user_block, source_block}"
)

LEASE_ACQUIRE: Final = "return redis.call('SET', KEYS[1], ARGV[1], 'NX', 'PX', ARGV[2])"
LEASE_EXTEND: Final = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('PEXPIRE', KEYS[1], ARGV[2])
end
return 0
"""
LEASE_RELEASE: Final = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""
LEASE_HELD: Final = "return redis.call('EXISTS', KEYS[1])"
