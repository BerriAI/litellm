-- GCRA (Generic Cell Rate Algorithm) token-bucket admission.
--
-- Vendored from the redis-cell CL.THROTTLE algorithm so governor stays portable
-- on managed Redis (ElastiCache, MemoryDB) where loading a custom module is not
-- possible. One theoretical-arrival-time key per scope, no background cleanup,
-- never deletes a key on the decision path.
--
-- KEYS[1] = gcra_key
-- ARGV[1] = period_seconds (int)
-- ARGV[2] = capacity       (int; events per period)
-- ARGV[3] = burst          (int; extra burst allowance beyond the steady rate)
-- ARGV[4] = cost           (int; events this request consumes, usually 1)
--
-- Returns: { limited(0|1), remaining, retry_after_seconds, reset_after_seconds }

local time_reply = redis.call('TIME')
local now = tonumber(time_reply[1]) + (tonumber(time_reply[2]) / 1000000)

local period   = tonumber(ARGV[1])
local capacity = tonumber(ARGV[2])
local burst    = tonumber(ARGV[3])
local cost     = tonumber(ARGV[4])

if capacity <= 0 or period <= 0 then
    return { 1, 0, period, period }
end

local emission_interval = period / capacity
local tolerance = emission_interval * (burst + 1)
local needed = emission_interval * cost

local tat = tonumber(redis.call('GET', KEYS[1]) or now)
if tat < now then
    tat = now
end

local new_tat = tat + needed
local allow_at = new_tat - tolerance
local diff = now - allow_at

if diff < 0 then
    local retry_after = -diff
    local reset_after = tat - now
    return { 1, 0, retry_after, reset_after }
end

local ttl = math.ceil(new_tat - now)
redis.call('SET', KEYS[1], new_tat, 'EX', ttl)

local remaining = math.floor(diff / emission_interval)
local reset_after = new_tat - now
return { 0, remaining, 0, reset_after }
