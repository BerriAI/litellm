/// `_TOUCH_LUA`, byte for byte, so pods on either stack share one script SHA: drop the members
/// last seen before the window, record the caller, keep the set for one window and list it.
pub(super) const TOUCH: &str = "redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[1]) redis.call('ZADD', KEYS[1], ARGV[2], ARGV[3]) redis.call('EXPIRE', KEYS[1], ARGV[4]) return redis.call('ZRANGE', KEYS[1], 0, -1)";
