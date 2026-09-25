"""Source of litellm's Redis Lua scripts, written as Python with redis-lua-py.

Not shipped: ``litellm/caching/_redis_scripts.py`` is generated from this module and
checked in, so litellm itself never imports redis-lua-py. After editing a script here,
regenerate with:

    uv run python -m redis_lua_py generate redis_scripts.scripts --out litellm/caching/_redis_scripts.py

``tests/test_litellm/caching/test_redis_scripts.py`` fails while the generated module
is out of date. Script bodies are compiled to Lua, never run as Python; see
https://ignacemaes.com/redis-lua-py/ for what they may contain.
"""

from __future__ import annotations

from redis_lua_py import Key, cjson, redis, script

# --- caching/redis_cache.py ---


@script
def increment_with_floor(key: Key, amount: int, ttl: int) -> int:
    """Add ``amount`` to ``key``, clamp the result at zero, and give a key without one ``ttl``."""
    count = redis.incrby(key, amount)
    if count < 0:
        count = redis.incrby(key, -count)
    if redis.ttl(key) < 0:
        redis.expire(key, ttl)
    return count


@script
def set_max(key: Key, value: str, ttl: int) -> bytes:
    """Store ``value`` unless ``key`` already holds a number at least as large; return what is stored.

    A ``ttl`` of 0 leaves the key's expiry alone.
    """
    current = redis.get(key)
    if current is None or float(current) < float(value):
        redis.set(key, value)
        if ttl > 0:
            redis.expire(key, ttl)
        return value
    return current


# --- caching/affinity_cache.py ---


@script
def claim_affinity_pin(key: Key, pin: str, ttl: int, eligible_json: str) -> bytes:
    """Return the pin stored at ``key``, claiming it for ``pin`` when there is none.

    With ``eligible_json`` empty, a stored pin always wins and is refreshed only when it
    equals ``pin``. Otherwise it is a JSON list of objects, and a stored pin that is not
    exactly one of them is replaced by ``pin``.
    """
    current = redis.get(key)
    if current is None:
        redis.set(key, pin, "EX", ttl)
        return pin
    if eligible_json:
        matched = False
        # Anything that is not a JSON object fails the key walk below, and so is
        # replaced, the same as an object that matches no eligible value.
        try:
            stored = cjson.decode(current)
            for eligible in cjson.decode(eligible_json):
                matches = True
                for name, value in eligible.items():
                    if stored.get(name) != value:
                        matches = False
                        break
                for name in stored.keys():  # noqa: SIM118  # .keys() compiles to pairs(); a bare loop over an untraced table would walk it as a list
                    if eligible.get(name) is None:
                        matches = False
                        break
                if matches:
                    matched = True
                    break
        except Exception:
            matched = False
        if matched:
            redis.expire(key, ttl)
            return current
        redis.set(key, pin, "EX", ttl)
        return pin
    if current == pin:
        redis.expire(key, ttl)
    return current


# --- proxy/auth/login_throttle.py ---
# Keys: pair counter, pair block, source counter, source block (one cluster slot via the
# source hash tag). Both scripts return [pair block TTL, source block TTL]; 0 or below
# means not blocked.


@script
def login_block_ttls(pair_counter: Key, pair_block: Key, source_counter: Key, source_block: Key) -> list[int]:
    return [redis.ttl(pair_block), redis.ttl(source_block)]


@script
def record_login_failure(
    pair_counter: Key,
    pair_block: Key,
    source_counter: Key,
    source_block: Key,
    pair_limit: int,
    source_limit: int,
    window_seconds: int,
    block_seconds: int,
) -> list[int]:
    """Count one failed login, blocking a scope that goes over its limit.

    A ``source_limit`` of 0 turns the source scope off, and a blocked pair stops
    counting against its source.
    """

    def bump(count_key: Key, block_key: Key, limit: int) -> int:
        blocked = redis.ttl(block_key)
        if blocked > 0:
            return blocked
        count = redis.incr(count_key)
        if redis.ttl(count_key) < 0:
            redis.expire(count_key, window_seconds)
        if count > limit:
            redis.set(block_key, "1", "EX", block_seconds)
            return block_seconds
        return 0

    pair_block_ttl = bump(pair_counter, pair_block, pair_limit)
    source_block_ttl = 0
    if source_limit > 0 and pair_block_ttl == 0:
        source_block_ttl = bump(source_counter, source_block, source_limit)
    return [pair_block_ttl, source_block_ttl]


# --- proxy/hooks/batch_enqueued_tokens.py ---


@script
def reserve_enqueued_tokens(key: Key, amount: int, ttl: int, limit: int) -> list[int]:
    """Add ``amount`` unless that takes the counter over ``limit``.

    Returns [1, new total] when reserved and [0, current total] when refused.
    """
    current = float(redis.get(key) or "0")
    if current + amount > limit:
        return [0, current]
    updated = redis.incrby(key, amount)
    redis.expire(key, ttl)
    return [1, updated]


@script
def refund_enqueued_tokens(key: Key, amount: int) -> int:
    """Take ``amount`` back off the counter, deleting it once nothing is reserved."""
    if redis.decrby(key, amount) <= 0:
        redis.delete(key)
    return 1


@script
def save_reservation(key: Key, record: str, ttl: int) -> int:
    redis.set(key, record, "EX", ttl)
    return 1


@script
def pop_reservation(key: Key, ttl: int) -> bytes | None:
    """Return the saved reservation and leave an empty marker for ``ttl``, so it pops once."""
    value = redis.get(key)
    if value:
        redis.set(key, "", "EX", ttl)
    return value


# --- proxy/hooks/max_iterations_limiter.py ---


@script
def increment_session_iterations(key: Key, ttl: int) -> int:
    """Count one more iteration, starting the ``ttl`` on the first."""
    current = redis.incr(key)
    if current == 1:
        redis.expire(key, ttl)
    return current


# --- proxy/hooks/max_budget_per_session_limiter.py ---


@script
def increment_session_spend(key: Key, amount: str, ttl: int) -> bytes:
    """Add ``amount`` to the session's spend, starting the ``ttl`` when the key is new."""
    existed = redis.exists(key)
    new_value = redis.incrbyfloat(key, amount)
    if existed == 0:
        redis.expire(key, ttl)
    return new_value


# --- Lock ownership: proxy/db/db_transaction_queue/pod_lock_manager.py and
# proxy/_experimental/mcp_server/outbound_credentials/redis_distributed_lock.py ---


@script
def delete_if_owner(key: Key, token: str) -> int:
    """Delete the lock only while it still holds ``token``, so an expired holder cannot free the next one's."""
    if redis.get(key) == token:
        return redis.delete(key)
    return 0


@script
def pexpire_if_owner(key: Key, token: str, milliseconds: int) -> int:
    if redis.get(key) == token:
        return redis.pexpire(key, milliseconds)
    return 0


# --- proxy/hooks/parallel_request_limiter_v3.py ---


@script
def batch_rate_limit(keys: list[Key], now: int, window_size: int) -> list[bytes | int]:
    """Count one request against each (window key, counter key) pair in ``keys``.

    A window that is missing or older than ``window_size`` restarts at ``now``, clearing
    the requests and tokens counters that share its prefix. Returns a flat
    [window start, counter, ...] per pair.
    """
    results: list[bytes | int] = []
    for i in range(0, len(keys), 2):
        window_key = keys[i]
        counter_key = keys[i + 1]
        window_start = redis.get(window_key)
        if window_start is None or now - float(window_start) >= window_size:
            prefix = window_key[: -len(":window")]
            redis.delete(prefix + ":requests", prefix + ":tokens")
            redis.set(window_key, str(now))
            redis.set(counter_key, 1)
            redis.expire(window_key, window_size)
            redis.expire(counter_key, window_size)
            results.append(str(now))
            results.append(1)
        else:
            counter = redis.incr(counter_key)
            # The window can outlive a counter created after it (a tokens key made after
            # the requests key it shares the window with), so give that counter one too.
            if redis.ttl(counter_key) == -1:
                redis.expire(counter_key, window_size)
            results.append(window_start)
            results.append(counter)
    return results


@script
def check_and_increment_by_n(keys: list[Key], args: list[str | int | float]) -> list[int]:
    """Check and increment one or more descriptors, all or nothing.

    ``keys`` holds a (window key, counter key) pair per descriptor, and ``args`` a
    (limit, increment, ttl seconds, window seconds) quadruple. If any descriptor would go
    over its limit, nothing is written and the result is
    [1, descriptor index (from 1), current counter, limit]. Otherwise it is
    [0, new counter, window start, ...] per descriptor.

    Windows are timed by the Redis server clock (TIME), not a client timestamp, so
    replicas with skewed wall clocks agree on when a window resets.
    """
    now = float(redis.time()[0])
    descriptor_count = len(keys) // 2

    # Pass 1: read and validate, returning before any write if one is over its limit.
    expired_windows: list[bool] = []
    window_starts: list[bytes | None] = []
    for i in range(descriptor_count):
        window_key = keys[i * 2]
        counter_key = keys[i * 2 + 1]
        limit = float(args[i * 4])
        increment = float(args[i * 4 + 1])
        window_size = float(args[i * 4 + 3])
        window_start = redis.get(window_key)
        window_expired = window_start is None or now - float(window_start) >= window_size
        current_counter = 0.0
        if not window_expired:
            current_counter = float(redis.get(counter_key) or 0)
        if increment > 0:
            blocked = current_counter + increment > limit
        else:
            blocked = current_counter >= limit
        if blocked:
            return [1, i + 1, current_counter, limit]
        expired_windows.append(window_expired)
        window_starts.append(window_start)

    # Pass 2: every check passed, so apply the increments.
    results = [0]
    reset_windows: dict[Key, bool] = {}
    for i in range(descriptor_count):
        window_key = keys[i * 2]
        counter_key = keys[i * 2 + 1]
        increment = float(args[i * 4 + 1])
        ttl = float(args[i * 4 + 2])
        window_size = float(args[i * 4 + 3])
        if expired_windows[i]:
            active_window_start = now
            if window_key not in reset_windows:
                prefix = window_key[: -len(":window")]
                redis.delete(prefix + ":requests", prefix + ":tokens")
                reset_windows[window_key] = True
            redis.set(window_key, str(now))
            redis.set(counter_key, increment)
            redis.expire(window_key, window_size)
            if ttl > 0:
                redis.expire(counter_key, ttl)
            results.append(increment)
        else:
            active_window_start = float(window_starts[i])
            new_counter = redis.incrby(counter_key, increment)
            if redis.ttl(counter_key) == -1 and ttl > 0:
                redis.expire(counter_key, ttl)
            results.append(new_counter)
        results.append(active_window_start)
    return results


@script
def window_guarded_token_increment(keys: list[Key], args: list[str | int | float]) -> list[int]:
    """Increment each counter only while its window is still the one the caller saw.

    ``keys`` holds (window key, counter key) pairs and ``args`` an (expected window start,
    increment, ttl seconds) triple per pair. Returns [1, new counter] per applied pair and
    [0, current counter] per pair whose window moved on.
    """
    results: list[int] = []
    for i in range(0, len(keys), 2):
        window_key = keys[i]
        counter_key = keys[i + 1]
        arg_base = (i // 2) * 3
        expected_window_start = args[arg_base]
        increment = float(args[arg_base + 1])
        ttl = float(args[arg_base + 2])
        active_window_start = redis.get(window_key)
        if active_window_start is not None and active_window_start == expected_window_start:
            new_counter = redis.incrby(counter_key, increment)
            if redis.ttl(counter_key) == -1 and ttl > 0:
                redis.expire(counter_key, ttl)
            results.append(1)
            results.append(new_counter)
        else:
            results.append(0)
            results.append(float(redis.get(counter_key) or 0))
    return results


@script
def parallel_acquire(keys: list[Key], args: list[str | int | float]) -> list[int]:
    """Take one slot in each max_parallel_requests gauge, or none at all.

    A gauge is a sorted set of per-request slot ids scored by acquire time (Redis server
    clock). In-flight requests are counted with ZCARD after pruning slots older than the
    slot TTL, so unlike the windowed RPM/TPM counters a gauge is never reset while requests
    are in flight, a rejected request never occupies a slot, and a slot leaked by a crashed
    worker heals after the slot TTL even under continuous traffic.

    ``args`` holds a (limit, slot ttl seconds, slot id) triple per key. Returns
    [0, in flight, ...] per key on success and [1, key index (from 1), in flight, limit]
    when a gauge is full.
    """
    now = float(redis.time()[0])
    for i in range(len(keys)):
        limit = float(args[i * 3])
        slot_ttl = float(args[i * 3 + 1])
        redis.zremrangebyscore(keys[i], "-inf", now - slot_ttl)
        in_flight = redis.zcard(keys[i])
        if in_flight + 1 > limit:
            return [1, i + 1, in_flight, limit]
    results = [0]
    for i in range(len(keys)):
        slot_ttl = float(args[i * 3 + 1])
        slot_id = args[i * 3 + 2]
        redis.zadd(keys[i], now, slot_id)
        redis.expire(keys[i], slot_ttl)
        results.append(redis.zcard(keys[i]))
    return results


@script
def parallel_release(keys: list[Key], slot_ids: list[str]) -> list[int]:
    """Free this request's slot in each gauge, returning the in-flight count left per key.

    ZREM of an absent member is a no-op, so a release without a matching acquire
    (proxy-side rejection, double-fired callback, slot already expired) can never free a
    slot owned by another request.
    """
    results: list[int] = []
    for i in range(len(keys)):
        redis.zrem(keys[i], slot_ids[i])
        results.append(redis.zcard(keys[i]))
    return results


@script
def parallel_count(keys: list[Key], slot_ttls: list[int]) -> list[int]:
    """Read the in-flight count per gauge, pruning expired slots so leaked ones don't count."""
    now = float(redis.time()[0])
    results: list[int] = []
    for i in range(len(keys)):
        redis.zremrangebyscore(keys[i], "-inf", now - slot_ttls[i])
        results.append(redis.zcard(keys[i]))
    return results


@script
def increment_tokens(keys: list[Key], args: list[str | int | float]) -> list[bytes]:
    """INCRBYFLOAT each key, keeping any expiry it already has.

    ``args`` holds an (increment, ttl seconds) pair per key. A ttl above 0 is set only on
    a key that has no expiry yet, so repeated increments never push it back.
    """
    results: list[bytes] = []
    for i in range(len(keys)):
        key = keys[i]
        new_value = redis.incrbyfloat(key, float(args[i * 2]))
        ttl_seconds = float(args[i * 2 + 1])
        if ttl_seconds > 0 and redis.ttl(key) == -1:
            redis.expire(key, ttl_seconds)
        results.append(new_value)
    return results
