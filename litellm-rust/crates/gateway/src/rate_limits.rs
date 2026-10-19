//! The rate limits hook: refuses a request when its key, user, team, team member or end user is
//! over its requests or tokens per window, or its key has too many requests in flight, with the
//! proxy's counters, order, messages and headers.
//!
//! The counters are the proxy's own Redis keys, `{<scope>:<id>}:window|requests|tokens` and
//! `{api_key:<hash>}:max_parallel_requests`, changed by scripts with the proxy's semantics, so
//! gateway replicas and Python proxies sharing that Redis enforce one limit. Windows are fixed and
//! start on their first request. A request counts against requests whether or not it is let
//! through; tokens are reserved up front from an estimate and settled to the real count when the
//! response ends. Without Redis, or while it is unreachable, the counters live in this replica's
//! memory.

use std::{
    collections::HashMap,
    sync::{Arc, LazyLock, Mutex},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use futures_util::future::BoxFuture;
use http::{HeaderName, HeaderValue};
use litellm_config::Config;
use litellm_db::PgPool;
use litellm_gateway_auth::keys::Limits;
use litellm_gateway_hooks::{
    Facts, Hook, KeyFacts, Record, Refusal, RequestEdit, RequestFacts, Route, Settle, Targets,
};
use redis::aio::ConnectionManager;
use serde_json::{Map, Value};

use crate::{coordination::Coordination, end_users::EndUsers, spend::civil};

const DEFAULT_WINDOW: u64 = 60;
/// How long an in-flight slot counts before it is presumed leaked.
const SLOT_TTL: u64 = 3600;
/// The output tokens assumed at most when a request does not cap them.
const MAX_OUTPUT_FLOOR: i64 = 1024;
/// Windows and slots held in memory before expired ones are dropped.
const MAX_LOCAL: usize = 100_000;
const FAIL_CLOSED: &str = "Rate limit enforcement unavailable: request counters could not be verified against Redis, and fail_closed_rate_limit_enforcement is enabled, so the request was rejected to avoid exceeding the configured rate limit. Retry shortly.";

/// Admits a request in one round trip, with the proxy's semantics for each counter: counts it on
/// every `(window, requests)` pair, starting a window that is missing or over; unless one is over
/// its limit, takes an in-flight slot unless the gauge is full; then adds the token estimate to
/// every `(window, tokens)` pair, or to none (returning the slot) when one would go over.
///
/// KEYS: each requests limit's window and counter, the in-flight set when the key caps it, then
/// each tokens limit's window and counter. ARGV: now, window size, the requests limits' count and
/// limits, the slot (`1` or `0`, limit, ttl, id), the token increment, then each tokens limit's
/// limit, ttl and window size.
///
/// Replies `{code, request counts..., in_flight, rest...}`: code 0 admitted with each tokens
/// count, 1 over a requests limit (no in-flight), 2 the gauge was full, 3 over the tokens limit
/// at `index` holding `current`.
const ADMIT: &str = r#"
local now = tonumber(ARGV[1])
local window_size = tonumber(ARGV[2])
local requests = tonumber(ARGV[3])
local arg = 4
local key = 1
local reply = { 0 }
local over = false
for i = 1, requests do
    local limit = tonumber(ARGV[arg])
    arg = arg + 1
    local window_key = KEYS[key]
    local counter_key = KEYS[key + 1]
    key = key + 2
    local window_start = redis.call('GET', window_key)
    local counter
    if not window_start or (now - tonumber(window_start)) >= window_size then
        local prefix = string.sub(window_key, 1, -(#':window') - 1)
        redis.call('DEL', prefix .. ':requests', prefix .. ':tokens')
        redis.call('SET', window_key, tostring(now))
        redis.call('SET', counter_key, 1)
        redis.call('EXPIRE', window_key, window_size)
        redis.call('EXPIRE', counter_key, window_size)
        counter = 1
    else
        counter = redis.call('INCR', counter_key)
        if redis.call('TTL', counter_key) == -1 then
            redis.call('EXPIRE', counter_key, window_size)
        end
    end
    if counter > limit then
        over = true
    end
    table.insert(reply, counter)
end
if over then
    reply[1] = 1
    return reply
end

local slot_key
local slot_id = ARGV[arg + 3]
local in_flight = 0
if ARGV[arg] == '1' then
    local limit = tonumber(ARGV[arg + 1])
    local slot_ttl = tonumber(ARGV[arg + 2])
    slot_key = KEYS[key]
    key = key + 1
    local time = tonumber(redis.call('TIME')[1])
    redis.call('ZREMRANGEBYSCORE', slot_key, '-inf', time - slot_ttl)
    in_flight = redis.call('ZCARD', slot_key)
    if in_flight + 1 > limit then
        reply[1] = 2
        table.insert(reply, in_flight)
        return reply
    end
    redis.call('ZADD', slot_key, time, slot_id)
    redis.call('EXPIRE', slot_key, slot_ttl)
    in_flight = redis.call('ZCARD', slot_key)
end
table.insert(reply, in_flight)
arg = arg + 4

local increment = tonumber(ARGV[arg])
arg = arg + 1
local first = key
local count = (#KEYS - first + 1) / 2
if count == 0 then
    return reply
end
local time = tonumber(redis.call('TIME')[1])
local state = {}
for i = 1, count do
    local window_key = KEYS[first + (i - 1) * 2]
    local counter_key = KEYS[first + (i - 1) * 2 + 1]
    local limit = tonumber(ARGV[arg + (i - 1) * 3])
    local size = tonumber(ARGV[arg + (i - 1) * 3 + 2])
    local window_start = redis.call('GET', window_key)
    local expired = (not window_start) or ((time - tonumber(window_start)) >= size)
    local current = 0
    if not expired then
        current = tonumber(redis.call('GET', counter_key) or 0)
    end
    local blocked
    if increment > 0 then
        blocked = current + increment > limit
    else
        blocked = current >= limit
    end
    if blocked then
        if slot_key then
            redis.call('ZREM', slot_key, slot_id)
        end
        reply[1] = 3
        table.insert(reply, i)
        table.insert(reply, current)
        return reply
    end
    state[i] = expired
end
local reset = {}
for i = 1, count do
    local window_key = KEYS[first + (i - 1) * 2]
    local counter_key = KEYS[first + (i - 1) * 2 + 1]
    local ttl = tonumber(ARGV[arg + (i - 1) * 3 + 1])
    local size = tonumber(ARGV[arg + (i - 1) * 3 + 2])
    if state[i] then
        if not reset[window_key] then
            local prefix = string.sub(window_key, 1, -(#':window') - 1)
            redis.call('DEL', prefix .. ':requests', prefix .. ':tokens')
            reset[window_key] = true
        end
        redis.call('SET', window_key, tostring(time))
        redis.call('SET', counter_key, increment)
        redis.call('EXPIRE', window_key, size)
        if ttl > 0 then
            redis.call('EXPIRE', counter_key, ttl)
        end
        table.insert(reply, increment)
    else
        local counter = redis.call('INCRBY', counter_key, increment)
        if redis.call('TTL', counter_key) == -1 and ttl > 0 then
            redis.call('EXPIRE', counter_key, ttl)
        end
        table.insert(reply, counter)
    end
end
return reply
"#;

/// Counts a request on every `(window, requests)` pair, starting a window that is missing or
/// over, and returns each window's start and count. The caller compares; the count always rises.
const REQUESTS: &str = r#"
local results = {}
local now = tonumber(ARGV[1])
local window_size = tonumber(ARGV[2])
for i = 1, #KEYS, 2 do
    local window_key = KEYS[i]
    local counter_key = KEYS[i + 1]
    local window_start = redis.call('GET', window_key)
    if not window_start or (now - tonumber(window_start)) >= window_size then
        local prefix = string.sub(window_key, 1, -(#':window') - 1)
        redis.call('DEL', prefix .. ':requests', prefix .. ':tokens')
        redis.call('SET', window_key, tostring(now))
        redis.call('SET', counter_key, 1)
        redis.call('EXPIRE', window_key, window_size)
        redis.call('EXPIRE', counter_key, window_size)
        table.insert(results, tostring(now))
        table.insert(results, 1)
    else
        local counter = redis.call('INCR', counter_key)
        if redis.call('TTL', counter_key) == -1 then
            redis.call('EXPIRE', counter_key, window_size)
        end
        table.insert(results, window_start)
        table.insert(results, counter)
    end
end
return results
"#;

/// Adds the increment to every `(window, tokens)` pair, or to none when one would go over its
/// limit: `{0, count, start, ...}` or `{1, index, current, limit}`.
const RESERVE: &str = r#"
local now = tonumber(redis.call('TIME')[1])
local count = #KEYS / 2
local state = {}
for i = 1, count do
    local window_key = KEYS[(i - 1) * 2 + 1]
    local counter_key = KEYS[(i - 1) * 2 + 2]
    local base = (i - 1) * 4 + 1
    local limit = tonumber(ARGV[base])
    local increment = tonumber(ARGV[base + 1])
    local window_size = tonumber(ARGV[base + 3])
    local window_start = redis.call('GET', window_key)
    local expired = (not window_start) or ((now - tonumber(window_start)) >= window_size)
    local current = 0
    if not expired then
        current = tonumber(redis.call('GET', counter_key) or 0)
    end
    local blocked
    if increment > 0 then
        blocked = current + increment > limit
    else
        blocked = current >= limit
    end
    if blocked then
        return { 1, i, current, limit }
    end
    state[i] = { expired, window_start }
end
local results = { 0 }
local reset = {}
for i = 1, count do
    local window_key = KEYS[(i - 1) * 2 + 1]
    local counter_key = KEYS[(i - 1) * 2 + 2]
    local base = (i - 1) * 4 + 1
    local increment = tonumber(ARGV[base + 1])
    local ttl = tonumber(ARGV[base + 2])
    local window_size = tonumber(ARGV[base + 3])
    if state[i][1] then
        if not reset[window_key] then
            local prefix = string.sub(window_key, 1, -(#':window') - 1)
            redis.call('DEL', prefix .. ':requests', prefix .. ':tokens')
            reset[window_key] = true
        end
        redis.call('SET', window_key, tostring(now))
        redis.call('SET', counter_key, increment)
        redis.call('EXPIRE', window_key, window_size)
        if ttl > 0 then
            redis.call('EXPIRE', counter_key, ttl)
        end
        table.insert(results, increment)
        table.insert(results, now)
    else
        local counter = redis.call('INCRBY', counter_key, increment)
        if redis.call('TTL', counter_key) == -1 and ttl > 0 then
            redis.call('EXPIRE', counter_key, ttl)
        end
        table.insert(results, counter)
        table.insert(results, tonumber(state[i][2]))
    end
end
return results
"#;

/// Adds each token delta, giving a counter without an expiry the window's.
const CHARGE: &str = r#"
local results = {}
for i = 1, #KEYS do
    local value = redis.call('INCRBYFLOAT', KEYS[i], tonumber(ARGV[i * 2 - 1]))
    local ttl = tonumber(ARGV[i * 2])
    if ttl and ttl > 0 and redis.call('TTL', KEYS[i]) == -1 then
        redis.call('EXPIRE', KEYS[i], ttl)
    end
    table.insert(results, value)
end
return results
"#;

/// Sent by hash, loaded once per Redis.
static ADMIT_SCRIPT: LazyLock<redis::Script> = LazyLock::new(|| redis::Script::new(ADMIT));
static REQUESTS_SCRIPT: LazyLock<redis::Script> = LazyLock::new(|| redis::Script::new(REQUESTS));
static RESERVE_SCRIPT: LazyLock<redis::Script> = LazyLock::new(|| redis::Script::new(RESERVE));
static CHARGE_SCRIPT: LazyLock<redis::Script> = LazyLock::new(|| redis::Script::new(CHARGE));

#[derive(Clone, Debug, PartialEq)]
pub struct RateLimitSettings {
    /// `LITELLM_RATE_LIMIT_WINDOW_SIZE`, in seconds; 60 by default.
    pub window: u64,
    /// `general_settings.fail_closed_rate_limit_enforcement`: refuse with a 503 rather than fall
    /// back to this replica's counters while Redis is unreachable.
    pub fail_closed: bool,
    /// `litellm_settings.max_end_user_budget_id`: the limits of end users without their own.
    pub end_user_budget_id: Option<String>,
}

impl Default for RateLimitSettings {
    fn default() -> Self {
        Self {
            window: DEFAULT_WINDOW,
            fail_closed: false,
            end_user_budget_id: None,
        }
    }
}

impl RateLimitSettings {
    pub fn from_config(config: &Config) -> Self {
        let general = &config.general_settings.additional_fields;
        Self {
            window: std::env::var("LITELLM_RATE_LIMIT_WINDOW_SIZE")
                .ok()
                .and_then(|window| window.trim().parse().ok())
                .filter(|window| *window > 0)
                .unwrap_or(DEFAULT_WINDOW),
            fail_closed: match general.get("fail_closed_rate_limit_enforcement") {
                Some(litellm_config::Value::Bool(flag)) => *flag,
                Some(litellm_config::Value::String(text)) => text.eq_ignore_ascii_case("true"),
                _ => false,
            },
            end_user_budget_id: config
                .litellm_settings
                .additional_fields
                .get("max_end_user_budget_id")
                .and_then(litellm_config::Value::as_str)
                .filter(|id| !id.is_empty())
                .map(str::to_owned),
        }
    }
}

/// One scope a request counts against: `{key}:{value}` with its limits.
#[derive(Clone, Debug, PartialEq)]
struct Descriptor {
    key: &'static str,
    value: String,
    limits: Limits,
    max_parallel_requests: Option<i64>,
}

impl Descriptor {
    fn prefix(&self) -> String {
        format!("{{{}:{}}}", self.key, self.value)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Kind {
    Requests,
    Tokens,
    Parallel,
}

impl Kind {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Requests => "requests",
            Self::Tokens => "tokens",
            Self::Parallel => "max_parallel_requests",
        }
    }
}

/// Where a descriptor stands after this request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Status {
    descriptor: usize,
    kind: Kind,
    limit: i64,
    remaining: i64,
}

/// A counter to check and add to: `{prefix}:requests` or `{prefix}:tokens` in its window.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Pair {
    pub(crate) prefix: String,
    pub(crate) kind: Kind,
    pub(crate) limit: i64,
    pub(crate) increment: i64,
}

/// Every pair's count after adding, or the pair at `index` that would go over, which held
/// `current`.
#[derive(Debug, PartialEq)]
pub(crate) enum Reserved {
    Counts(Vec<i64>),
    Over { index: usize, current: i64 },
}

/// Redis could not be used and the settings say to refuse rather than fall back.
pub(crate) struct Unavailable;

#[derive(Debug)]
struct Window {
    start: u64,
    requests: i64,
    tokens: i64,
}

/// This replica's counters, used without Redis or while it is unreachable.
#[derive(Default)]
struct Local {
    windows: HashMap<String, Window>,
    slots: HashMap<String, HashMap<String, Instant>>,
}

impl Local {
    fn window(&mut self, prefix: &str, now: u64, size: u64) -> (&mut Window, bool) {
        if self.windows.len() >= MAX_LOCAL {
            self.windows
                .retain(|_, window| now.saturating_sub(window.start) < size);
        }
        let window = self.windows.entry(prefix.to_owned()).or_insert(Window {
            start: now,
            requests: 0,
            tokens: 0,
        });
        let expired = now.saturating_sub(window.start) >= size;
        (window, expired)
    }

    fn requests(&mut self, prefixes: &[String], now: u64, size: u64) -> Vec<i64> {
        prefixes
            .iter()
            .map(|prefix| {
                let (window, expired) = self.window(prefix, now, size);
                if expired {
                    *window = Window {
                        start: now,
                        requests: 0,
                        tokens: 0,
                    };
                }
                window.requests += 1;
                window.requests
            })
            .collect()
    }

    fn check_and_add(&mut self, pairs: &[Pair], now: u64, size: u64) -> Reserved {
        for (index, pair) in pairs.iter().enumerate() {
            let current = match self.windows.get(&pair.prefix) {
                Some(window) if now.saturating_sub(window.start) < size => match pair.kind {
                    Kind::Requests => window.requests,
                    _ => window.tokens,
                },
                _ => 0,
            };
            if blocked(current, pair.increment, pair.limit) {
                return Reserved::Over { index, current };
            }
        }
        Reserved::Counts(
            pairs
                .iter()
                .map(|pair| {
                    let (window, expired) = self.window(&pair.prefix, now, size);
                    if expired {
                        *window = Window {
                            start: now,
                            requests: 0,
                            tokens: 0,
                        };
                    }
                    let count = match pair.kind {
                        Kind::Requests => &mut window.requests,
                        _ => &mut window.tokens,
                    };
                    *count += pair.increment;
                    *count
                })
                .collect(),
        )
    }

    /// A counter's value in its current window.
    fn read(&self, prefix: &str, kind: Kind, now: u64, size: u64) -> Option<i64> {
        let window = self.windows.get(prefix)?;
        (now.saturating_sub(window.start) < size).then_some(match kind {
            Kind::Requests => window.requests,
            _ => window.tokens,
        })
    }

    fn charge(&mut self, deltas: &[(String, i64)]) {
        for (prefix, delta) in deltas {
            if let Some(window) = self.windows.get_mut(prefix) {
                window.tokens += delta;
            }
        }
    }

    fn acquire(&mut self, prefix: &str, limit: i64, slot: &str) -> Result<i64, i64> {
        let ttl = Duration::from_secs(SLOT_TTL);
        if self.slots.len() >= MAX_LOCAL {
            self.slots.retain(|_, slots| !slots.is_empty());
        }
        let slots = self.slots.entry(prefix.to_owned()).or_default();
        slots.retain(|_, taken| taken.elapsed() < ttl);
        let in_flight = slots.len() as i64;
        if in_flight + 1 > limit {
            return Err(in_flight);
        }
        slots.insert(slot.to_owned(), Instant::now());
        Ok(in_flight + 1)
    }

    fn release(&mut self, prefix: &str, slot: &str) {
        if let Some(slots) = self.slots.get_mut(prefix) {
            slots.remove(slot);
        }
    }
}

fn blocked(current: i64, increment: i64, limit: i64) -> bool {
    match increment > 0 {
        true => current + increment > limit,
        false => current >= limit,
    }
}

/// Where the counters live.
pub(crate) struct Store {
    redis: Option<Arc<Coordination>>,
    local: Mutex<Local>,
    window: u64,
    fail_closed: bool,
}

impl Store {
    pub(crate) fn new(redis: Option<Arc<Coordination>>, window: u64, fail_closed: bool) -> Self {
        Self {
            redis,
            local: Mutex::default(),
            window,
            fail_closed,
        }
    }

    fn connection(&self) -> Option<(ConnectionManager, &Coordination)> {
        let redis = self.redis.as_deref()?;
        Some((redis.connection()?, redis))
    }

    fn local(&self) -> std::sync::MutexGuard<'_, Local> {
        self.local.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Logs why Redis was not used; refuses when the settings say to.
    fn fall_back(&self, error: Option<redis::RedisError>) -> Result<(), Unavailable> {
        match error {
            Some(error) => {
                tracing::warn!(%error, "rate limit counters failed; using this replica's")
            }
            None => {
                tracing::debug!("Redis not connected; using this replica's rate limit counters")
            }
        }
        match self.fail_closed {
            true => Err(Unavailable),
            false => Ok(()),
        }
    }

    /// Counts the request on each prefix's window and returns the counts.
    pub(crate) async fn requests(&self, prefixes: &[String]) -> Result<Vec<i64>, Unavailable> {
        let now = epoch_seconds();
        if self.redis.is_some() {
            let error = match self.connection() {
                Some((mut connection, redis)) => {
                    let mut invocation = REQUESTS_SCRIPT.prepare_invoke();
                    for prefix in prefixes {
                        invocation
                            .key(redis.namespaced(&format!("{prefix}:window")))
                            .key(redis.namespaced(&format!("{prefix}:requests")));
                    }
                    invocation.arg(now).arg(self.window);
                    match integers(invocation.invoke_async(&mut connection).await) {
                        Ok(reply) if reply.len() == prefixes.len() * 2 => {
                            return Ok(reply.iter().skip(1).step_by(2).copied().collect());
                        }
                        Ok(_) => Some(malformed()),
                        Err(error) => Some(error),
                    }
                }
                None => None,
            };
            self.fall_back(error)?;
        }
        Ok(self.local().requests(prefixes, now, self.window))
    }

    /// Counts the request, takes its in-flight slot and reserves its tokens, stopping at the first
    /// limit it is over: one script with Redis, else this replica's counters.
    async fn admit(&self, plan: &Plan) -> Result<(Admission, bool), Unavailable> {
        if self.redis.is_some() {
            let error = match self.connection() {
                Some((mut connection, redis)) => {
                    match admit_redis(&mut connection, redis, plan, self.window).await {
                        Ok(admission) => return Ok((admission, true)),
                        Err(error) => Some(error),
                    }
                }
                None => None,
            };
            self.fall_back(error)?;
        }
        Ok((self.admit_local(plan), false))
    }

    fn admit_local(&self, plan: &Plan) -> Admission {
        let mut local = self.local();
        let now = epoch_seconds();
        let prefixes: Vec<String> = plan
            .requests
            .iter()
            .map(|(prefix, _)| prefix.clone())
            .collect();
        let requests = local.requests(&prefixes, now, self.window);
        if plan
            .requests
            .iter()
            .zip(&requests)
            .any(|((_, limit), count)| count > limit)
        {
            return Admission {
                requests,
                outcome: Outcome::RequestsOver,
            };
        }
        let in_flight = match &plan.slot {
            Some(slot) => match local.acquire(&slot.prefix, slot.limit, &slot.id) {
                Ok(in_flight) => Some(in_flight),
                Err(in_flight) => {
                    return Admission {
                        requests,
                        outcome: Outcome::SlotFull(in_flight),
                    };
                }
            },
            None => None,
        };
        let tokens = match plan.tokens.is_empty() {
            true => Vec::new(),
            false => match local.check_and_add(&plan.token_pairs(), now, self.window) {
                Reserved::Counts(counts) => counts,
                Reserved::Over { index, current } => {
                    if let Some(slot) = &plan.slot {
                        local.release(&slot.prefix, &slot.id);
                    }
                    return Admission {
                        requests,
                        outcome: Outcome::TokensOver { index, current },
                    };
                }
            },
        };
        Admission {
            requests,
            outcome: Outcome::Admitted { in_flight, tokens },
        }
    }

    async fn release(&self, prefix: &str, slot: &str, shared: bool) {
        if shared && let Some((mut connection, redis)) = self.connection() {
            let key = redis.namespaced(&format!("{prefix}:max_parallel_requests"));
            if let Err(error) = redis::cmd("ZREM")
                .arg(key)
                .arg(slot)
                .query_async::<i64>(&mut connection)
                .await
            {
                tracing::warn!(%error, "releasing an in-flight slot failed; it expires on its own");
            }
            return;
        }
        self.local().release(prefix, slot);
    }

    /// Adds every pair's increment, or none when one would go over its limit. `true` when the
    /// counters are in Redis.
    pub(crate) async fn check_and_add(
        &self,
        pairs: &[Pair],
    ) -> Result<(Reserved, bool), Unavailable> {
        if self.redis.is_some() {
            let error = match self.connection() {
                Some((mut connection, redis)) => {
                    let mut invocation = RESERVE_SCRIPT.prepare_invoke();
                    for pair in pairs {
                        invocation
                            .key(redis.namespaced(&format!("{}:window", pair.prefix)))
                            .key(redis.namespaced(&format!(
                                "{}:{}",
                                pair.prefix,
                                pair.kind.name()
                            )));
                    }
                    for pair in pairs {
                        invocation
                            .arg(pair.limit)
                            .arg(pair.increment)
                            .arg(self.window)
                            .arg(self.window);
                    }
                    match integers(invocation.invoke_async(&mut connection).await) {
                        Ok(reply) if reply.first() == Some(&0) => {
                            return Ok((
                                Reserved::Counts(
                                    reply.iter().skip(1).step_by(2).copied().collect(),
                                ),
                                true,
                            ));
                        }
                        Ok(reply) => match reply.as_slice() {
                            [1, index, current, _] if (1..=pairs.len() as i64).contains(index) => {
                                return Ok((
                                    Reserved::Over {
                                        index: (*index - 1) as usize,
                                        current: *current,
                                    },
                                    true,
                                ));
                            }
                            _ => Some(malformed()),
                        },
                        Err(error) => Some(error),
                    }
                }
                None => None,
            };
            self.fall_back(error)?;
        }
        Ok((
            self.local()
                .check_and_add(pairs, epoch_seconds(), self.window),
            false,
        ))
    }

    /// A counter's current value, `None` when it is not set.
    pub(crate) async fn read(&self, prefix: &str, kind: Kind) -> Option<i64> {
        if let Some((mut connection, redis)) = self.connection() {
            let key = redis.namespaced(&format!("{prefix}:{}", kind.name()));
            match redis::cmd("GET")
                .arg(key)
                .query_async::<Option<String>>(&mut connection)
                .await
            {
                Ok(value) => {
                    return value
                        .and_then(|value| value.parse::<f64>().ok())
                        .map(|value| value as i64);
                }
                Err(error) => tracing::warn!(%error, "reading a rate limit counter failed"),
            }
        }
        self.local()
            .read(prefix, kind, epoch_seconds(), self.window)
    }

    /// Adds each delta to its prefix's tokens.
    pub(crate) async fn charge(&self, deltas: Vec<(String, i64)>, shared: bool) {
        if shared && let Some((mut connection, redis)) = self.connection() {
            let mut invocation = CHARGE_SCRIPT.prepare_invoke();
            for (prefix, _) in &deltas {
                invocation.key(redis.namespaced(&format!("{prefix}:tokens")));
            }
            for (_, delta) in &deltas {
                invocation.arg(*delta).arg(self.window);
            }
            if let Err(error) = invocation
                .invoke_async::<redis::Value>(&mut connection)
                .await
            {
                tracing::warn!(%error, "settling reserved tokens failed");
            }
            return;
        }
        self.local().charge(&deltas);
    }
}

/// What one request asks of the counters, in the proxy's order.
struct Plan {
    /// Each requests limit: its prefix and limit.
    requests: Vec<(String, i64)>,
    slot: Option<SlotPlan>,
    /// Each tokens limit: its prefix and limit.
    tokens: Vec<(String, i64)>,
    increment: i64,
}

impl Plan {
    /// The tokens limits as the pairs to reserve the increment on.
    fn token_pairs(&self) -> Vec<Pair> {
        self.tokens
            .iter()
            .map(|(prefix, limit)| Pair {
                prefix: prefix.clone(),
                kind: Kind::Tokens,
                limit: *limit,
                increment: self.increment,
            })
            .collect()
    }
}

struct SlotPlan {
    prefix: String,
    limit: i64,
    id: String,
}

/// Each requests count, then how far the request got.
#[derive(Debug, PartialEq)]
struct Admission {
    requests: Vec<i64>,
    outcome: Outcome,
}

#[derive(Debug, PartialEq)]
enum Outcome {
    RequestsOver,
    SlotFull(i64),
    TokensOver {
        index: usize,
        current: i64,
    },
    Admitted {
        in_flight: Option<i64>,
        tokens: Vec<i64>,
    },
}

async fn admit_redis(
    connection: &mut ConnectionManager,
    redis: &Coordination,
    plan: &Plan,
    window: u64,
) -> redis::RedisResult<Admission> {
    let mut invocation = ADMIT_SCRIPT.prepare_invoke();
    invocation
        .arg(epoch_seconds())
        .arg(window)
        .arg(plan.requests.len());
    for (prefix, limit) in &plan.requests {
        invocation
            .key(redis.namespaced(&format!("{prefix}:window")))
            .key(redis.namespaced(&format!("{prefix}:requests")))
            .arg(*limit);
    }
    match &plan.slot {
        Some(slot) => {
            invocation
                .key(redis.namespaced(&format!("{}:max_parallel_requests", slot.prefix)))
                .arg(1)
                .arg(slot.limit)
                .arg(SLOT_TTL)
                .arg(&slot.id);
        }
        None => {
            invocation.arg(0).arg(0).arg(0).arg("");
        }
    }
    invocation.arg(plan.increment);
    for (prefix, limit) in &plan.tokens {
        invocation
            .key(redis.namespaced(&format!("{prefix}:window")))
            .key(redis.namespaced(&format!("{prefix}:tokens")))
            .arg(*limit)
            .arg(window)
            .arg(window);
    }
    let reply = integers(invocation.invoke_async(connection).await)?;
    let counted = plan.requests.len();
    let (code, rest) = reply.split_first().ok_or_else(malformed)?;
    if rest.len() < counted {
        return Err(malformed());
    }
    let (requests, rest) = rest.split_at(counted);
    let requests = requests.to_vec();
    let outcome = match (code, rest) {
        (1, []) => Outcome::RequestsOver,
        (2, [in_flight]) => Outcome::SlotFull(*in_flight),
        (3, [_, index, current]) if (1..=plan.tokens.len() as i64).contains(index) => {
            Outcome::TokensOver {
                index: (*index - 1) as usize,
                current: *current,
            }
        }
        (0, [in_flight, tokens @ ..]) if tokens.len() == plan.tokens.len() => Outcome::Admitted {
            in_flight: plan.slot.as_ref().map(|_| *in_flight),
            tokens: tokens.to_vec(),
        },
        _ => return Err(malformed()),
    };
    Ok(Admission { requests, outcome })
}

fn malformed() -> redis::RedisError {
    redis::RedisError::from((
        redis::ErrorKind::UnexpectedReturnType,
        "unexpected rate limit script reply",
    ))
}

/// A script's array reply as integers; counters may come back as strings.
fn integers(reply: redis::RedisResult<Vec<redis::Value>>) -> redis::RedisResult<Vec<i64>> {
    reply?
        .iter()
        .map(|value| {
            let text = match value {
                redis::Value::Int(number) => return Ok(*number),
                redis::Value::BulkString(bytes) => std::str::from_utf8(bytes).ok(),
                redis::Value::SimpleString(text) => Some(text.as_str()),
                _ => None,
            };
            text.and_then(|text| text.parse::<f64>().ok())
                .map(|number| number as i64)
                .ok_or_else(malformed)
        })
        .collect()
}

fn epoch_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// `%Y-%m-%d %H:%M:%S UTC`, as the proxy prints when a limit resets.
fn utc(seconds: u64) -> String {
    let (year, month, day) = civil((seconds / 86_400) as i64);
    let time = seconds % 86_400;
    format!(
        "{year:04}-{month:02}-{day:02} {:02}:{:02}:{:02} UTC",
        time / 3600,
        time % 3600 / 60,
        time % 60
    )
}

/// The proxy's token estimate for a request: a token per four characters of message text,
/// plus the output it may produce for every candidate.
#[derive(Debug, PartialEq)]
struct Estimate {
    tokens: i64,
    /// The output cap to send upstream when the request has none and the limits are small.
    implicit_max_tokens: Option<i64>,
}

fn estimate(
    body: &Map<String, Value>,
    key: Option<&KeyFacts>,
    model: &str,
    min_tpm: i64,
) -> Estimate {
    let chars: usize = body
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|message| match message.get("content") {
            Some(Value::String(text)) => text.chars().count(),
            Some(Value::Array(blocks)) => blocks
                .iter()
                .filter_map(|block| block.get("text").and_then(Value::as_str))
                .map(|text| text.chars().count())
                .sum(),
            _ => 0,
        })
        .sum();
    let chars = i64::try_from(chars).unwrap_or(i64::MAX);
    let input = match chars {
        0 => 0,
        chars => (chars / 4).max(1),
    };
    let integer = |value: &Value| match value {
        Value::Number(number) => number.as_f64().map(|number| number as i64),
        Value::String(text) => text.trim().parse::<f64>().ok().map(|number| number as i64),
        _ => None,
    };
    let cap = ["max_tokens", "max_completion_tokens"]
        .iter()
        .filter_map(|field| body.get(*field).and_then(integer))
        .max();
    let configured = key.and_then(|key| {
        [&key.metadata, &key.team_metadata]
            .into_iter()
            .find_map(|metadata| {
                metadata
                    .get("default_estimated_output_tokens_per_model")
                    .and_then(|models| models.get(model))
                    .and_then(integer)
                    .or_else(|| {
                        metadata
                            .get("default_estimated_output_tokens")
                            .and_then(integer)
                    })
            })
    });
    let floor = (min_tpm / 4).clamp(1, MAX_OUTPUT_FLOOR);
    let output = match (cap, configured) {
        (Some(cap), _) => cap,
        (None, Some(configured)) => configured,
        (None, None) if chars == 0 => 0,
        (None, None) => input.max(floor),
    };
    let candidates = ["n", "best_of"]
        .iter()
        .filter_map(|field| body.get(*field).and_then(integer))
        .fold(1, i64::max);
    Estimate {
        tokens: input
            .saturating_add(output.saturating_mul(candidates))
            .max(1),
        implicit_max_tokens: (floor < MAX_OUTPUT_FLOOR && cap.is_none())
            .then(|| floor.max(configured.unwrap_or(0))),
    }
}

/// The `rate_limits` hook.
pub struct RateLimiter {
    settings: RateLimitSettings,
    store: Arc<Store>,
    end_users: Option<EndUsers>,
}

impl RateLimiter {
    /// `redis`, when given, holds the shared counters; `database` gives end users' limits.
    pub fn new(
        settings: RateLimitSettings,
        redis: Option<Arc<Coordination>>,
        database: Option<PgPool>,
    ) -> Self {
        Self {
            store: Arc::new(Store::new(redis, settings.window, settings.fail_closed)),
            settings,
            end_users: database.map(EndUsers::new),
        }
    }

    pub fn for_gateway(
        config: &Config,
        redis: Option<Arc<Coordination>>,
        database: Option<PgPool>,
    ) -> Self {
        Self::new(RateLimitSettings::from_config(config), redis, database)
    }

    /// The scopes this request counts against, in the proxy's order.
    async fn descriptors(&self, request: &RequestFacts) -> Vec<Descriptor> {
        let Some(key) = request.key.as_deref() else {
            return Vec::new();
        };
        let limits = &key.limits;
        let mut descriptors = Vec::new();
        let mut push = |key: &'static str, value: String, limits: Limits, parallel: Option<i64>| {
            if limits.is_set() || parallel.is_some() {
                descriptors.push(Descriptor {
                    key,
                    value,
                    limits,
                    max_parallel_requests: parallel,
                });
            }
        };
        if !key.hash.is_empty() {
            push(
                "api_key",
                key.hash.clone(),
                limits.key,
                limits.max_parallel_requests,
            );
        }
        if let Some(user_id) = &key.user_id {
            push("user", user_id.clone(), limits.user, None);
        }
        if let Some(team_id) = &key.team_id {
            push("team", team_id.clone(), limits.team, None);
        }
        if let (Some(team_id), Some(user_id)) = (&key.team_id, &key.user_id) {
            push(
                "team_member",
                format!("{team_id}:{user_id}"),
                limits.team_member,
                None,
            );
        }
        if let (Some(end_user_id), Some(end_users)) = (request.end_user.as_deref(), &self.end_users)
            && let Some(end_user) = end_users
                .get(
                    end_user_id,
                    self.settings.end_user_budget_id.as_deref(),
                    key.budgets.end_user_budget_id.as_deref(),
                )
                .await
        {
            push(
                "end_user",
                end_user_id.to_owned(),
                Limits {
                    rpm: end_user.limits.rpm,
                    tpm: end_user.limits.tpm,
                },
                None,
            );
        }
        descriptors
    }

    fn refusal(&self, request: &RequestFacts, descriptor: &Descriptor, status: Status) -> Refusal {
        let reset_at = utc(epoch_seconds() + self.settings.window);
        let message = format!(
            "Rate limit exceeded for {}: {}. Limit type: {}. Current limit: {}, Remaining: {}. Limit resets at: {reset_at}",
            descriptor.key,
            descriptor.value,
            status.kind.name(),
            status.limit,
            status.remaining.max(0),
        );
        let mut refusal = throttled(
            request.route,
            message,
            vec![
                ("retry-after", self.settings.window.to_string()),
                ("rate_limit_type", status.kind.name().to_owned()),
                ("reset_at", reset_at),
            ],
        );
        for (name, value) in key_limit_headers(request.key.as_deref()) {
            refusal = refusal.with_header(name, value);
        }
        refusal
    }

    fn unavailable(&self, request: &RequestFacts) -> Refusal {
        unavailable(request.route)
    }
}

/// The proxy's 429 from one of its own limiters: on Chat Completions its error body with the
/// limiter's headers, on `/v1/messages` Anthropic's envelope, where the proxy drops them.
pub(crate) fn throttled(
    route: Route,
    message: String,
    headers: Vec<(&'static str, String)>,
) -> Refusal {
    match route {
        Route::Messages => Refusal::new(429, "rate_limit_error", message).anthropic(),
        Route::ChatCompletions => {
            let mut refusal = Refusal::new(429, "throttling_error", message).without_param();
            for (name, value) in headers {
                if let Ok(value) = HeaderValue::from_str(&value) {
                    refusal = refusal.with_header(HeaderName::from_static(name), value);
                }
            }
            refusal
        }
    }
}

/// The proxy's 503 when counters cannot be verified and enforcement fails closed.
pub(crate) fn unavailable(route: Route) -> Refusal {
    match route {
        Route::Messages => Refusal::new(503, "api_error", FAIL_CLOSED).anthropic(),
        Route::ChatCompletions => {
            Refusal::new(503, "internal_server_error", FAIL_CLOSED).without_param()
        }
    }
}

/// `x-litellm-key-tpm-limit` and `x-litellm-key-rpm-limit`, which the proxy sends whenever the
/// key sets them.
fn key_limit_headers(key: Option<&KeyFacts>) -> Vec<(HeaderName, HeaderValue)> {
    let Some(key) = key else {
        return Vec::new();
    };
    [
        ("x-litellm-key-tpm-limit", key.limits.key.tpm),
        ("x-litellm-key-rpm-limit", key.limits.key.rpm),
    ]
    .into_iter()
    .filter_map(|(name, limit)| Some((HeaderName::from_static(name), HeaderValue::from(limit?))))
    .collect()
}

impl Hook for RateLimiter {
    fn name(&self) -> &'static str {
        "rate_limits"
    }

    fn reads(&self) -> Facts {
        Facts::KEY | Facts::END_USER | Facts::REQUEST_BODY | Facts::USAGE
    }

    fn writes(&self) -> Targets {
        Targets::ADMISSION | Targets::UPSTREAM_BODY | Targets::RESPONSE_HEADERS
    }

    fn run<'a>(
        &'a self,
        request: &'a Arc<RequestFacts>,
        edit: &'a mut RequestEdit,
    ) -> BoxFuture<'a, Result<Option<Box<dyn Settle>>, Refusal>> {
        Box::pin(async move {
            let descriptors = self.descriptors(request).await;
            if descriptors.is_empty() {
                return Ok(None);
            }
            // Requests per window (counted whether or not the request goes through), then
            // requests in flight for the key only, then tokens per window, reserved from an
            // estimate and settled when the response ends.
            let windowed: Vec<usize> = (0..descriptors.len())
                .filter(|index| descriptors[*index].limits.rpm.is_some())
                .collect();
            let parallel = descriptors
                .iter()
                .position(|descriptor| descriptor.max_parallel_requests.is_some());
            let limited: Vec<(usize, i64)> = descriptors
                .iter()
                .enumerate()
                .filter_map(|(index, descriptor)| Some((index, descriptor.limits.tpm?)))
                .collect();
            let mut increment = 0;
            if let Some(min_tpm) = limited.iter().map(|(_, limit)| *limit).min() {
                let body: Map<String, Value> =
                    serde_json::from_slice(&request.body()).unwrap_or_default();
                let estimate = estimate(
                    &body,
                    request.key.as_deref(),
                    &request.requested_model,
                    min_tpm,
                );
                if let Some(max_tokens) = estimate.implicit_max_tokens {
                    edit.set_upstream_body_field("max_tokens", Value::from(max_tokens));
                }
                increment = estimate.tokens;
            }
            let plan = Plan {
                requests: windowed
                    .iter()
                    .map(|index| {
                        let descriptor = &descriptors[*index];
                        (
                            descriptor.prefix(),
                            descriptor.limits.rpm.unwrap_or_default(),
                        )
                    })
                    .collect(),
                slot: parallel.map(|index| SlotPlan {
                    prefix: descriptors[index].prefix(),
                    limit: descriptors[index].max_parallel_requests.unwrap_or_default(),
                    id: uuid::Uuid::new_v4().simple().to_string(),
                }),
                tokens: limited
                    .iter()
                    .map(|(index, limit)| (descriptors[*index].prefix(), *limit))
                    .collect(),
                increment,
            };
            let (admission, shared) = self
                .store
                .admit(&plan)
                .await
                .map_err(|Unavailable| self.unavailable(request))?;

            let mut statuses: Vec<Status> = windowed
                .iter()
                .zip(&admission.requests)
                .map(|(index, count)| {
                    let limit = descriptors[*index].limits.rpm.unwrap_or_default();
                    Status {
                        descriptor: *index,
                        kind: Kind::Requests,
                        limit,
                        remaining: limit - count,
                    }
                })
                .collect();
            let (in_flight, counts) = match admission.outcome {
                Outcome::RequestsOver => {
                    let over = statuses
                        .iter()
                        .find(|status| status.remaining < 0)
                        .copied()
                        .unwrap_or(statuses[0]);
                    return Err(self.refusal(request, &descriptors[over.descriptor], over));
                }
                Outcome::SlotFull(in_flight) => {
                    let (index, slot) = (parallel.unwrap_or_default(), &plan.slot);
                    let limit = slot.as_ref().map_or(0, |slot| slot.limit);
                    let status = Status {
                        descriptor: index,
                        kind: Kind::Parallel,
                        limit,
                        remaining: limit - in_flight,
                    };
                    return Err(self.refusal(request, &descriptors[index], status));
                }
                Outcome::TokensOver { index, current } => {
                    let (descriptor, limit) = limited[index];
                    let status = Status {
                        descriptor,
                        kind: Kind::Tokens,
                        limit,
                        remaining: limit - current,
                    };
                    return Err(self.refusal(request, &descriptors[descriptor], status));
                }
                Outcome::Admitted { in_flight, tokens } => (in_flight, tokens),
            };
            let slot = match (parallel, plan.slot, in_flight) {
                (Some(index), Some(slot), Some(in_flight)) => {
                    statuses.push(Status {
                        descriptor: index,
                        kind: Kind::Parallel,
                        limit: slot.limit,
                        remaining: (slot.limit - in_flight).max(0),
                    });
                    Some(Slot {
                        prefix: slot.prefix,
                        id: slot.id,
                        shared,
                    })
                }
                _ => None,
            };
            for ((index, limit), count) in limited.iter().zip(&counts) {
                statuses.push(Status {
                    descriptor: *index,
                    kind: Kind::Tokens,
                    limit: *limit,
                    remaining: (limit - count).max(0),
                });
            }
            let tokens = (!plan.tokens.is_empty()).then(|| Tokens {
                prefixes: plan.tokens.into_iter().map(|(prefix, _)| prefix).collect(),
                reserved: increment,
                shared,
            });

            // The proxy sends where each limit stands on complete responses only.
            if !request.stream {
                for status in &statuses {
                    let key = descriptors[status.descriptor].key;
                    let kind = status.kind.name();
                    for (name, value) in [
                        (
                            format!("x-ratelimit-{key}-remaining-{kind}"),
                            status.remaining,
                        ),
                        (format!("x-ratelimit-{key}-limit-{kind}"), status.limit),
                    ] {
                        if let Ok(name) = HeaderName::from_bytes(name.as_bytes()) {
                            edit.add_response_header(name, HeaderValue::from(value));
                        }
                    }
                }
            }
            for (name, value) in key_limit_headers(request.key.as_deref()) {
                edit.add_response_header(name, value);
            }
            if slot.is_none() && tokens.is_none() {
                return Ok(None);
            }
            Ok(Some(Box::new(Held {
                store: self.store.clone(),
                slot,
                tokens,
            }) as Box<dyn Settle>))
        })
    }
}

struct Slot {
    prefix: String,
    id: String,
    shared: bool,
}

struct Tokens {
    prefixes: Vec<String>,
    reserved: i64,
    shared: bool,
}

/// What a request holds until it ends: an in-flight slot and reserved tokens.
struct Held {
    store: Arc<Store>,
    slot: Option<Slot>,
    tokens: Option<Tokens>,
}

impl Settle for Held {
    fn settle(self: Box<Self>, record: &Record) {
        let Held {
            store,
            slot,
            tokens,
        } = *self;
        let charge = tokens.and_then(|tokens| {
            // The proxy counts every token but cache reads. Without usage, a success or a request
            // cut off after it reached the provider keeps its reservation, and a failure returns it.
            let actual = match (&record.outcome.usage, record.succeeded()) {
                (Some(usage), _) => i64::try_from(usage.total_tokens() - usage.cache_read_tokens)
                    .unwrap_or(i64::MAX),
                (None, true) => tokens.reserved,
                (None, false) if record.cut_off_after_dispatch() => tokens.reserved,
                (None, false) => 0,
            };
            let delta = actual - tokens.reserved;
            (delta != 0).then(|| {
                (
                    tokens
                        .prefixes
                        .into_iter()
                        .map(|prefix| (prefix, delta))
                        .collect::<Vec<_>>(),
                    tokens.shared,
                )
            })
        });
        tokio::spawn(async move {
            if let Some(slot) = slot {
                store.release(&slot.prefix, &slot.id, slot.shared).await;
            }
            if let Some((deltas, shared)) = charge {
                store.charge(deltas, shared).await;
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn body(value: Value) -> Map<String, Value> {
        match value {
            Value::Object(object) => object,
            _ => unreachable!(),
        }
    }

    #[rstest]
    #[case::utc_epoch(0, "1970-01-01 00:00:00 UTC")]
    #[case::utc_later(1_792_238_465, "2026-10-17 12:01:05 UTC")]
    fn prints_reset_times_like_the_proxy(#[case] seconds: u64, #[case] expected: &str) {
        assert_eq!(utc(seconds), expected);
    }

    #[rstest]
    // 40 characters: 10 input tokens; the small limit floors output at 100_000 / 4 = 1024.
    #[case::uncapped(json!({"messages": [{"role": "user", "content": "a".repeat(40)}]}), 100_000, 10 + 1024, None)]
    // A small limit floors output at 400 / 4 = 100 and caps the request there.
    #[case::small_limit(json!({"messages": [{"role": "user", "content": "a".repeat(40)}]}), 400, 110, Some(100))]
    #[case::capped(json!({"max_tokens": 50, "messages": [{"role": "user", "content": [{"type": "text", "text": "a".repeat(8)}]}]}), 400, 52, None)]
    #[case::candidates(json!({"max_completion_tokens": 5, "n": 3, "messages": [{"role": "user", "content": "abcd"}]}), 100_000, 16, None)]
    #[case::no_text(json!({"messages": []}), 100_000, 1, None)]
    fn estimates_tokens_like_the_proxy(
        #[case] request: Value,
        #[case] min_tpm: i64,
        #[case] tokens: i64,
        #[case] implicit_max_tokens: Option<i64>,
    ) {
        assert_eq!(
            estimate(&body(request), None, "m", min_tpm),
            Estimate {
                tokens,
                implicit_max_tokens
            }
        );
    }

    #[rstest]
    fn configured_output_estimates_come_from_the_key_then_the_team() {
        let metadata = |value: Value| Arc::new(body(value));
        let key = KeyFacts {
            metadata: metadata(json!({"default_estimated_output_tokens_per_model": {"m": 7}})),
            team_metadata: metadata(json!({"default_estimated_output_tokens": 9})),
            ..KeyFacts::default()
        };
        let request = body(json!({"messages": [{"role": "user", "content": "abcd"}]}));
        assert_eq!(estimate(&request, Some(&key), "m", 100_000).tokens, 8);
        assert_eq!(estimate(&request, Some(&key), "other", 100_000).tokens, 10);
    }

    fn reserve(local: &mut Local, increment: i64, now: u64) -> Reserved {
        let pair = Pair {
            prefix: "{api_key:h}".into(),
            kind: Kind::Tokens,
            limit: 10,
            increment,
        };
        local.check_and_add(&[pair], now, 60)
    }

    #[rstest]
    fn local_windows_count_reserve_and_reset() {
        let mut local = Local::default();
        let prefixes = vec!["{api_key:h}".to_owned()];
        assert_eq!(local.requests(&prefixes, 100, 60), [1]);
        assert_eq!(local.requests(&prefixes, 159, 60), [2]);
        assert_eq!(reserve(&mut local, 6, 159), Reserved::Counts(vec![6]));
        assert_eq!(
            reserve(&mut local, 6, 159),
            Reserved::Over {
                index: 0,
                current: 6
            }
        );
        local.charge(&[("{api_key:h}".to_owned(), -4)]);
        assert_eq!(reserve(&mut local, 8, 159), Reserved::Counts(vec![10]));
        // The window started at 100 is over at 160: requests and tokens start again.
        assert_eq!(local.requests(&prefixes, 160, 60), [1]);
        assert_eq!(reserve(&mut local, 10, 160), Reserved::Counts(vec![10]));
    }

    #[rstest]
    fn local_slots_fill_and_release() {
        let mut local = Local::default();
        assert_eq!(local.acquire("{api_key:h}", 2, "a"), Ok(1));
        assert_eq!(local.acquire("{api_key:h}", 2, "b"), Ok(2));
        assert_eq!(local.acquire("{api_key:h}", 2, "c"), Err(2));
        local.release("{api_key:h}", "a");
        assert_eq!(local.acquire("{api_key:h}", 2, "c"), Ok(2));
    }
}
