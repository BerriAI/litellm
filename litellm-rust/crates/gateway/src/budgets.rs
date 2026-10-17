//! The budgets hook: refuses a request when its key, team, user, team member or end user is over
//! budget, with the proxy's checks, order and messages, and keeps the proxy's `spend:*` counters
//! in Redis so gateway replicas and Python proxies sharing that Redis see each other's spend.
//!
//! Each counter is read with the stored spend as its floor: a missing counter is seeded from it
//! and one below it is raised to it, as the proxy reseeds and repairs counters. A request then
//! reserves its most expensive possible cost on every counter with a budget, and settles the
//! reservation to its real cost when it ends, so concurrent requests cannot all slip under one
//! budget. Reading, checking and reserving is one Redis script, atomic per request. Without
//! Redis, or while it is unreachable, the counters live in this replica's memory.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, Instant},
};

use futures_util::future::BoxFuture;
use litellm_config::Config;
use litellm_db::{EndUser, PgPool};
use litellm_gateway_hooks::{
    Facts, Hook, Prices, Record, Refusal, RequestEdit, RequestFacts, Settle, Targets, Usage,
};
use redis::aio::ConnectionManager;

use crate::coordination::Coordination;

/// The output tokens the proxy assumes when neither the request nor the model caps them.
const DEFAULT_OUTPUT_TOKENS: u64 = 16_384;
/// The proxy's slack when a reservation exactly meets a budget.
const EPSILON: f64 = 1e-12;
/// How long an end user's budget is reused before it is read again.
const END_USER_TTL: Duration = Duration::from_secs(60);
/// End users held in memory; the cache starts over past this.
const MAX_END_USERS: usize = 10_000;
const FAIL_CLOSED_READ: &str = "{'error': 'Budget enforcement unavailable: current spend could not be verified against Redis or the database, and fail_closed_budget_enforcement is enabled, so the request was rejected to avoid exceeding the configured budget. Retry shortly.'}";

/// Seeds or raises each counter to its floor, runs the budget checks in the order given and
/// reserves the estimate on every counter that takes a reservation, shrinking it to what is
/// left when a counter would go over, as the proxy's reservation does.
const ADMIT: &str = r#"
local ttl = tonumber(ARGV[1])
local estimate = tonumber(ARGV[2])
local n = #KEYS
local current, max, mode, reserve = {}, {}, {}, {}
for i = 1, n do
  local base = 3 + (i - 1) * 4
  local floor = ARGV[base + 1]
  max[i] = tonumber(ARGV[base + 2])
  mode[i] = ARGV[base + 3]
  reserve[i] = ARGV[base + 4] == '1'
  local raw = redis.call('GET', KEYS[i])
  local value = raw and tonumber(raw) or nil
  if value == nil or value < tonumber(floor) then
    redis.call('SET', KEYS[i], floor, 'EX', ttl)
    raw = floor
  end
  current[i] = raw
end
for index in string.gmatch(ARGV[3], '%d+') do
  local i = tonumber(index)
  local value = tonumber(current[i])
  if (mode[i] == '1' and value >= max[i]) or (mode[i] == '2' and value > max[i]) then
    return {'check', i, current[i]}
  end
end
local charged = {}
local applied = {}
local cost = estimate
for i = 1, n do
  charged[i] = 0
  if reserve[i] and cost > 0 then
    local raw = redis.call('INCRBYFLOAT', KEYS[i], cost)
    redis.call('EXPIRE', KEYS[i], ttl)
    charged[i] = cost
    applied[#applied + 1] = i
    local post = tonumber(raw)
    if post > max[i] then
      local remaining = max[i] - (post - cost)
      if remaining <= 1e-12 then
        for _, j in ipairs(applied) do
          redis.call('INCRBYFLOAT', KEYS[j], -charged[j])
        end
        return {'reserve', i, raw}
      end
      for _, j in ipairs(applied) do
        redis.call('INCRBYFLOAT', KEYS[j], -(charged[j] - remaining))
        charged[j] = remaining
      end
      cost = remaining
    end
  end
end
local out = {'ok'}
for i = 1, n do
  out[#out + 1] = string.format('%.17g', charged[i])
end
return out
"#;

/// Adds each request's settled cost: seeds a missing counter from its floor (charging the whole
/// cost, since its reservation was lost), raises one below its floor, then applies the change.
const SETTLE: &str = r#"
local ttl = tonumber(ARGV[1])
for i, key in ipairs(KEYS) do
  local base = 1 + (i - 1) * 3
  local floor, delta, fresh = ARGV[base + 1], ARGV[base + 2], ARGV[base + 3]
  local raw = redis.call('GET', key)
  local value = raw and tonumber(raw) or nil
  if value == nil then
    redis.call('SET', key, floor)
    delta = fresh
  elseif value < tonumber(floor) then
    redis.call('SET', key, floor)
  end
  if tonumber(delta) ~= 0 then
    redis.call('INCRBYFLOAT', key, delta)
  end
  redis.call('EXPIRE', key, ttl)
end
return #KEYS
"#;

#[derive(Clone, Debug, PartialEq)]
pub struct BudgetSettings {
    /// `litellm_settings.budget_exceeded_status_code`, 422 by default.
    pub status: u16,
    /// Off with `general_settings.disable_budget_reservation`.
    pub reservations: bool,
    /// `general_settings.apply_user_budget_to_team_keys`: a user's budget also covers their
    /// team keys, not only their personal ones.
    pub user_budget_on_team_keys: bool,
    /// `litellm_settings.max_end_user_budget_id`: the budget of end users without their own.
    pub end_user_budget_id: Option<String>,
    /// `general_settings.fail_closed_budget_enforcement`: refuse rather than fall back to this
    /// replica's counters while Redis is unreachable.
    pub fail_closed: bool,
    /// `default_redis_ttl` (in `litellm_settings` or `cache_params`), 60 s by default.
    pub ttl: Duration,
}

impl BudgetSettings {
    pub fn from_config(config: &Config) -> Self {
        let general = &config.general_settings.additional_fields;
        let litellm = &config.litellm_settings;
        let ttl = litellm
            .additional_fields
            .get("default_redis_ttl")
            .or_else(|| {
                litellm
                    .cache_params
                    .as_ref()
                    .and_then(|params| params.get("default_redis_ttl"))
            })
            .and_then(number)
            .filter(|seconds| *seconds >= 1.0)
            .unwrap_or(60.0);
        Self {
            status: litellm
                .additional_fields
                .get("budget_exceeded_status_code")
                .and_then(number)
                .and_then(|status| u16::try_from(status as i64).ok())
                .filter(|status| (400..600).contains(status))
                .unwrap_or(422),
            reservations: !flag(general.get("disable_budget_reservation")),
            user_budget_on_team_keys: flag(general.get("apply_user_budget_to_team_keys")),
            end_user_budget_id: litellm
                .additional_fields
                .get("max_end_user_budget_id")
                .and_then(litellm_config::Value::as_str)
                .filter(|id| !id.is_empty())
                .map(str::to_owned),
            fail_closed: flag(general.get("fail_closed_budget_enforcement")),
            ttl: Duration::from_secs(ttl as u64),
        }
    }
}

fn number(value: &litellm_config::Value) -> Option<f64> {
    match value {
        litellm_config::Value::Number(number) => number.as_f64(),
        litellm_config::Value::String(text) => text.trim().parse().ok(),
        _ => None,
    }
}

fn flag(value: Option<&litellm_config::Value>) -> bool {
    match value {
        Some(litellm_config::Value::Bool(flag)) => *flag,
        Some(litellm_config::Value::String(text)) => text.eq_ignore_ascii_case("true"),
        _ => false,
    }
}

/// When a counter refuses at check time.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Check {
    None,
    /// `spend >= max_budget`.
    AtLeast,
    /// `spend > max_budget`, for teams and end users.
    Above,
}

/// Whose counter it is, which names it in refusals.
#[derive(Clone, Debug, PartialEq)]
enum Owner {
    Key {
        hash: String,
        descriptor: String,
    },
    Team(String),
    User(String),
    /// The team member check the proxy runs while building the key: a positive budget.
    Member {
        user_id: String,
        team_id: String,
    },
    /// The proxy's later team member check, which also refuses on a zero budget.
    ZeroBudgetMember {
        user_id: String,
        team_id: String,
    },
    EndUser(String),
}

impl Owner {
    /// The counter, before the namespace, as the proxy names it.
    fn counter(&self) -> String {
        match self {
            Self::Key { hash, .. } => format!("spend:key:{hash}"),
            Self::Team(id) => format!("spend:team:{id}"),
            Self::User(id) => format!("spend:user:{id}"),
            Self::Member { user_id, team_id } | Self::ZeroBudgetMember { user_id, team_id } => {
                format!("spend:team_member:{user_id}:{team_id}")
            }
            Self::EndUser(id) => format!("spend:end_user:{id}"),
        }
    }

    /// The auth-time refusal message.
    fn check_message(&self, spend: f64, max_budget: f64) -> String {
        let (spend, max_budget) = (python_float(spend), python_float(max_budget));
        match self {
            Self::Key { descriptor, .. } => format!(
                "Budget has been exceeded! Key={descriptor} Current cost: {spend}, Max budget: {max_budget}"
            ),
            Self::Team(id) => format!(
                "Budget has been exceeded! Team={id} Current cost: {spend}, Max budget: {max_budget}"
            ),
            Self::User(id) => {
                format!("ExceededBudget: User={id} over budget. Spend={spend}, Budget={max_budget}")
            }
            Self::Member { user_id, team_id } => format!(
                "Budget has been exceeded! TeamMember={user_id}:{team_id} Current cost: {spend}, Max budget: {max_budget}"
            ),
            Self::ZeroBudgetMember { user_id, team_id } => format!(
                "Budget has been exceeded! User={user_id} in Team={team_id} Current cost: {spend}, Max budget: {max_budget}"
            ),
            Self::EndUser(id) => format!(
                "ExceededBudget: End User={id} over budget. Spend={spend}, Budget={max_budget}"
            ),
        }
    }

    /// The reservation's refusal message, which names keys by their hash.
    fn reservation_message(&self, spend: f64, max_budget: f64) -> String {
        let owner = match self {
            Self::Key { hash, .. } => format!("Key={hash}"),
            Self::Team(id) => format!("Team={id}"),
            Self::User(id) => format!("User={id}"),
            Self::Member { user_id, team_id } | Self::ZeroBudgetMember { user_id, team_id } => {
                format!("TeamMember={user_id}:{team_id}")
            }
            Self::EndUser(id) => format!("EndUser={id}"),
        };
        format!(
            "Budget has been exceeded! {owner} Current cost: {}, Max budget: {}",
            python_float(spend),
            python_float(max_budget)
        )
    }
}

/// One spend counter a request touches.
#[derive(Clone, Debug, PartialEq)]
struct Counter {
    owner: Owner,
    /// The stored spend, below which the counter is never read.
    floor: f64,
    max_budget: Option<f64>,
    check: Check,
    reserve: bool,
}

/// The counters a request touches, in the proxy's reservation order, and the order its checks
/// run in.
#[derive(Debug, Default, PartialEq)]
struct Plan {
    counters: Vec<Counter>,
    checks: Vec<usize>,
}

/// How admission went.
#[derive(Debug, PartialEq)]
enum Admitted {
    /// What was reserved on each counter.
    Charged(Vec<f64>),
    /// A check refused, with the counter's spend.
    Over(usize, f64),
    /// The reservation could not fit, with the counter's spend after it.
    Reserved(usize, f64),
}

/// Counters in this replica's memory: the proxy's in-memory spend counters, used without Redis
/// and while it is unreachable.
#[derive(Default)]
struct Local(Mutex<HashMap<String, (f64, Instant)>>);

impl Local {
    fn admit(&self, keys: &[String], plan: &Plan, estimate: f64, ttl: Duration) -> Admitted {
        let mut counters = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let now = Instant::now();
        if counters.len() > 100_000 {
            counters.retain(|_, (_, expires)| *expires > now);
        }
        let mut current = Vec::with_capacity(keys.len());
        for (key, counter) in keys.iter().zip(&plan.counters) {
            let entry = counters
                .entry(key.clone())
                .or_insert((counter.floor, now + ttl));
            if entry.1 <= now || entry.0 < counter.floor {
                *entry = (counter.floor, now + ttl);
            }
            current.push(entry.0);
        }
        for &index in &plan.checks {
            let counter = &plan.counters[index];
            if over(counter.check, current[index], counter.max_budget) {
                return Admitted::Over(index, current[index]);
            }
        }
        let mut charged = vec![0.0; keys.len()];
        let mut applied = Vec::new();
        let mut cost = estimate;
        for (index, counter) in plan.counters.iter().enumerate() {
            if !counter.reserve || cost <= 0.0 {
                continue;
            }
            let Some(max_budget) = counter.max_budget else {
                continue;
            };
            let entry = counters.get_mut(&keys[index]).expect("seeded above");
            entry.0 += cost;
            entry.1 = now + ttl;
            let post = entry.0;
            charged[index] = cost;
            applied.push(index);
            if post > max_budget {
                let remaining = max_budget - (post - cost);
                let refund =
                    |counters: &mut HashMap<String, (f64, Instant)>, index: usize, amount| {
                        if let Some(entry) = counters.get_mut(&keys[index]) {
                            entry.0 -= amount;
                        }
                    };
                if remaining <= EPSILON {
                    for &applied in &applied {
                        refund(&mut counters, applied, charged[applied]);
                    }
                    return Admitted::Reserved(index, post);
                }
                for &applied in &applied {
                    refund(&mut counters, applied, charged[applied] - remaining);
                    charged[applied] = remaining;
                }
                cost = remaining;
            }
        }
        Admitted::Charged(charged)
    }

    fn settle(&self, keys: &[String], changes: &[Change], ttl: Duration) {
        let mut counters = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let now = Instant::now();
        for (key, change) in keys.iter().zip(changes) {
            let entry = counters.entry(key.clone()).or_insert((f64::NAN, now));
            let delta = if entry.0.is_nan() || entry.1 <= now {
                entry.0 = change.floor;
                change.fresh
            } else {
                entry.0 = entry.0.max(change.floor);
                change.delta
            };
            entry.0 += delta;
            entry.1 = now + ttl;
        }
    }
}

fn over(check: Check, spend: f64, max_budget: Option<f64>) -> bool {
    match (check, max_budget) {
        (Check::AtLeast, Some(max_budget)) => spend >= max_budget,
        (Check::Above, Some(max_budget)) => spend > max_budget,
        _ => false,
    }
}

/// How a counter changes when a request ends.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Change {
    floor: f64,
    /// Added to a counter that still exists.
    delta: f64,
    /// Added to a counter that expired while the request ran.
    fresh: f64,
}

/// Where the counters live.
struct Counters {
    redis: Option<Arc<Coordination>>,
    local: Local,
    ttl: Duration,
    fail_closed: bool,
}

/// Redis could not be used and the settings say to refuse rather than fall back.
struct Unavailable;

impl Counters {
    fn connection(&self) -> Option<(ConnectionManager, &Coordination)> {
        let redis = self.redis.as_deref()?;
        Some((redis.connection()?, redis))
    }

    async fn admit(&self, plan: &Plan, estimate: f64) -> Result<(Admitted, bool), Unavailable> {
        let keys: Vec<String> = plan
            .counters
            .iter()
            .map(|counter| counter.owner.counter())
            .collect();
        if self.redis.is_some() {
            match self.connection() {
                Some((mut connection, redis)) => {
                    let keys: Vec<String> = keys.iter().map(|key| redis.namespaced(key)).collect();
                    match admit_redis(&mut connection, &keys, plan, estimate, self.ttl).await {
                        Ok(admitted) => return Ok((admitted, true)),
                        Err(error) => {
                            tracing::warn!(%error, "reading spend counters failed; using this replica's counters")
                        }
                    }
                }
                None => tracing::debug!("Redis not connected; using this replica's spend counters"),
            }
            if self.fail_closed {
                return Err(Unavailable);
            }
        }
        Ok((self.local.admit(&keys, plan, estimate, self.ttl), false))
    }

    async fn settle(&self, keys: Vec<String>, changes: Vec<Change>, shared: bool) {
        if shared && let Some((mut connection, redis)) = self.connection() {
            let keys: Vec<String> = keys.iter().map(|key| redis.namespaced(key)).collect();
            let mut command = redis::cmd("EVAL");
            command
                .arg(SETTLE)
                .arg(keys.len())
                .arg(&keys)
                .arg(self.ttl.as_secs());
            for change in &changes {
                command
                    .arg(change.floor.to_string())
                    .arg(change.delta.to_string())
                    .arg(change.fresh.to_string());
            }
            if let Err(error) = command.query_async::<i64>(&mut connection).await {
                // A counter left half-written would mislead every replica; drop it so the next
                // read reseeds it from the database, as the proxy does.
                tracing::warn!(%error, "updating spend counters failed; dropping them");
                let _: Result<i64, _> = redis::cmd("DEL")
                    .arg(&keys)
                    .query_async(&mut connection)
                    .await;
            }
            return;
        }
        self.local.settle(&keys, &changes, self.ttl);
    }

    /// Keeps reserved counters alive while a request runs, renewing them every half TTL.
    fn lease(&self, keys: Vec<String>) -> Option<tokio::sync::oneshot::Sender<()>> {
        let (mut connection, redis) = self.connection()?;
        let keys: Vec<String> = keys.iter().map(|key| redis.namespaced(key)).collect();
        let ttl = self.ttl;
        let (done, mut finished) = tokio::sync::oneshot::channel::<()>();
        tokio::spawn(async move {
            let mut renew = tokio::time::interval(ttl / 2);
            renew.tick().await;
            loop {
                tokio::select! {
                    _ = &mut finished => return,
                    _ = renew.tick() => {
                        let mut pipeline = redis::pipe();
                        for key in &keys {
                            pipeline.cmd("EXPIRE").arg(key).arg(ttl.as_secs()).ignore();
                        }
                        let _: Result<(), _> = pipeline.query_async(&mut connection).await;
                    }
                }
            }
        });
        Some(done)
    }
}

async fn admit_redis(
    connection: &mut ConnectionManager,
    keys: &[String],
    plan: &Plan,
    estimate: f64,
    ttl: Duration,
) -> redis::RedisResult<Admitted> {
    let mut command = redis::cmd("EVAL");
    command
        .arg(ADMIT)
        .arg(keys.len())
        .arg(keys)
        .arg(ttl.as_secs())
        .arg(estimate.to_string())
        .arg(
            plan.checks
                .iter()
                .map(|index| (index + 1).to_string())
                .collect::<Vec<_>>()
                .join(","),
        );
    for counter in &plan.counters {
        let mode = match (counter.check, counter.max_budget) {
            (_, None) | (Check::None, _) => "0",
            (Check::AtLeast, Some(_)) => "1",
            (Check::Above, Some(_)) => "2",
        };
        command
            .arg(counter.floor.to_string())
            .arg(
                counter
                    .max_budget
                    .map(|max| max.to_string())
                    .unwrap_or_default(),
            )
            .arg(mode)
            .arg(if counter.reserve && counter.max_budget.is_some() {
                "1"
            } else {
                "0"
            });
    }
    let reply: Vec<redis::Value> = command.query_async(connection).await?;
    let text = |value: &redis::Value| -> Option<String> {
        match value {
            redis::Value::BulkString(bytes) => String::from_utf8(bytes.clone()).ok(),
            redis::Value::SimpleString(text) => Some(text.clone()),
            redis::Value::Int(number) => Some(number.to_string()),
            _ => None,
        }
    };
    let malformed = || {
        redis::RedisError::from((
            redis::ErrorKind::UnexpectedReturnType,
            "unexpected budget script reply",
        ))
    };
    let status = reply.first().and_then(text).ok_or_else(malformed)?;
    let float = |value: Option<&redis::Value>| {
        value
            .and_then(text)
            .and_then(|value| value.parse::<f64>().ok())
            .ok_or_else(malformed)
    };
    let index = || {
        reply
            .get(1)
            .and_then(text)
            .and_then(|value| value.parse::<usize>().ok())
            .and_then(|index| index.checked_sub(1))
            .filter(|index| *index < keys.len())
            .ok_or_else(malformed)
    };
    match status.as_str() {
        "ok" => reply[1..]
            .iter()
            .map(|value| float(Some(value)))
            .collect::<Result<_, _>>()
            .map(Admitted::Charged),
        "check" => Ok(Admitted::Over(index()?, float(reply.get(2))?)),
        "reserve" => Ok(Admitted::Reserved(index()?, float(reply.get(2))?)),
        _ => Err(malformed()),
    }
}

/// An end user and the default budget it was read with.
type EndUserKey = (String, Option<String>);

/// End users' budgets, read from the proxy's database and reused for a minute.
struct EndUsers {
    pool: PgPool,
    cache: Mutex<HashMap<EndUserKey, (EndUser, Instant)>>,
}

impl EndUsers {
    async fn get(&self, id: &str, default_budget_id: Option<&str>) -> Option<EndUser> {
        let key = (id.to_owned(), default_budget_id.map(str::to_owned));
        let now = Instant::now();
        if let Some((end_user, read_at)) = self
            .cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&key)
            && now.duration_since(*read_at) < END_USER_TTL
        {
            return Some(end_user.clone());
        }
        match litellm_db::find_end_user(&self.pool, id, default_budget_id).await {
            Ok(end_user) => {
                let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
                if cache.len() >= MAX_END_USERS {
                    cache.clear();
                }
                cache.insert(key, (end_user.clone(), now));
                Some(end_user)
            }
            Err(error) => {
                tracing::warn!(%error, "reading an end user's budget failed; not checking it");
                None
            }
        }
    }
}

/// The `budgets` hook.
pub struct Budgets {
    settings: BudgetSettings,
    counters: Arc<Counters>,
    end_users: Option<EndUsers>,
    /// Each model group's deployment ids, for pricing a request's most expensive outcome.
    groups: HashMap<String, Vec<Arc<str>>>,
    prices: OnceLock<Arc<Prices>>,
}

impl Budgets {
    /// `deployments` pairs each deployment's model group with its id. `redis`, when given, holds
    /// the shared counters; `database` gives end users' budgets.
    pub fn new(
        settings: BudgetSettings,
        deployments: impl IntoIterator<Item = (String, Arc<str>)>,
        redis: Option<Arc<Coordination>>,
        database: Option<PgPool>,
    ) -> Self {
        let mut groups: HashMap<String, Vec<Arc<str>>> = HashMap::new();
        for (group, id) in deployments {
            groups.entry(group).or_default().push(id);
        }
        Self {
            counters: Arc::new(Counters {
                redis,
                local: Local::default(),
                ttl: settings.ttl,
                fail_closed: settings.fail_closed,
            }),
            settings,
            end_users: database.map(|pool| EndUsers {
                pool,
                cache: Mutex::default(),
            }),
            groups,
            prices: OnceLock::new(),
        }
    }

    /// The hook for a gateway: its deployments, the coordination Redis and the proxy database.
    pub fn for_gateway(
        config: &Config,
        inference: &litellm_gateway_inference::Gateway,
        redis: Option<Arc<Coordination>>,
        database: Option<PgPool>,
    ) -> Self {
        let deployments: Vec<(String, Arc<str>)> = config
            .model_list
            .iter()
            .map(|model| model.model_name.clone())
            .zip(inference.models.deployment_ids().map(Arc::from))
            .collect();
        Self::new(
            BudgetSettings::from_config(config),
            deployments,
            redis,
            database,
        )
    }

    /// Every deployment of the requested group has explicit zero prices: the proxy skips budget
    /// checks for free models.
    fn free(&self, request: &RequestFacts) -> bool {
        let (Some(prices), Some(ids)) = (
            self.prices.get(),
            self.groups.get(&*request.requested_model),
        ) else {
            return false;
        };
        let usage = Usage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            ..Usage::default()
        };
        !ids.is_empty() && ids.iter().all(|id| prices.cost(id, &usage) == Some(0.0))
    }

    /// The proxy's estimate of a request's most expensive cost: its input, and as much output
    /// as it may ask for, priced on the group's most expensive deployment. `None` when a
    /// deployment's prices are unknown.
    fn estimate(&self, request: &RequestFacts) -> Option<f64> {
        let prices = self.prices.get()?;
        let ids = self.groups.get(&*request.requested_model)?;
        let input_tokens = request.input_tokens();
        let mut most = None::<f64>;
        for id in ids {
            let output_tokens = request.max_tokens.unwrap_or(DEFAULT_OUTPUT_TOKENS).min(
                prices
                    .max_output_tokens(id)
                    .unwrap_or(DEFAULT_OUTPUT_TOKENS),
            );
            let cost = prices.cost(
                id,
                &Usage {
                    input_tokens,
                    output_tokens,
                    ..Usage::default()
                },
            )?;
            most = Some(most.map_or(cost, |most| most.max(cost)));
        }
        most
    }

    async fn plan(&self, request: &RequestFacts) -> Plan {
        let mut plan = Plan::default();
        let Some(key) = request.key.as_deref() else {
            return plan;
        };
        let budgets = &key.budgets;
        let reserve = |max_budget: Option<f64>| {
            self.settings.reservations && max_budget.is_some_and(|max| max > 0.0 && max.is_finite())
        };
        let finite = |max_budget: Option<f64>| max_budget.filter(|max| max.is_finite());
        let push = |plan: &mut Plan, counter: Counter| {
            plan.counters.push(counter);
            plan.counters.len() - 1
        };
        // Reservation order: key, team, user, end user, team member.
        let mut key_index = None;
        if !key.hash.is_empty() {
            let max_budget = finite(budgets.key.max_budget);
            key_index = Some(push(
                &mut plan,
                Counter {
                    owner: Owner::Key {
                        hash: key.hash.clone(),
                        descriptor: key_descriptor(
                            key.alias.as_deref(),
                            budgets.key_name.as_deref(),
                        ),
                    },
                    floor: budgets.key.spend,
                    max_budget,
                    check: Check::AtLeast,
                    reserve: reserve(max_budget),
                },
            ));
        }
        let mut team_index = None;
        if let Some(team_id) = &key.team_id {
            let team = budgets.team.unwrap_or_default();
            let max_budget = finite(team.max_budget);
            team_index = Some(push(
                &mut plan,
                Counter {
                    owner: Owner::Team(team_id.clone()),
                    floor: team.spend,
                    max_budget,
                    check: Check::Above,
                    reserve: reserve(max_budget),
                },
            ));
        }
        let mut user_index = None;
        if let Some(user_id) = &key.user_id {
            let user = budgets.user.unwrap_or_default();
            let applies = key.team_id.is_none() || self.settings.user_budget_on_team_keys;
            let max_budget = finite(user.max_budget).filter(|_| applies);
            user_index = Some(push(
                &mut plan,
                Counter {
                    owner: Owner::User(user_id.clone()),
                    floor: user.spend,
                    max_budget,
                    check: Check::AtLeast,
                    reserve: reserve(max_budget),
                },
            ));
        }
        let mut end_user_index = None;
        if let Some(end_user_id) = request.end_user.as_deref() {
            let default_budget_id = budgets
                .end_user_budget_id
                .as_deref()
                .or(self.settings.end_user_budget_id.as_deref());
            let end_user = match &self.end_users {
                Some(end_users) => end_users.get(end_user_id, default_budget_id).await,
                None => None,
            }
            .unwrap_or_default();
            let max_budget = finite(end_user.max_budget);
            end_user_index = Some(push(
                &mut plan,
                Counter {
                    owner: Owner::EndUser(end_user_id.to_owned()),
                    floor: end_user.spend.unwrap_or(0.0),
                    max_budget,
                    // The proxy's check needs an end-user row; its reservation does not.
                    check: match end_user.spend {
                        Some(_) => Check::Above,
                        None => Check::None,
                    },
                    reserve: reserve(max_budget),
                },
            ));
        }
        let mut member_index = None;
        if let (Some(user_id), Some(team_id)) = (&key.user_id, &key.team_id) {
            let member = budgets.team_member;
            let max_budget = member.and_then(|member| {
                finite(member.max_budget)
                    .or_else(|| finite(member.default_max_budget).filter(|max| *max > 0.0))
            });
            let (owner, check) = match max_budget {
                Some(max) if max <= 0.0 => (
                    Owner::ZeroBudgetMember {
                        user_id: user_id.clone(),
                        team_id: team_id.clone(),
                    },
                    Check::AtLeast,
                ),
                _ => (
                    Owner::Member {
                        user_id: user_id.clone(),
                        team_id: team_id.clone(),
                    },
                    Check::AtLeast,
                ),
            };
            member_index = Some(push(
                &mut plan,
                Counter {
                    owner,
                    floor: member.map_or(0.0, |member| member.spend),
                    max_budget,
                    check,
                    reserve: reserve(max_budget),
                },
            ));
        }
        // Check order: the positive member budget and the key while the proxy builds the key,
        // then team, user, a zero member budget and the end user in its common checks.
        let positive_member = member_index
            .filter(|index| matches!(plan.counters[*index].owner, Owner::Member { .. }));
        let zero_member = member_index
            .filter(|index| matches!(plan.counters[*index].owner, Owner::ZeroBudgetMember { .. }));
        plan.checks = [
            positive_member,
            key_index,
            team_index,
            user_index,
            zero_member,
            end_user_index,
        ]
        .into_iter()
        .flatten()
        .filter(|index| plan.counters[*index].max_budget.is_some())
        .collect();
        plan
    }

    fn refusal(&self, message: String) -> Refusal {
        Refusal::new(self.settings.status, "budget_exceeded", message).without_param()
    }
}

impl Hook for Budgets {
    fn name(&self) -> &'static str {
        "budgets"
    }

    fn reads(&self) -> Facts {
        Facts::KEY | Facts::END_USER | Facts::INPUT_TOKENS | Facts::COST
    }

    fn writes(&self) -> Targets {
        Targets::ADMISSION
    }

    fn prepare(&self, prices: &Arc<Prices>) {
        let _ = self.prices.set(prices.clone());
    }

    fn run<'a>(
        &'a self,
        request: &'a Arc<RequestFacts>,
        _edit: &'a mut RequestEdit,
    ) -> BoxFuture<'a, Result<Option<Box<dyn Settle>>, Refusal>> {
        Box::pin(async move {
            let plan = self.plan(request).await;
            if plan.counters.is_empty() {
                return Ok(None);
            }
            let checked = !self.free(request);
            let reserving = checked && plan.counters.iter().any(|counter| counter.reserve);
            let estimate = match reserving {
                true => self.estimate(request).filter(|cost| *cost > 0.0),
                false => None,
            };
            let charged = if checked && (!plan.checks.is_empty() || estimate.is_some()) {
                let (admitted, shared) = self
                    .counters
                    .admit(&plan, estimate.unwrap_or(0.0))
                    .await
                    .map_err(|Unavailable| Refusal::new(503, "auth_error", FAIL_CLOSED_READ))?;
                match admitted {
                    Admitted::Charged(charged) => Some((charged, shared)),
                    Admitted::Over(index, spend) => {
                        let counter = &plan.counters[index];
                        let max_budget = counter.max_budget.unwrap_or_default();
                        return Err(self.refusal(counter.owner.check_message(spend, max_budget)));
                    }
                    Admitted::Reserved(index, spend) => {
                        let counter = &plan.counters[index];
                        let max_budget = counter.max_budget.unwrap_or_default();
                        return Err(
                            self.refusal(counter.owner.reservation_message(spend, max_budget))
                        );
                    }
                }
            } else {
                None
            };
            let shared = charged
                .as_ref()
                .map_or(self.counters.redis.is_some(), |(_, shared)| *shared);
            let charged =
                charged.map_or_else(|| vec![0.0; plan.counters.len()], |(charged, _)| charged);
            let reserved: Vec<String> = plan
                .counters
                .iter()
                .zip(&charged)
                .filter(|(_, charged)| **charged != 0.0)
                .map(|(counter, _)| counter.owner.counter())
                .collect();
            let lease = match shared && !reserved.is_empty() {
                true => self.counters.lease(reserved),
                false => None,
            };
            Ok(Some(Box::new(Reservation {
                counters: self.counters.clone(),
                plan,
                charged,
                shared,
                _lease: lease,
            }) as Box<dyn Settle>))
        })
    }
}

/// What a request reserved, settled to its real cost when it ends.
struct Reservation {
    counters: Arc<Counters>,
    plan: Plan,
    charged: Vec<f64>,
    shared: bool,
    /// Dropping it stops the lease.
    _lease: Option<tokio::sync::oneshot::Sender<()>>,
}

impl Settle for Reservation {
    fn settle(self: Box<Self>, record: &Record) {
        let success = (200..300).contains(&record.outcome.status);
        let cost = match success {
            true => record.outcome.cost.unwrap_or(0.0),
            false => 0.0,
        };
        let mut keys = Vec::new();
        let mut changes = Vec::new();
        for (counter, charged) in self.plan.counters.iter().zip(&self.charged) {
            let delta = cost - charged;
            if delta == 0.0 && *charged == 0.0 {
                continue;
            }
            keys.push(counter.owner.counter());
            changes.push(Change {
                floor: counter.floor,
                delta,
                fresh: cost,
            });
        }
        if keys.is_empty() {
            return;
        }
        let Reservation {
            counters,
            shared,
            _lease,
            ..
        } = *self;
        tokio::spawn(async move {
            counters.settle(keys, changes, shared).await;
            drop(_lease);
        });
    }
}

/// The proxy names a key by its alias, adding `key_name` only while it still looks masked.
fn key_descriptor(alias: Option<&str>, key_name: Option<&str>) -> String {
    let alias = alias.filter(|alias| !alias.is_empty()).unwrap_or("key");
    match key_name.filter(|name| is_masked(name)) {
        Some(name) => format!("{alias} ({name})"),
        None => alias.to_owned(),
    }
}

/// The proxy's `_MASKED_KEY_NAME_RE`, `sk-\.\.\.(?:[^\s\x00-\x1f\x7f-\x9f]{4})?`.
fn is_masked(name: &str) -> bool {
    name.strip_prefix("sk-...").is_some_and(|rest| {
        rest.is_empty()
            || (rest.chars().count() == 4
                && rest
                    .chars()
                    .all(|character| !character.is_whitespace() && !character.is_control()))
    })
}

/// Python's `repr` of a float.
fn python_float(value: f64) -> String {
    if value.is_nan() {
        return "nan".into();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf" } else { "-inf" }.into();
    }
    let magnitude = value.abs();
    if magnitude != 0.0 && !(1e-4..1e16).contains(&magnitude) {
        let scientific = format!("{value:e}");
        let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
        let exponent: i32 = exponent.parse().unwrap_or(0);
        let sign = if exponent < 0 { '-' } else { '+' };
        return format!("{mantissa}e{sign}{:02}", exponent.abs());
    }
    format!("{value:?}")
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::no_alias(None, None, "key")]
    #[case::empty_alias(Some(""), None, "key")]
    #[case::alias(Some("prod"), None, "prod")]
    #[case::masked_name(Some("prod"), Some("sk-...abcd"), "prod (sk-...abcd)")]
    #[case::bare_masked_name(None, Some("sk-..."), "key (sk-...)")]
    #[case::unmasked_name(Some("prod"), Some("sk-unmasked90"), "prod")]
    #[case::name_with_space(Some("prod"), Some("sk-...ab d"), "prod")]
    #[case::name_too_long(Some("prod"), Some("sk-...abcde"), "prod")]
    fn describes_keys_like_the_proxy(
        #[case] alias: Option<&str>,
        #[case] name: Option<&str>,
        #[case] expected: &str,
    ) {
        assert_eq!(key_descriptor(alias, name), expected);
    }

    #[rstest]
    #[case::whole(10.0, "10.0")]
    #[case::fraction(0.1, "0.1")]
    #[case::zero(0.0, "0.0")]
    #[case::small(0.0001, "0.0001")]
    #[case::tiny(0.00001, "1e-05")]
    #[case::large(1e15, "1000000000000000.0")]
    #[case::huge(1e16, "1e+16")]
    #[case::huge_fraction(1.5e17, "1.5e+17")]
    #[case::negative(-2.5, "-2.5")]
    #[case::infinite(f64::INFINITY, "inf")]
    fn prints_floats_like_python(#[case] value: f64, #[case] expected: &str) {
        assert_eq!(python_float(value), expected);
    }

    fn counter(owner: Owner, floor: f64, max_budget: Option<f64>, check: Check) -> Counter {
        Counter {
            owner,
            floor,
            max_budget,
            check,
            reserve: max_budget.is_some_and(|max| max > 0.0),
        }
    }

    fn plan(counters: Vec<Counter>) -> (Vec<String>, Plan) {
        let keys = counters.iter().map(|c| c.owner.counter()).collect();
        let checks = (0..counters.len()).collect();
        (keys, Plan { counters, checks })
    }

    #[rstest]
    fn local_counters_check_reserve_shrink_and_refuse() {
        let local = Local::default();
        let ttl = Duration::from_secs(60);
        let (keys, plan) = plan(vec![
            counter(Owner::Team("t".into()), 4.0, Some(10.0), Check::Above),
            counter(Owner::User("u".into()), 0.0, Some(100.0), Check::AtLeast),
        ]);
        // Fits: both counters take the whole estimate.
        assert_eq!(
            local.admit(&keys, &plan, 5.0, ttl),
            Admitted::Charged(vec![5.0, 5.0])
        );
        // Team has 1.0 left: the reservation shrinks to it on every counter.
        assert_eq!(
            local.admit(&keys, &plan, 5.0, ttl),
            Admitted::Charged(vec![1.0, 1.0])
        );
        // Exactly at budget passes the team's strict check, but nothing is left to reserve.
        assert_eq!(
            local.admit(&keys, &plan, 5.0, ttl),
            Admitted::Reserved(0, 15.0)
        );
        // Settling the first request to 0.5 frees 4.5.
        local.settle(
            &keys,
            &[
                Change {
                    floor: 4.0,
                    delta: -4.5,
                    fresh: 0.5,
                },
                Change {
                    floor: 0.0,
                    delta: -4.5,
                    fresh: 0.5,
                },
            ],
            ttl,
        );
        assert_eq!(
            local.admit(&keys, &plan, 1.0, ttl),
            Admitted::Charged(vec![1.0, 1.0])
        );
        // A stored spend above the counter raises it, and the check refuses.
        let (keys, plan) = plan_with_floor(30.0);
        assert_eq!(local.admit(&keys, &plan, 1.0, ttl), Admitted::Over(0, 30.0));
    }

    fn plan_with_floor(floor: f64) -> (Vec<String>, Plan) {
        plan(vec![counter(
            Owner::Team("t".into()),
            floor,
            Some(10.0),
            Check::Above,
        )])
    }

    #[rstest]
    fn refusals_read_like_the_proxy() {
        let key = Owner::Key {
            hash: "abc".into(),
            descriptor: "prod (sk-...abcd)".into(),
        };
        assert_eq!(
            key.check_message(10.0, 10.0),
            "Budget has been exceeded! Key=prod (sk-...abcd) Current cost: 10.0, Max budget: 10.0"
        );
        assert_eq!(
            key.reservation_message(12.5, 10.0),
            "Budget has been exceeded! Key=abc Current cost: 12.5, Max budget: 10.0"
        );
        assert_eq!(
            Owner::EndUser("e".into()).check_message(2.0, 1.0),
            "ExceededBudget: End User=e over budget. Spend=2.0, Budget=1.0"
        );
        assert_eq!(
            Owner::ZeroBudgetMember {
                user_id: "u".into(),
                team_id: "t".into()
            }
            .check_message(0.0, 0.0),
            "Budget has been exceeded! User=u in Team=t Current cost: 0.0, Max budget: 0.0"
        );
    }
}
