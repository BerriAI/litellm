//! The model capacity hook, the proxy's `dynamic_rate_limiter_v3`: caps each model group at the
//! sum of its deployments' `rpm` and `tpm` across every replica, and with `priority_reservation`
//! splits that capacity between priorities once the group is busy.
//!
//! Each request adds one to `{model_saturation_check:<group>}:requests` and checks the group's
//! tokens, all or nothing, with the proxy's keys and scripts. With priorities, the request's pool
//! `{priority_model:<group>:<priority>}` (or `:default_pool`) is held to its share of capacity
//! once the group's saturation reaches `saturation_threshold`, and only counted below it. Tokens
//! are added to both counters when the response ends.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use futures_util::future::BoxFuture;
use http::{HeaderName, HeaderValue};
use litellm_config::Config;
use litellm_gateway_hooks::{
    Facts, Hook, KeyFacts, Record, Refusal, RequestEdit, RequestFacts, Settle, Targets,
};
use serde_json::Value;

use crate::{
    coordination::Coordination,
    rate_limits::{Kind, Pair, RateLimitSettings, Reserved, Store, Unavailable},
};

/// The names a config's `callbacks` list turns this hook on with.
const NAMES: [&str; 2] = ["dynamic_rate_limiter_v3", "model_capacity"];

/// A priority's share of a group's capacity, as `priority_reservation` writes it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Share {
    Percent(f64),
    Rpm(f64),
    Tpm(f64),
}

impl Share {
    /// The share as a fraction of the group, as the proxy's `convert_priority_to_percent`.
    fn fraction(self, capacity: &Capacity) -> f64 {
        match self {
            Self::Percent(value) => value,
            Self::Rpm(value) => match capacity.rpm {
                Some(rpm) if rpm > 0 => value / rpm as f64,
                _ => value,
            },
            Self::Tpm(value) => match capacity.tpm {
                Some(tpm) if tpm > 0 => value / tpm as f64,
                _ => value,
            },
        }
    }
}

/// Which tokens count: `general_settings.token_rate_limit_type`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum TokenCount {
    #[default]
    Total,
    Input,
    Output,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CapacitySettings {
    pub window: u64,
    pub fail_closed: bool,
    /// `litellm_settings.priority_reservation`, in config order.
    pub reservation: Option<Vec<(String, Share)>>,
    /// `priority_reservation_settings.default_priority`: the share of keys without a listed
    /// priority, together.
    pub default_priority: f64,
    /// `priority_reservation_settings.saturation_threshold`.
    pub saturation_threshold: f64,
    /// `priority_reservation_settings.saturation_check_cache_ttl`: how long a group's
    /// saturation is reused before it is read again.
    pub saturation_ttl: Duration,
    pub tokens: TokenCount,
}

impl Default for CapacitySettings {
    fn default() -> Self {
        let rate_limits = RateLimitSettings::default();
        Self {
            window: rate_limits.window,
            fail_closed: rate_limits.fail_closed,
            reservation: None,
            default_priority: 0.25,
            saturation_threshold: 0.5,
            saturation_ttl: Duration::from_secs(60),
            tokens: TokenCount::Total,
        }
    }
}

impl CapacitySettings {
    /// `None` when the config's `callbacks` do not list the hook.
    pub fn from_config(config: &Config) -> Option<Self> {
        let litellm = &config.litellm_settings;
        let listed = litellm.callbacks.iter().flatten().any(|callback| {
            callback
                .as_str()
                .is_some_and(|name| NAMES.contains(&name.trim()))
        });
        if !listed {
            return None;
        }
        let rate_limits = RateLimitSettings::from_config(config);
        let number = |value: &litellm_config::Value| match value {
            litellm_config::Value::Number(number) => number.as_f64(),
            litellm_config::Value::String(text) => text.trim().parse().ok(),
            _ => None,
        };
        let reservation = match litellm.additional_fields.get("priority_reservation") {
            Some(litellm_config::Value::Mapping(mapping)) => Some(
                mapping
                    .iter()
                    .filter_map(|(priority, share)| {
                        let priority = match priority {
                            litellm_config::Value::String(text) => text.clone(),
                            litellm_config::Value::Number(number) => number.to_string(),
                            _ => return None,
                        };
                        let share = match share {
                            litellm_config::Value::Mapping(share) => {
                                let value = share.get("value").and_then(number).unwrap_or(1.0);
                                match share.get("type").and_then(litellm_config::Value::as_str) {
                                    Some("rpm") => Share::Rpm(value),
                                    Some("tpm") => Share::Tpm(value),
                                    _ => Share::Percent(value),
                                }
                            }
                            other => Share::Percent(number(other)?),
                        };
                        Some((priority, share))
                    })
                    .collect(),
            ),
            _ => None,
        };
        let settings = litellm
            .additional_fields
            .get("priority_reservation_settings");
        let setting = |name: &str| {
            settings
                .and_then(|settings| settings.get(name))
                .and_then(number)
        };
        let defaults = Self::default();
        Some(Self {
            window: rate_limits.window,
            fail_closed: rate_limits.fail_closed,
            reservation,
            default_priority: setting("default_priority").unwrap_or(defaults.default_priority),
            saturation_threshold: setting("saturation_threshold")
                .unwrap_or(defaults.saturation_threshold),
            saturation_ttl: setting("saturation_check_cache_ttl")
                .filter(|seconds| *seconds >= 0.0)
                .map_or(defaults.saturation_ttl, Duration::from_secs_f64),
            tokens: match config
                .general_settings
                .additional_fields
                .get("token_rate_limit_type")
                .and_then(litellm_config::Value::as_str)
            {
                Some("input") => TokenCount::Input,
                Some("output") => TokenCount::Output,
                _ => TokenCount::Total,
            },
        })
    }
}

/// A model group's requests and tokens per window: its deployments' `rpm` and `tpm`, summed.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Capacity {
    pub rpm: Option<i64>,
    pub tpm: Option<i64>,
}

/// Each configured group's capacity, from `rpm`/`tpm` on the deployment, its `litellm_params`
/// or its `model_info`, in that order, as the proxy's router sums them.
pub fn capacities(config: &Config) -> HashMap<String, Capacity> {
    let mut groups: HashMap<String, Capacity> = HashMap::new();
    for model in &config.model_list {
        let configured = |field: &str| -> Option<f64> {
            let from_value = |value: &litellm_config::Value| match value {
                litellm_config::Value::Number(number) => number.as_f64(),
                litellm_config::Value::String(text) => text.trim().parse().ok(),
                _ => None,
            };
            let spelled = |value: &litellm_config::Spelled<f64>| match value {
                litellm_config::Spelled::Value(value) => Some(*value),
                litellm_config::Spelled::Text(text) => text.trim().parse().ok(),
            };
            let params = match field {
                "tpm" => model.litellm_params.tpm.as_ref(),
                _ => model.litellm_params.rpm.as_ref(),
            };
            model
                .additional_fields
                .get(field)
                .and_then(from_value)
                .or_else(|| params.and_then(spelled))
                .or_else(|| model.model_info.get(field).and_then(from_value))
        };
        let group = groups.entry(model.model_name.clone()).or_default();
        let add = |total: &mut Option<i64>, value: Option<f64>| {
            if let Some(value) = value {
                *total = Some(total.unwrap_or(0) + value as i64);
            }
        };
        add(&mut group.tpm, configured("tpm"));
        add(&mut group.rpm, configured("rpm"));
    }
    groups
}

/// The pool a request draws on and its share of the group.
#[derive(Clone, Debug, PartialEq)]
struct Pool {
    name: String,
    share: f64,
}

/// The `model_capacity` hook.
pub struct ModelCapacity {
    settings: CapacitySettings,
    groups: HashMap<String, Capacity>,
    store: Arc<Store>,
    saturation: Mutex<HashMap<String, (Option<i64>, Instant)>>,
}

impl ModelCapacity {
    pub fn new(
        settings: CapacitySettings,
        groups: HashMap<String, Capacity>,
        redis: Option<Arc<Coordination>>,
    ) -> Self {
        Self {
            store: Arc::new(Store::new(redis, settings.window, settings.fail_closed)),
            settings,
            groups,
            saturation: Mutex::default(),
        }
    }

    /// The hook for a gateway whose config lists it, else `None`.
    pub fn for_gateway(config: &Config, redis: Option<Arc<Coordination>>) -> Option<Self> {
        Some(Self::new(
            CapacitySettings::from_config(config)?,
            capacities(config),
            redis,
        ))
    }

    /// The pool the proxy's `_get_priority_allocation` picks, with its normalized share.
    fn pool(&self, group: &str, capacity: &Capacity, priority: Option<&str>) -> Pool {
        let explicit = self.settings.reservation.as_ref().and_then(|reservation| {
            let priority = priority?;
            reservation
                .iter()
                .any(|(name, _)| name == priority)
                .then(|| {
                    let fractions: Vec<(&str, f64)> = reservation
                        .iter()
                        .map(|(name, share)| (name.as_str(), share.fraction(capacity)))
                        .collect();
                    let total: f64 = fractions.iter().map(|(_, fraction)| fraction).sum();
                    let fraction = fractions
                        .iter()
                        .find(|(name, _)| *name == priority)
                        .map_or(0.0, |(_, fraction)| *fraction);
                    (
                        priority,
                        if total > 1.0 {
                            fraction / total
                        } else {
                            fraction
                        },
                    )
                })
        });
        match explicit {
            Some((priority, share)) => Pool {
                name: format!("{group}:{priority}"),
                share,
            },
            None => Pool {
                name: format!("{group}:default_pool"),
                share: self.settings.default_priority,
            },
        }
    }

    /// A counter's value, reused for `saturation_check_cache_ttl`.
    async fn cached(&self, prefix: &str, kind: Kind) -> Option<i64> {
        let key = format!("{prefix}:{}", kind.name());
        if let Some((value, read_at)) = self
            .saturation
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&key)
            && read_at.elapsed() < self.settings.saturation_ttl
        {
            return *value;
        }
        let value = self.store.read(prefix, kind).await;
        let mut cache = self.saturation.lock().unwrap_or_else(|e| e.into_inner());
        if cache.len() >= 10_000 {
            cache.clear();
        }
        cache.insert(key, (value, Instant::now()));
        value
    }

    /// How full the group is: its busiest of requests and tokens against capacity.
    async fn saturation(&self, prefix: &str, capacity: &Capacity) -> f64 {
        let mut saturation = 0.0_f64;
        for (kind, limit) in [(Kind::Requests, capacity.rpm), (Kind::Tokens, capacity.tpm)] {
            if let Some(limit) = limit.filter(|limit| *limit > 0)
                && let Some(count) = self.cached(prefix, kind).await
            {
                saturation = saturation.max(count as f64 / limit as f64);
            }
        }
        saturation
    }

    fn unavailable(&self, request: &RequestFacts) -> Refusal {
        crate::rate_limits::unavailable(request.route)
    }
}

/// `team_metadata.priority`, else the key's `metadata.priority`.
fn priority(key: Option<&KeyFacts>) -> Option<String> {
    let key = key?;
    let read = |value: &Value| match value {
        Value::String(text) => Some(text.clone()),
        Value::Number(number) => Some(number.to_string()),
        _ => None,
    };
    key.team_metadata
        .get("priority")
        .and_then(read)
        .or_else(|| key.metadata.get("priority").and_then(read))
}

fn limit_text(limit: Option<i64>) -> String {
    limit.map_or_else(|| "not configured".to_owned(), |limit| limit.to_string())
}

/// `{:.<places>%}` as Python formats a fraction.
fn percent(fraction: f64, places: usize) -> String {
    format!("{:.*}%", places, fraction * 100.0)
}

impl Hook for ModelCapacity {
    fn name(&self) -> &'static str {
        "model_capacity"
    }

    fn reads(&self) -> Facts {
        Facts::KEY | Facts::USAGE
    }

    fn writes(&self) -> Targets {
        Targets::ADMISSION | Targets::RESPONSE_HEADERS
    }

    fn run<'a>(
        &'a self,
        request: &'a Arc<RequestFacts>,
        edit: &'a mut RequestEdit,
    ) -> BoxFuture<'a, Result<Option<Box<dyn Settle>>, Refusal>> {
        Box::pin(async move {
            let priority = priority(request.key.as_deref());
            let priority_header = priority.as_deref().unwrap_or("default");
            if !request.stream {
                if priority_header.chars().all(|char| (char as u32) < 256)
                    && let Ok(value) = HeaderValue::from_str(priority_header)
                {
                    edit.add_response_header(HeaderName::from_static("x-litellm-priority"), value);
                }
                edit.add_response_header(
                    HeaderName::from_static("x-litellm-rate-limiter-version"),
                    HeaderValue::from_static("v3"),
                );
            }
            let group = &*request.requested_model;
            let Some(capacity) = self.groups.get(group).copied() else {
                return Ok(None);
            };
            if capacity.rpm.is_none() && capacity.tpm.is_none() {
                return Ok(None);
            }
            let model_prefix = format!("{{model_saturation_check:{group}}}");
            let saturation = self.saturation(&model_prefix, &capacity).await;
            let pool = self.pool(group, &capacity, priority.as_deref());
            let pool_prefix = format!("{{priority_model:{}}}", pool.name);

            // The group counts every request against its whole capacity; a zero limit is off.
            let mut pairs = Vec::new();
            let push = |pairs: &mut Vec<(&'static str, Pair)>,
                        key: &'static str,
                        prefix: &str,
                        limits: [Option<i64>; 2]| {
                for (kind, limit, increment) in
                    [(Kind::Requests, limits[0], 1), (Kind::Tokens, limits[1], 0)]
                {
                    if let Some(limit) = limit {
                        pairs.push((
                            key,
                            Pair {
                                prefix: prefix.to_owned(),
                                kind,
                                limit,
                                increment,
                            },
                        ));
                    }
                }
            };
            push(
                &mut pairs,
                "model_saturation_check",
                &model_prefix,
                [
                    capacity.rpm.filter(|rpm| *rpm != 0),
                    capacity.tpm.filter(|tpm| *tpm != 0),
                ],
            );
            let share = |limit: Option<i64>| limit.map(|limit| (limit as f64 * pool.share) as i64);
            let pool_limits = [share(capacity.rpm), share(capacity.tpm)];
            let reserving =
                self.settings.reservation.is_some() && pool_limits.iter().any(Option::is_some);
            let enforced = reserving && saturation >= self.settings.saturation_threshold;
            if enforced {
                push(&mut pairs, "priority_model", &pool_prefix, pool_limits);
            }
            let checks: Vec<Pair> = pairs.iter().map(|(_, pair)| pair.clone()).collect();
            let (reserved, shared) = self
                .store
                .check_and_add(&checks)
                .await
                .map_err(|Unavailable| self.unavailable(request))?;
            let counts = match reserved {
                Reserved::Counts(counts) => counts,
                Reserved::Over { index, current } => {
                    let (key, pair) = &pairs[index];
                    let remaining = pair.limit - current;
                    let kind = pair.kind.name();
                    let priority_text = priority.as_deref().unwrap_or("None");
                    let (message, saturation_header) = match *key {
                        "model_saturation_check" => (
                            format!(
                                "Model capacity reached for {group}. Priority: {priority_text}, Rate limit type: {kind}, Model TPM: {}, Model RPM: {}, Remaining: {remaining}",
                                limit_text(capacity.tpm),
                                limit_text(capacity.rpm),
                            ),
                            None,
                        ),
                        _ => (
                            format!(
                                "Priority-based rate limit exceeded. Model: {group}, Priority: {priority_text}, Rate limit type: {kind}, Model TPM: {}, Model RPM: {}, Remaining: {remaining}, Model saturation: {}",
                                limit_text(capacity.tpm),
                                limit_text(capacity.rpm),
                                percent(saturation, 1),
                            ),
                            Some(percent(saturation, 2)),
                        ),
                    };
                    let mut headers = vec![
                        ("retry-after", self.settings.window.to_string()),
                        ("rate_limit_type", kind.to_owned()),
                        ("x-litellm-priority", priority_header.to_owned()),
                    ];
                    if let Some(saturation) = saturation_header {
                        headers.push(("x-litellm-saturation", saturation));
                    }
                    return Err(crate::rate_limits::throttled(
                        request.route,
                        message,
                        headers,
                    ));
                }
            };
            let mut statuses: Vec<(&str, Kind, i64, i64)> = pairs
                .iter()
                .zip(&counts)
                .map(|((key, pair), count)| (*key, pair.kind, pair.limit, pair.limit - count))
                .collect();
            // Below the threshold the pool's requests are only counted, so its share is known
            // once it is enforced.
            if reserving
                && !enforced
                && let Some(rpm) = pool_limits[0]
                && let Ok(counts) = self
                    .store
                    .requests(std::slice::from_ref(&pool_prefix))
                    .await
            {
                statuses.push(("priority_model", Kind::Requests, rpm, rpm - counts[0]));
            }
            if !request.stream {
                for (key, kind, limit, remaining) in statuses {
                    let kind = kind.name();
                    for (name, value) in [
                        (format!("x-ratelimit-{key}-remaining-{kind}"), remaining),
                        (format!("x-ratelimit-{key}-limit-{kind}"), limit),
                    ] {
                        if let Ok(name) = HeaderName::from_bytes(name.as_bytes()) {
                            edit.add_response_header(name, HeaderValue::from(value));
                        }
                    }
                }
            }
            Ok(Some(Box::new(Charge {
                store: self.store.clone(),
                prefixes: [model_prefix, pool_prefix],
                tokens: self.settings.tokens,
                shared,
            }) as Box<dyn Settle>))
        })
    }
}

/// Adds a successful response's tokens to the group and its pool.
struct Charge {
    store: Arc<Store>,
    prefixes: [String; 2],
    tokens: TokenCount,
    shared: bool,
}

impl Settle for Charge {
    fn settle(self: Box<Self>, record: &Record) {
        if !(200..300).contains(&record.outcome.status) {
            return;
        }
        let Some(usage) = record.outcome.usage else {
            return;
        };
        let tokens = match self.tokens {
            TokenCount::Total => usage.total_tokens(),
            TokenCount::Input => usage.prompt_tokens(),
            TokenCount::Output => usage.output_tokens,
        };
        let Ok(tokens) = i64::try_from(tokens) else {
            return;
        };
        if tokens == 0 {
            return;
        }
        let Charge {
            store,
            prefixes,
            shared,
            ..
        } = *self;
        tokio::spawn(async move {
            let deltas = prefixes
                .into_iter()
                .map(|prefix| (prefix, tokens))
                .collect();
            store.charge(deltas, shared).await;
        });
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn capacity_hook(reservation: Option<Vec<(String, Share)>>) -> ModelCapacity {
        ModelCapacity::new(
            CapacitySettings {
                reservation,
                ..CapacitySettings::default()
            },
            HashMap::new(),
            None,
        )
    }

    #[rstest]
    fn pools_follow_the_proxy_allocation() {
        let capacity = Capacity {
            rpm: Some(100),
            tpm: Some(1_000),
        };
        let hook = capacity_hook(Some(vec![
            ("prod".into(), Share::Percent(0.6)),
            ("dev".into(), Share::Rpm(80.0)),
        ]));
        // 0.6 + 0.8 is over-allocated, so both shrink to their proportion.
        let prod = hook.pool("m", &capacity, Some("prod"));
        assert_eq!(prod.name, "m:prod");
        assert!((prod.share - 0.6 / 1.4).abs() < 1e-12);
        assert_eq!(
            hook.pool("m", &capacity, Some("other")),
            Pool {
                name: "m:default_pool".into(),
                share: 0.25
            }
        );
        assert_eq!(
            capacity_hook(None).pool("m", &capacity, Some("prod")).name,
            "m:default_pool"
        );
    }

    #[rstest]
    #[case(0.625, 1, "62.5%")]
    #[case(0.5, 2, "50.00%")]
    fn formats_percentages_like_python(
        #[case] fraction: f64,
        #[case] places: usize,
        #[case] expected: &str,
    ) {
        assert_eq!(percent(fraction, places), expected);
    }

    #[rstest]
    fn capacities_sum_deployments() {
        let config = Config::from_yaml(
            "model_list:
  - model_name: a
    litellm_params: {model: openai/x, rpm: 10, tpm: 1000}
  - model_name: a
    rpm: 5
    litellm_params: {model: openai/y, rpm: 99}
    model_info: {tpm: 500}
  - model_name: b
    litellm_params: {model: openai/z}
",
        )
        .unwrap();
        let groups = capacities(&config);
        assert_eq!(
            groups["a"],
            Capacity {
                rpm: Some(15),
                tpm: Some(1_500)
            }
        );
        assert_eq!(groups["b"], Capacity::default());
    }
}
