//! The `prometheus` callback: the proxy's request metrics, under its metric and label names,
//! served on the gateway's `/metrics`.
//!
//! It is an observer, so nothing here runs on the request path. Each finished request updates a
//! few series per metric: the label values are hashed without allocating, and a series is created
//! once, the first time its label values are seen. The text is built only when Prometheus scrapes.

use std::{
    collections::{HashMap, HashSet},
    fmt::Write as _,
    hash::{BuildHasher, Hash, Hasher, RandomState},
    sync::{
        RwLock,
        atomic::{AtomicU64, Ordering},
    },
    time::SystemTime,
};

use futures_util::future::BoxFuture;
use litellm_config::Config;
use litellm_gateway_hooks::{Facts, Observer, Record, ServiceTier};
use serde::Deserialize;

use crate::{Error, metrics::escape, spend::logged_model};

/// The names a config's callback lists turn this observer on with.
const NAMES: [&str; 1] = ["prometheus"];

/// The proxy's default latency buckets, in seconds; `+Inf` is implied.
const LATENCY_BUCKETS: [f64; 17] = [
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 420.0,
    600.0,
];

/// The Prometheus client's default buckets, which the proxy keeps for latency per output token.
const DEFAULT_BUCKETS: [f64; 14] = [
    0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0,
];

/// A label the proxy puts on its request metrics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::EnumString, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
enum Label {
    EndUser,
    User,
    UserEmail,
    HashedApiKey,
    ApiKeyAlias,
    Team,
    TeamAlias,
    RequestedModel,
    ModelGroup,
    Model,
    LitellmModelName,
    ModelId,
    ApiBase,
    ApiProvider,
    StatusCode,
    ExceptionStatus,
    ExceptionClass,
    Route,
    ClientIp,
    UserAgent,
    ServiceTier,
}

use Label::*;

const LLM_API_LATENCY: &[Label] = &[
    ModelGroup,
    Model,
    HashedApiKey,
    ApiKeyAlias,
    Team,
    TeamAlias,
    RequestedModel,
    EndUser,
    User,
    ModelId,
    ApiProvider,
    ServiceTier,
];
const REQUEST_TOTAL_LATENCY: &[Label] = &[
    ModelGroup,
    EndUser,
    HashedApiKey,
    ApiKeyAlias,
    RequestedModel,
    Team,
    TeamAlias,
    User,
    Model,
    ModelId,
    ApiProvider,
    ServiceTier,
];
const PROXY_TOTAL_REQUESTS: &[Label] = &[
    EndUser,
    HashedApiKey,
    ApiKeyAlias,
    RequestedModel,
    Team,
    TeamAlias,
    User,
    StatusCode,
    UserEmail,
    Route,
    ClientIp,
    UserAgent,
    ModelId,
    ApiProvider,
];
const PROXY_FAILED_REQUESTS: &[Label] = &[
    EndUser,
    HashedApiKey,
    ApiKeyAlias,
    RequestedModel,
    Team,
    TeamAlias,
    User,
    UserEmail,
    ExceptionStatus,
    ExceptionClass,
    Route,
    ClientIp,
    UserAgent,
    ModelId,
    ApiProvider,
];
const LATENCY_PER_OUTPUT_TOKEN: &[Label] = &[
    ModelGroup,
    LitellmModelName,
    ModelId,
    ApiBase,
    ApiProvider,
    HashedApiKey,
    ApiKeyAlias,
    Team,
    TeamAlias,
];
const REQUESTS: &[Label] = &[
    EndUser,
    HashedApiKey,
    ApiKeyAlias,
    Model,
    Team,
    TeamAlias,
    User,
    UserEmail,
    ClientIp,
    UserAgent,
    RequestedModel,
    ModelId,
    ApiProvider,
];
const SPEND: &[Label] = &[
    EndUser,
    HashedApiKey,
    ApiKeyAlias,
    Model,
    Team,
    TeamAlias,
    User,
    UserEmail,
    ClientIp,
    UserAgent,
    RequestedModel,
    ModelId,
    ApiProvider,
    ServiceTier,
];
const TOKENS: &[Label] = &[
    EndUser,
    HashedApiKey,
    ApiKeyAlias,
    Model,
    Team,
    TeamAlias,
    User,
    UserEmail,
    RequestedModel,
    ModelId,
    ApiProvider,
];
const DEPLOYMENT_STATE: &[Label] = &[LitellmModelName, ModelId, ApiBase, ApiProvider];
const DEPLOYMENT_REQUESTS: &[Label] = &[
    ModelGroup,
    RequestedModel,
    LitellmModelName,
    ModelId,
    ApiBase,
    ApiProvider,
    HashedApiKey,
    ApiKeyAlias,
    Team,
    TeamAlias,
    ClientIp,
    UserAgent,
];
const DEPLOYMENT_FAILURES: &[Label] = &[
    ModelGroup,
    RequestedModel,
    LitellmModelName,
    ModelId,
    ApiBase,
    ApiProvider,
    ExceptionStatus,
    ExceptionClass,
    HashedApiKey,
    ApiKeyAlias,
    Team,
    TeamAlias,
    ClientIp,
    UserAgent,
];
const LLM_API_FAILED_REQUESTS: &[Label] = &[
    EndUser,
    HashedApiKey,
    ApiKeyAlias,
    Model,
    Team,
    TeamAlias,
    User,
    ModelId,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Counter,
    Gauge,
    /// Latency histograms, whose buckets `prometheus_latency_buckets` can replace.
    Latency,
    /// A histogram on the Prometheus client's default buckets.
    Histogram,
}

/// The metrics this observer emits, in the order `/metrics` lists them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Metric {
    ProxyTotalRequests,
    ProxyFailedRequests,
    RequestTotalLatency,
    LlmApiLatency,
    LlmApiTimeToFirstToken,
    Requests,
    Spend,
    TotalTokens,
    InputTokens,
    OutputTokens,
    InputCachedTokens,
    InputCacheCreationTokens,
    LlmApiFailedRequests,
    DeploymentState,
    DeploymentSuccessResponses,
    DeploymentFailureResponses,
    DeploymentTotalRequests,
    DeploymentLatencyPerOutputToken,
}

impl Metric {
    const ALL: [Self; 18] = [
        Self::ProxyTotalRequests,
        Self::ProxyFailedRequests,
        Self::RequestTotalLatency,
        Self::LlmApiLatency,
        Self::LlmApiTimeToFirstToken,
        Self::Requests,
        Self::Spend,
        Self::TotalTokens,
        Self::InputTokens,
        Self::OutputTokens,
        Self::InputCachedTokens,
        Self::InputCacheCreationTokens,
        Self::LlmApiFailedRequests,
        Self::DeploymentState,
        Self::DeploymentSuccessResponses,
        Self::DeploymentFailureResponses,
        Self::DeploymentTotalRequests,
        Self::DeploymentLatencyPerOutputToken,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::ProxyTotalRequests => "litellm_proxy_total_requests_metric",
            Self::ProxyFailedRequests => "litellm_proxy_failed_requests_metric",
            Self::RequestTotalLatency => "litellm_request_total_latency_metric",
            Self::LlmApiLatency => "litellm_llm_api_latency_metric",
            Self::LlmApiTimeToFirstToken => "litellm_llm_api_time_to_first_token_metric",
            Self::Requests => "litellm_requests_metric",
            Self::Spend => "litellm_spend_metric",
            Self::TotalTokens => "litellm_total_tokens_metric",
            Self::InputTokens => "litellm_input_tokens_metric",
            Self::OutputTokens => "litellm_output_tokens_metric",
            Self::InputCachedTokens => "litellm_input_cached_tokens_metric",
            Self::InputCacheCreationTokens => "litellm_input_cache_creation_tokens_metric",
            Self::LlmApiFailedRequests => "litellm_llm_api_failed_requests_metric",
            Self::DeploymentState => "litellm_deployment_state",
            Self::DeploymentSuccessResponses => "litellm_deployment_success_responses",
            Self::DeploymentFailureResponses => "litellm_deployment_failure_responses",
            Self::DeploymentTotalRequests => "litellm_deployment_total_requests",
            Self::DeploymentLatencyPerOutputToken => "litellm_deployment_latency_per_output_token",
        }
    }

    fn help(self) -> &'static str {
        match self {
            Self::ProxyTotalRequests => {
                "Total number of requests made to the proxy server - track number of client side requests"
            }
            Self::ProxyFailedRequests => {
                "Total number of failed responses from proxy - the client did not get a success response from litellm proxy"
            }
            Self::RequestTotalLatency => "Total latency (seconds) for a request to LiteLLM",
            Self::LlmApiLatency => "Total latency (seconds) for a models LLM API call",
            Self::LlmApiTimeToFirstToken => "Time to first token for a models LLM API call",
            Self::Requests => {
                "deprecated - use litellm_proxy_total_requests_metric. Total number of LLM calls to litellm - track total per API Key, team, user"
            }
            Self::Spend => "Total spend on LLM requests",
            Self::TotalTokens => "Total number of input + output tokens from LLM requests",
            Self::InputTokens => "Total number of input tokens from LLM requests",
            Self::OutputTokens => "Total number of output tokens from LLM requests",
            Self::InputCachedTokens => {
                "Total number of cached input tokens read from the provider cache"
            }
            Self::InputCacheCreationTokens => {
                "Total number of input tokens written to the provider cache"
            }
            Self::LlmApiFailedRequests => "deprecated - use litellm_proxy_failed_requests_metric",
            Self::DeploymentState => {
                "LLM Deployment Analytics - The state of the deployment: 0 = healthy, 1 = partial outage, 2 = complete outage"
            }
            Self::DeploymentSuccessResponses => {
                "LLM Deployment Analytics - Total number of successful LLM API calls via litellm"
            }
            Self::DeploymentFailureResponses => {
                "LLM Deployment Analytics - Total number of failed LLM API calls for a specific LLM deploymeny. exception_status is the status of the exception from the llm api"
            }
            Self::DeploymentTotalRequests => {
                "LLM Deployment Analytics - Total number of LLM API calls via litellm - success + failure"
            }
            Self::DeploymentLatencyPerOutputToken => {
                "LLM Deployment Analytics - Latency per output token"
            }
        }
    }

    fn kind(self) -> Kind {
        match self {
            Self::RequestTotalLatency | Self::LlmApiLatency | Self::LlmApiTimeToFirstToken => {
                Kind::Latency
            }
            Self::DeploymentLatencyPerOutputToken => Kind::Histogram,
            Self::DeploymentState => Kind::Gauge,
            _ => Kind::Counter,
        }
    }

    fn labels(self) -> &'static [Label] {
        match self {
            Self::ProxyTotalRequests => PROXY_TOTAL_REQUESTS,
            Self::ProxyFailedRequests => PROXY_FAILED_REQUESTS,
            Self::RequestTotalLatency => REQUEST_TOTAL_LATENCY,
            Self::LlmApiLatency | Self::LlmApiTimeToFirstToken => LLM_API_LATENCY,
            Self::Requests => REQUESTS,
            Self::Spend => SPEND,
            Self::TotalTokens
            | Self::InputTokens
            | Self::OutputTokens
            | Self::InputCachedTokens
            | Self::InputCacheCreationTokens => TOKENS,
            Self::LlmApiFailedRequests => LLM_API_FAILED_REQUESTS,
            Self::DeploymentState => DEPLOYMENT_STATE,
            Self::DeploymentSuccessResponses | Self::DeploymentTotalRequests => DEPLOYMENT_REQUESTS,
            Self::DeploymentFailureResponses => DEPLOYMENT_FAILURES,
            Self::DeploymentLatencyPerOutputToken => LATENCY_PER_OUTPUT_TOKEN,
        }
    }

    fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|metric| metric.name() == name)
    }
}

#[derive(Deserialize)]
struct GroupConfig {
    #[serde(default)]
    metrics: Vec<String>,
    #[serde(default)]
    include_labels: Option<Vec<String>>,
}

/// The series that share a hash of their label values, each with those values.
type Bucket = Vec<(Box<[Option<Box<str>>]>, Box<Cell>)>;

/// One metric's label names and series.
struct Family {
    metric: Metric,
    labels: Box<[Label]>,
    buckets: Box<[f64]>,
    series: RwLock<HashMap<u64, Bucket>>,
}

/// A counter or gauge value, or a histogram.
enum Cell {
    Value(AtomicF64),
    Histogram(Histogram),
}

#[derive(Default)]
struct AtomicF64(AtomicU64);

impl AtomicF64 {
    fn add(&self, amount: f64) {
        let _ = self
            .0
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |bits| {
                Some((f64::from_bits(bits) + amount).to_bits())
            });
    }

    fn set(&self, value: f64) {
        self.0.store(value.to_bits(), Ordering::Relaxed);
    }

    fn get(&self) -> f64 {
        f64::from_bits(self.0.load(Ordering::Relaxed))
    }
}

struct Histogram {
    /// Per-bucket counts, not cumulative; the last slot is `+Inf`.
    buckets: Box<[AtomicU64]>,
    sum: AtomicF64,
    count: AtomicU64,
}

impl Histogram {
    fn new(bounds: usize) -> Self {
        Self {
            buckets: (0..=bounds).map(|_| AtomicU64::new(0)).collect(),
            sum: AtomicF64::default(),
            count: AtomicU64::new(0),
        }
    }

    fn observe(&self, bounds: &[f64], value: f64) {
        let index = bounds
            .iter()
            .position(|bound| value <= *bound)
            .unwrap_or(bounds.len());
        self.buckets[index].fetch_add(1, Ordering::Relaxed);
        self.sum.add(value);
        self.count.fetch_add(1, Ordering::Relaxed);
    }
}

impl Family {
    fn new(metric: Metric, labels: Vec<Label>, latency_buckets: &[f64]) -> Self {
        let buckets: Box<[f64]> = match metric.kind() {
            Kind::Latency => latency_buckets.into(),
            Kind::Histogram => DEFAULT_BUCKETS.into(),
            Kind::Counter | Kind::Gauge => Box::default(),
        };
        Self {
            metric,
            labels: labels.into(),
            buckets,
            series: RwLock::default(),
        }
    }

    fn with<R>(&self, hasher: &RandomState, values: &Values<'_>, apply: impl Fn(&Cell) -> R) -> R {
        let mut state = hasher.build_hasher();
        for label in &self.labels {
            values.get(*label).hash(&mut state);
        }
        let hash = state.finish();
        let matches = |stored: &[Option<Box<str>>]| {
            stored
                .iter()
                .zip(self.labels.iter())
                .all(|(stored, label)| stored.as_deref() == values.get(*label))
        };
        {
            let series = self.series.read().unwrap_or_else(|e| e.into_inner());
            if let Some((_, cell)) = series
                .get(&hash)
                .and_then(|bucket| bucket.iter().find(|(stored, _)| matches(stored)))
            {
                return apply(cell);
            }
        }
        let mut series = self.series.write().unwrap_or_else(|e| e.into_inner());
        let bucket = series.entry(hash).or_default();
        let index = match bucket.iter().position(|(stored, _)| matches(stored)) {
            Some(index) => index,
            None => {
                let stored = self
                    .labels
                    .iter()
                    .map(|label| values.get(*label).map(Box::from))
                    .collect();
                let cell = match self.metric.kind() {
                    Kind::Counter | Kind::Gauge => Cell::Value(AtomicF64::default()),
                    Kind::Latency | Kind::Histogram => {
                        Cell::Histogram(Histogram::new(self.buckets.len()))
                    }
                };
                bucket.push((stored, Box::new(cell)));
                bucket.len() - 1
            }
        };
        apply(&bucket[index].1)
    }

    fn render(&self, out: &mut String) {
        let name = self.metric.name();
        let kind = match self.metric.kind() {
            Kind::Counter => "counter",
            Kind::Gauge => "gauge",
            Kind::Latency | Kind::Histogram => "histogram",
        };
        let _ = writeln!(out, "# HELP {name} {}", self.metric.help());
        let _ = writeln!(out, "# TYPE {name} {kind}");
        let series = self.series.read().unwrap_or_else(|e| e.into_inner());
        let mut rows: Vec<(String, &Cell)> = series
            .values()
            .flatten()
            .map(|(values, cell)| {
                let labels = self
                    .labels
                    .iter()
                    .zip(values.iter())
                    .map(|(label, value)| {
                        let label: &'static str = label.into();
                        format!("{label}=\"{}\"", escape(value.as_deref().unwrap_or("None")))
                    })
                    .collect::<Vec<_>>()
                    .join(",");
                (labels, &**cell)
            })
            .collect();
        rows.sort_by(|left, right| left.0.cmp(&right.0));
        for (labels, cell) in rows {
            match cell {
                Cell::Value(value) => {
                    let suffix = match self.metric.kind() {
                        Kind::Counter => "_total",
                        _ => "",
                    };
                    let _ = writeln!(out, "{name}{suffix}{{{labels}}} {}", number(value.get()));
                }
                Cell::Histogram(histogram) => {
                    let separator = if labels.is_empty() { "" } else { "," };
                    let mut cumulative = 0;
                    for (index, count) in histogram.buckets.iter().enumerate() {
                        cumulative += count.load(Ordering::Relaxed);
                        let bound = self
                            .buckets
                            .get(index)
                            .map_or_else(|| "+Inf".to_owned(), |bound| number(*bound));
                        let _ = writeln!(
                            out,
                            "{name}_bucket{{{labels}{separator}le=\"{bound}\"}} {}",
                            number(cumulative as f64)
                        );
                    }
                    let _ = writeln!(
                        out,
                        "{name}_count{{{labels}}} {}",
                        number(histogram.count.load(Ordering::Relaxed) as f64)
                    );
                    let _ = writeln!(
                        out,
                        "{name}_sum{{{labels}}} {}",
                        number(histogram.sum.get())
                    );
                }
            }
        }
    }
}

/// A number the way the Prometheus client writes it: whole numbers keep a `.0`, so bucket
/// bounds read `le="1.0"` as dashboards built on the proxy expect.
fn number(value: f64) -> String {
    if value.is_infinite() {
        return if value > 0.0 { "+Inf" } else { "-Inf" }.to_owned();
    }
    if value.fract() == 0.0 && value.abs() < 1e15 {
        format!("{value:.1}")
    } else {
        format!("{value}")
    }
}

/// One request's label values.
struct Values<'a> {
    record: &'a Record,
    end_user: Option<&'a str>,
    model: Option<String>,
    status_code: String,
    exception_class: Option<String>,
    user_agent: Option<&'a str>,
    /// Pre-routing failures carry no deployment labels and the client's model.
    deployment_selected: bool,
}

impl<'a> Values<'a> {
    fn new(record: &'a Record, end_user_label: bool) -> Self {
        let request = &record.request;
        let deployment = record.outcome.deployment.as_ref();
        let status = record.outcome.status;
        let success = (200..300).contains(&status);
        Self {
            record,
            end_user: request.end_user.as_deref().filter(|_| end_user_label),
            model: deployment.map(|_| logged_model(record)),
            status_code: status.to_string(),
            exception_class: (!success).then(|| exception_class(record)),
            user_agent: request
                .client_headers
                .get(http::header::USER_AGENT)
                .and_then(|value| value.to_str().ok()),
            deployment_selected: deployment.is_some(),
        }
    }

    fn get(&self, label: Label) -> Option<&str> {
        let record = self.record;
        let key = record.key();
        let deployment = record.outcome.deployment.as_ref();
        match label {
            EndUser => self.end_user,
            User => key.and_then(|key| key.user_id.as_deref()),
            UserEmail | ClientIp => None,
            HashedApiKey => key.map(|key| key.hash.as_str()),
            ApiKeyAlias => key.and_then(|key| key.alias.as_deref()),
            Team => key.and_then(|key| key.team_id.as_deref()),
            TeamAlias => key.and_then(|key| key.team_alias.as_deref()),
            RequestedModel | ModelGroup => Some(
                deployment
                    .map(|deployment| &*deployment.model_group)
                    .unwrap_or(&record.request.requested_model),
            ),
            Model | LitellmModelName => match self.deployment_selected {
                true => self.model.as_deref(),
                false => Some(""),
            },
            ModelId => match deployment {
                Some(deployment) => Some(&deployment.id),
                None => Some(""),
            },
            ApiBase => match deployment {
                Some(deployment) => deployment.api_base.as_deref(),
                None => Some(""),
            },
            ApiProvider => match deployment {
                Some(deployment) => deployment.provider(),
                None => Some(""),
            },
            StatusCode | ExceptionStatus => Some(&self.status_code),
            ExceptionClass => self.exception_class.as_deref(),
            Route => Some(record.request.route.endpoint()),
            UserAgent => self.user_agent,
            Label::ServiceTier => record.outcome.usage.map(|usage| match usage.service_tier {
                ServiceTier::Standard => "standard",
                ServiceTier::Priority => "priority",
                ServiceTier::Flex => "flex",
                ServiceTier::Batch => "batch",
            }),
        }
    }
}

/// The proxy's `exception_class` label. The gateway's own refusals keep the classes the proxy
/// reports for them; a provider error is named the way the proxy maps its status, prefixed by
/// the provider.
fn exception_class(record: &Record) -> String {
    let status = record.outcome.status;
    let Some(deployment) = &record.outcome.deployment else {
        return match status {
            400 => "BudgetExceededError",
            429 | 503 => "HTTPException",
            _ => "ProxyException",
        }
        .to_owned();
    };
    let class = match status {
        400 => "BadRequestError",
        401 => "AuthenticationError",
        403 => "PermissionDeniedError",
        404 => "NotFoundError",
        408 => "Timeout",
        422 => "UnprocessableEntityError",
        429 => "RateLimitError",
        500 => "InternalServerError",
        503 => "ServiceUnavailableError",
        _ => "APIError",
    };
    match deployment.provider() {
        Some(provider) if !provider.is_empty() => {
            let mut letters = provider.chars();
            let first = letters.next().map(|c| c.to_uppercase().collect::<String>());
            format!("{}{}.{class}", first.unwrap_or_default(), letters.as_str())
        }
        _ => class.to_owned(),
    }
}

fn seconds(from: SystemTime, to: SystemTime) -> Option<f64> {
    to.duration_since(from)
        .ok()
        .map(|elapsed| elapsed.as_secs_f64())
}

/// The `prometheus` observer.
pub struct Prometheus {
    families: Vec<Option<Family>>,
    hasher: RandomState,
    end_user_label: bool,
    reads_headers: bool,
}

impl Prometheus {
    /// `Ok(None)` when no callback list names `prometheus`. A metric or label the proxy would
    /// reject stops the gateway, the way it stops the proxy.
    pub fn from_config(config: &Config) -> Result<Option<Self>, Error> {
        let litellm = &config.litellm_settings;
        let listed = [
            &litellm.callbacks,
            &litellm.success_callback,
            &litellm.failure_callback,
        ]
        .into_iter()
        .flatten()
        .flatten()
        .any(|callback| {
            callback
                .as_str()
                .is_some_and(|name| NAMES.contains(&name.trim()))
        });
        if !listed {
            return Ok(None);
        }
        let error = |message: String| Error::Hooks(format!("prometheus: {message}"));
        let setting = |name: &str| litellm.additional_fields.get(name).cloned();
        let parse = |name: &str| -> Result<Option<Vec<String>>, Error> {
            setting(name)
                .filter(|value| !value.is_null())
                .map(Vec::<String>::deserialize)
                .transpose()
                .map_err(|e| error(format!("{name}: {e}")))
        };
        let groups: Vec<GroupConfig> = setting("prometheus_metrics_config")
            .filter(|value| !value.is_null())
            .map(Vec::<GroupConfig>::deserialize)
            .transpose()
            .map_err(|e| error(format!("prometheus_metrics_config: {e}")))?
            .unwrap_or_default();
        let excluded_metrics: HashSet<String> = parse("prometheus_exclude_metrics")?
            .unwrap_or_default()
            .into_iter()
            .collect();
        let excluded_labels: HashSet<String> = parse("prometheus_exclude_labels")?
            .unwrap_or_default()
            .into_iter()
            .collect();
        for label in &excluded_labels {
            if label.parse::<Label>().is_err() {
                tracing::warn!(%label, "prometheus_exclude_labels names a label this gateway does not emit");
            }
        }
        let latency_buckets: Vec<f64> = match setting("prometheus_latency_buckets") {
            Some(value) if !value.is_null() => Vec::<f64>::deserialize(value)
                .map_err(|e| error(format!("prometheus_latency_buckets: {e}")))?
                .into_iter()
                .filter(|bound| bound.is_finite())
                .collect(),
            _ => LATENCY_BUCKETS.to_vec(),
        };
        let mut enabled = HashSet::new();
        let mut include: HashMap<&'static str, Vec<Label>> = HashMap::new();
        for group in &groups {
            for name in &group.metrics {
                let Some(metric) = Metric::from_name(name) else {
                    tracing::warn!(metric = %name, "prometheus_metrics_config names a metric this gateway does not emit");
                    continue;
                };
                enabled.insert(metric.name());
                let Some(labels) = &group.include_labels else {
                    continue;
                };
                let mut resolved = Vec::new();
                for label in labels {
                    match label
                        .parse::<Label>()
                        .ok()
                        .filter(|label| metric.labels().contains(label))
                    {
                        Some(label) => resolved.push(label),
                        None => {
                            return Err(error(format!(
                                "label {label} is not a label of {}",
                                metric.name()
                            )));
                        }
                    }
                }
                include.insert(metric.name(), resolved);
            }
        }
        let families: Vec<Option<Family>> = Metric::ALL
            .into_iter()
            .map(|metric| {
                let name = metric.name();
                if excluded_metrics.contains(name)
                    || (!enabled.is_empty() && !enabled.contains(name))
                {
                    return None;
                }
                let labels = metric
                    .labels()
                    .iter()
                    .copied()
                    .filter(|label| {
                        let label_name: &'static str = label.into();
                        !excluded_labels.contains(label_name)
                            && include
                                .get(name)
                                .is_none_or(|included| included.contains(label))
                    })
                    .collect();
                Some(Family::new(metric, labels, &latency_buckets))
            })
            .collect();
        let reads_headers = families
            .iter()
            .flatten()
            .any(|family| family.labels.contains(&UserAgent));
        let end_user_label = !matches!(
            litellm
                .additional_fields
                .get("disable_end_user_cost_tracking_prometheus_only"),
            Some(litellm_config::Value::Bool(true))
        );
        Ok(Some(Self {
            families,
            hasher: RandomState::new(),
            end_user_label,
            reads_headers,
        }))
    }

    fn family(&self, metric: Metric) -> Option<&Family> {
        self.families[metric as usize].as_ref()
    }

    fn add(&self, metric: Metric, values: &Values<'_>, amount: f64) {
        if let Some(family) = self.family(metric) {
            family.with(&self.hasher, values, |cell| {
                if let Cell::Value(value) = cell {
                    value.add(amount);
                }
            });
        }
    }

    fn set(&self, metric: Metric, values: &Values<'_>, value: f64) {
        if let Some(family) = self.family(metric) {
            family.with(&self.hasher, values, |cell| {
                if let Cell::Value(cell) = cell {
                    cell.set(value);
                }
            });
        }
    }

    fn observe_value(&self, metric: Metric, values: &Values<'_>, value: Option<f64>) {
        let (Some(family), Some(value)) = (self.family(metric), value) else {
            return;
        };
        family.with(&self.hasher, values, |cell| {
            if let Cell::Histogram(histogram) = cell {
                histogram.observe(&family.buckets, value);
            }
        });
    }

    fn record(&self, record: &Record) {
        let values = Values::new(record, self.end_user_label);
        let outcome = &record.outcome;
        let request = &record.request;
        let first_byte = outcome
            .time_to_first_byte
            .map(|first_byte| request.started_at + first_byte);
        if !(200..300).contains(&outcome.status) {
            self.add(Metric::ProxyFailedRequests, &values, 1.0);
            self.add(Metric::ProxyTotalRequests, &values, 1.0);
            if values.deployment_selected {
                self.add(Metric::LlmApiFailedRequests, &values, 1.0);
                self.set(Metric::DeploymentState, &values, 1.0);
                self.add(Metric::DeploymentFailureResponses, &values, 1.0);
                self.add(Metric::DeploymentTotalRequests, &values, 1.0);
            }
            return;
        }
        let usage = outcome.usage.unwrap_or_default();
        self.add(Metric::Requests, &values, 1.0);
        self.add(Metric::Spend, &values, outcome.cost.unwrap_or(0.0));
        self.add(Metric::TotalTokens, &values, usage.total_tokens() as f64);
        self.add(Metric::InputTokens, &values, usage.prompt_tokens() as f64);
        self.add(Metric::OutputTokens, &values, usage.output_tokens as f64);
        if usage.cache_read_tokens > 0 {
            self.add(
                Metric::InputCachedTokens,
                &values,
                usage.cache_read_tokens as f64,
            );
        }
        if usage.cache_write_tokens > 0 {
            self.add(
                Metric::InputCacheCreationTokens,
                &values,
                usage.cache_write_tokens as f64,
            );
        }
        let api_started = outcome.upstream_started_at;
        if request.stream {
            self.observe_value(
                Metric::LlmApiTimeToFirstToken,
                &values,
                api_started
                    .zip(first_byte)
                    .and_then(|(from, to)| seconds(from, to)),
            );
        }
        self.observe_value(
            Metric::LlmApiLatency,
            &values,
            api_started.and_then(|from| seconds(from, outcome.ended_at)),
        );
        self.observe_value(
            Metric::RequestTotalLatency,
            &values,
            seconds(request.started_at, outcome.ended_at),
        );
        if values.deployment_selected {
            self.set(Metric::DeploymentState, &values, 0.0);
            self.add(Metric::DeploymentSuccessResponses, &values, 1.0);
            self.add(Metric::DeploymentTotalRequests, &values, 1.0);
            // The proxy divides time to first token for streams, total time otherwise.
            let from = api_started.unwrap_or(request.started_at);
            let to = match request.stream {
                true => first_byte.unwrap_or(outcome.ended_at),
                false => outcome.ended_at,
            };
            if usage.output_tokens > 0 {
                self.observe_value(
                    Metric::DeploymentLatencyPerOutputToken,
                    &values,
                    seconds(from, to).map(|elapsed| elapsed / usage.output_tokens as f64),
                );
            }
        }
        self.add(Metric::ProxyTotalRequests, &values, 1.0);
    }

    /// The metrics in the Prometheus text format.
    pub fn render(&self, out: &mut String) {
        for family in self.families.iter().flatten() {
            family.render(out);
        }
    }
}

impl Observer for Prometheus {
    fn name(&self) -> &'static str {
        "prometheus"
    }

    fn reads(&self) -> Facts {
        let facts = Facts::KEY
            | Facts::END_USER
            | Facts::DEPLOYMENT
            | Facts::USAGE
            | Facts::COST
            | Facts::TIMINGS;
        match self.reads_headers {
            true => facts | Facts::CLIENT_HEADERS,
            false => facts,
        }
    }

    fn observe<'a>(&'a self, record: &'a Record) -> BoxFuture<'a, ()> {
        self.record(record);
        Box::pin(async {})
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use axum::body::Bytes;
    use http::HeaderMap;
    use litellm_gateway_hooks::{
        Annotations, DeploymentFacts, KeyFacts, Outcome, RequestFacts, Route, Usage,
    };
    use rstest::rstest;

    use super::*;

    fn duration(seconds: f64) -> Duration {
        Duration::from_secs_f64(seconds)
    }

    fn config(settings: &str) -> Config {
        Config::from_yaml(&format!(
            "model_list: []\nlitellm_settings:\n  callbacks: [prometheus]\n{settings}"
        ))
        .unwrap()
    }

    fn record(status: u16, stream: bool, deployment: bool) -> Record {
        let started_at = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000);
        let mut headers = HeaderMap::new();
        headers.insert("user-agent", "client/1.0".parse().unwrap());
        let key = KeyFacts {
            hash: "hash-1".into(),
            alias: Some("alias-1".into()),
            user_id: Some("user-1".into()),
            team_id: Some("team-1".into()),
            team_alias: Some("research".into()),
            ..KeyFacts::default()
        };
        let mut request = RequestFacts::new(
            "r".into(),
            Route::Messages,
            Some(Arc::new(key)),
            Some("end-1".into()),
            "claude".into(),
            Arc::new(headers),
            stream,
            None,
            Bytes::new(),
        );
        request.started_at = started_at;
        Record {
            request: Arc::new(request),
            outcome: Outcome {
                deployment: deployment.then(|| DeploymentFacts {
                    id: "dep-1".into(),
                    model_group: "claude".into(),
                    model: "anthropic/claude-x".into(),
                    provider: None,
                    api_base: Some("https://api.example".into()),
                }),
                status,
                usage: Some(Usage {
                    input_tokens: 10,
                    output_tokens: 20,
                    cache_read_tokens: 5,
                    ..Usage::default()
                }),
                cost: Some(0.25),
                time_to_first_byte: Some(duration(0.5)),
                upstream_started_at: Some(started_at + duration(0.1)),
                ended_at: started_at + duration(2.1),
                interrupted: false,
            },
            annotations: Annotations::default(),
            response_body: None,
        }
    }

    fn rendered(prometheus: &Prometheus) -> String {
        let mut out = String::new();
        prometheus.render(&mut out);
        out
    }

    #[test]
    fn off_unless_a_callback_list_names_it() {
        let config =
            Config::from_yaml("model_list: []\nlitellm_settings:\n  success_callback: [s3_v2]\n")
                .unwrap();
        assert!(Prometheus::from_config(&config).unwrap().is_none());
        let config = Config::from_yaml(
            "model_list: []\nlitellm_settings:\n  success_callback: [prometheus]\n",
        )
        .unwrap();
        assert!(Prometheus::from_config(&config).unwrap().is_some());
    }

    #[test]
    fn a_success_counts_spend_tokens_and_latency_under_the_proxy_labels() {
        let prometheus = Prometheus::from_config(&config("")).unwrap().unwrap();
        prometheus.record(&record(200, true, true));
        prometheus.record(&record(200, true, true));
        let out = rendered(&prometheus);
        let labels = r#"end_user="end-1",hashed_api_key="hash-1",api_key_alias="alias-1",model="claude-x",team="team-1",team_alias="research",user="user-1",user_email="None",client_ip="None",user_agent="client/1.0",requested_model="claude",model_id="dep-1",api_provider="anthropic",service_tier="standard""#;
        assert!(
            out.contains(&format!("litellm_spend_metric_total{{{labels}}} 0.5\n")),
            "{out}"
        );
        assert!(out.contains(r#"litellm_input_tokens_metric_total{end_user="end-1",hashed_api_key="hash-1",api_key_alias="alias-1",model="claude-x",team="team-1",team_alias="research",user="user-1",user_email="None",requested_model="claude",model_id="dep-1",api_provider="anthropic"} 30.0"#), "{out}");
        assert!(out.contains("litellm_output_tokens_metric_total{"), "{out}");
        assert!(out.contains(r#"litellm_input_cached_tokens_metric_total{end_user="end-1",hashed_api_key="hash-1",api_key_alias="alias-1",model="claude-x",team="team-1",team_alias="research",user="user-1",user_email="None",requested_model="claude",model_id="dep-1",api_provider="anthropic"} 10.0"#), "{out}");
        assert!(!out.contains("litellm_input_cache_creation_tokens_metric_total{"));
        assert!(out.contains(r#"litellm_proxy_total_requests_metric_total{end_user="end-1",hashed_api_key="hash-1",api_key_alias="alias-1",requested_model="claude",team="team-1",team_alias="research",user="user-1",status_code="200",user_email="None",route="/v1/messages",client_ip="None",user_agent="client/1.0",model_id="dep-1",api_provider="anthropic"} 2.0"#), "{out}");
        // Time to first token is 0.4 s after the upstream call, the call took 2.0 s and the
        // request 2.1 s.
        let latency = r#"model_group="claude",model="claude-x",hashed_api_key="hash-1",api_key_alias="alias-1",team="team-1",team_alias="research",requested_model="claude",end_user="end-1",user="user-1",model_id="dep-1",api_provider="anthropic",service_tier="standard""#;
        assert!(
            out.contains(&format!(
                "litellm_llm_api_time_to_first_token_metric_bucket{{{latency},le=\"0.25\"}} 0.0\n"
            )),
            "{out}"
        );
        assert!(
            out.contains(&format!(
                "litellm_llm_api_time_to_first_token_metric_bucket{{{latency},le=\"0.5\"}} 2.0\n"
            )),
            "{out}"
        );
        assert!(
            out.contains(&format!(
                "litellm_llm_api_latency_metric_bucket{{{latency},le=\"2.0\"}} 2.0\n"
            )),
            "{out}"
        );
        assert!(
            out.contains(&format!(
                "litellm_llm_api_latency_metric_bucket{{{latency},le=\"1.0\"}} 0.0\n"
            )),
            "{out}"
        );
        assert!(out.contains(
            "litellm_request_total_latency_metric_bucket{model_group=\"claude\",end_user=\"end-1\""
        ));
        assert!(out.contains(r#"litellm_deployment_state{litellm_model_name="claude-x",model_id="dep-1",api_base="https://api.example",api_provider="anthropic"} 0.0"#), "{out}");
        assert!(out.contains(r#"litellm_deployment_success_responses_total{model_group="claude",requested_model="claude",litellm_model_name="claude-x",model_id="dep-1",api_base="https://api.example",api_provider="anthropic",hashed_api_key="hash-1",api_key_alias="alias-1",team="team-1",team_alias="research",client_ip="None",user_agent="client/1.0"} 2.0"#), "{out}");
        // 0.4 s to the first token over 20 output tokens.
        assert!(out.contains(r#"litellm_deployment_latency_per_output_token_sum{model_group="claude",litellm_model_name="claude-x",model_id="dep-1",api_base="https://api.example",api_provider="anthropic",hashed_api_key="hash-1",api_key_alias="alias-1",team="team-1",team_alias="research"} 0.04"#), "{out}");
        assert!(out.contains("# TYPE litellm_spend_metric counter\n"));
        assert!(out.contains("# TYPE litellm_deployment_state gauge\n"));
        assert!(!out.contains("litellm_proxy_failed_requests_metric_total{"));
    }

    #[test]
    fn non_streams_have_no_time_to_first_token() {
        let prometheus = Prometheus::from_config(&config("")).unwrap().unwrap();
        prometheus.record(&record(200, false, true));
        let out = rendered(&prometheus);
        assert!(!out.contains("litellm_llm_api_time_to_first_token_metric_count"));
        assert!(out.contains("litellm_llm_api_latency_metric_count{"));
    }

    #[test]
    fn a_provider_failure_marks_the_deployment() {
        let prometheus = Prometheus::from_config(&config("")).unwrap().unwrap();
        prometheus.record(&record(200, false, true));
        prometheus.record(&record(429, false, true));
        let out = rendered(&prometheus);
        assert!(out.contains(r#"litellm_deployment_state{litellm_model_name="claude-x",model_id="dep-1",api_base="https://api.example",api_provider="anthropic"} 1.0"#), "{out}");
        assert!(out.contains(r#"litellm_deployment_failure_responses_total{model_group="claude",requested_model="claude",litellm_model_name="claude-x",model_id="dep-1",api_base="https://api.example",api_provider="anthropic",exception_status="429",exception_class="Anthropic.RateLimitError",hashed_api_key="hash-1""#), "{out}");
        assert!(out.contains(r#"litellm_deployment_total_requests_total{model_group="claude",requested_model="claude",litellm_model_name="claude-x",model_id="dep-1",api_base="https://api.example",api_provider="anthropic",hashed_api_key="hash-1",api_key_alias="alias-1",team="team-1",team_alias="research",client_ip="None",user_agent="client/1.0"} 2.0"#), "{out}");
        assert!(out.contains(r#"litellm_llm_api_failed_requests_metric_total{end_user="end-1",hashed_api_key="hash-1",api_key_alias="alias-1",model="claude-x",team="team-1",team_alias="research",user="user-1",model_id="dep-1"} 1.0"#), "{out}");
        assert!(out.contains(r#"status_code="429""#));
        // A failure's tokens and spend are not counted.
        assert!(out.contains("litellm_requests_metric_total{") && out.contains("} 1.0\n"));
    }

    #[test]
    fn a_refusal_counts_as_a_proxy_failure_only() {
        let prometheus = Prometheus::from_config(&config("")).unwrap().unwrap();
        prometheus.record(&record(429, false, false));
        let out = rendered(&prometheus);
        assert!(out.contains(r#"litellm_proxy_failed_requests_metric_total{end_user="end-1",hashed_api_key="hash-1",api_key_alias="alias-1",requested_model="claude",team="team-1",team_alias="research",user="user-1",user_email="None",exception_status="429",exception_class="HTTPException",route="/v1/messages",client_ip="None",user_agent="client/1.0",model_id="",api_provider=""} 1.0"#), "{out}");
        assert!(!out.contains("litellm_deployment_total_requests_total{"));
        assert!(!out.contains("litellm_llm_api_failed_requests_metric_total{"));
    }

    #[test]
    fn the_config_picks_metrics_and_labels() {
        let prometheus = Prometheus::from_config(&config(
            "  prometheus_metrics_config:
    - group: spend
      metrics: [litellm_spend_metric, litellm_remaining_team_budget_metric]
      include_labels: [team, model]
    - metrics: [litellm_proxy_total_requests_metric]
  prometheus_exclude_labels: [user_agent]
  disable_end_user_cost_tracking_prometheus_only: true
",
        ))
        .unwrap()
        .unwrap();
        assert!(!prometheus.reads().contains(Facts::CLIENT_HEADERS));
        prometheus.record(&record(200, false, true));
        let out = rendered(&prometheus);
        assert!(
            out.contains("litellm_spend_metric_total{model=\"claude-x\",team=\"team-1\"} 0.25\n"),
            "{out}"
        );
        assert!(out.contains("litellm_proxy_total_requests_metric_total{end_user=\"None\","));
        assert!(!out.contains("user_agent="));
        assert!(!out.contains("litellm_input_tokens_metric"));
    }

    #[rstest]
    #[case::bad_label(
        "  prometheus_metrics_config:\n    - metrics: [litellm_spend_metric]\n      include_labels: [route]\n",
        "label route is not a label of litellm_spend_metric"
    )]
    #[case::bad_buckets("  prometheus_latency_buckets: fast\n", "prometheus_latency_buckets")]
    fn a_bad_config_stops_the_gateway(#[case] settings: &str, #[case] message: &str) {
        let error = Prometheus::from_config(&config(settings))
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains(message), "{error}");
    }

    #[test]
    fn custom_latency_buckets_replace_the_defaults() {
        let prometheus =
            Prometheus::from_config(&config("  prometheus_latency_buckets: [1, 10, .inf]\n"))
                .unwrap()
                .unwrap();
        prometheus.record(&record(200, false, true));
        let out = rendered(&prometheus);
        assert!(out.contains(",le=\"10.0\"} 1.0\n"), "{out}");
        assert!(!out.contains("litellm_llm_api_latency_metric_bucket{model_group=\"claude\",model=\"claude-x\",hashed_api_key=\"hash-1\",api_key_alias=\"alias-1\",team=\"team-1\",team_alias=\"research\",requested_model=\"claude\",end_user=\"end-1\",user=\"user-1\",model_id=\"dep-1\",api_provider=\"anthropic\",service_tier=\"standard\",le=\"0.5\"}"));
    }

    #[rstest]
    #[case(1.0, "1.0")]
    #[case(0.25, "0.25")]
    #[case(f64::INFINITY, "+Inf")]
    #[case(30.0, "30.0")]
    fn numbers_are_written_like_the_prometheus_client(#[case] value: f64, #[case] text: &str) {
        assert_eq!(number(value), text);
    }
}
