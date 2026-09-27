use std::{
    collections::{BTreeSet, HashMap},
    sync::Arc,
    time::{Duration, Instant},
};

use axum::{
    Router,
    extract::{Request, State},
    http::HeaderMap,
    middleware::Next,
    response::Response,
};
use litellm_config::Config;
use litellm_cost::{PricingPlan, PromptConvention, Rate, Rates, ServiceTier, ThresholdPolicy};
use litellm_gateway_inference::{Metered, Metering};
use litellm_jobs::{Exclusivity, HolderId, JobName, JobSpec, MemoryLeaseStore, Schedule};
use litellm_spend::{
    Attribution, Buffer, Charges, Cost, Counters, EntityKey, FlushOutcome, MemoryBuffer, Outcome,
    SpendEvent, Usage, charges, flush,
};
use litellm_spend_redis::{
    BufferSettings, PythonCounterNaming, PythonTotalsCodec, RedisBuffer, RedisCounters,
};
use percent_encoding::{NON_ALPHANUMERIC, utf8_percent_encode};
use redis::aio::MultiplexedConnection;
use time::OffsetDateTime;
use tokio::task::JoinHandle;
use tokio_util::{sync::CancellationToken, task::TaskTracker};

use crate::Error;

const DEFAULT_FLUSH_PERIOD: Duration = Duration::from_secs(10);
const FLUSH_JITTER: Duration = Duration::from_secs(5);
const FLUSH_MAX_ENTRIES: usize = 1_000;
const CLAIM_TTL: Duration = Duration::from_secs(60);
const REMEMBER_APPLIED_FOR: Duration = Duration::from_secs(600);
const BLOBS_PER_CLAIM: usize = 100;

pub struct Spend {
    counters: RedisCounters<MultiplexedConnection, PythonCounterNaming>,
    local: MemoryBuffer<EntityKey, Cost>,
    shared: RedisBuffer<MultiplexedConnection, PythonTotalsCodec>,
    prices: HashMap<String, PricingPlan>,
    flush_period: Duration,
    tracker: TaskTracker,
}

pub struct SpendWorker {
    spend: Arc<Spend>,
    flusher: JoinHandle<()>,
    shutdown: CancellationToken,
}

impl Spend {
    pub async fn connect(
        config: &Config,
        redis_url: Option<String>,
    ) -> Result<Option<Arc<Self>>, Error> {
        let settings = &config.general_settings;
        if !settings.use_redis_transaction_buffer {
            return Ok(None);
        }
        let url = redis_url.ok_or(Error::NoRedis)?;
        let connection = redis::Client::open(url)?
            .get_multiplexed_async_connection()
            .await?;
        Ok(Some(Arc::new(Self {
            counters: RedisCounters::new(connection.clone(), PythonCounterNaming, None),
            local: MemoryBuffer::new(CLAIM_TTL, REMEMBER_APPLIED_FOR),
            shared: RedisBuffer::new(
                connection,
                PythonTotalsCodec,
                BufferSettings {
                    claim_ttl: CLAIM_TTL,
                    remember_applied_for: REMEMBER_APPLIED_FOR,
                    blobs_per_claim: BLOBS_PER_CLAIM,
                },
            ),
            prices: prices(config)?,
            flush_period: settings
                .proxy_batch_write_at
                .map_or(DEFAULT_FLUSH_PERIOD, Duration::from_secs),
            tracker: TaskTracker::new(),
        })))
    }

    pub fn start(self: Arc<Self>) -> SpendWorker {
        let shutdown = CancellationToken::new();
        let spend = Arc::clone(&self);
        let token = shutdown.clone();
        let flusher = tokio::spawn(async move {
            let spec = JobSpec {
                name: JobName::new("spend_local_flush"),
                schedule: Schedule::Every {
                    period: spend.flush_period,
                    jitter: FLUSH_JITTER,
                },
                exclusivity: Exclusivity::EveryPod,
            };
            let body = |_| {
                let spend = Arc::clone(&spend);
                async move { spend.drain().await }
            };
            litellm_jobs::run(
                &MemoryLeaseStore::default(),
                &HolderId::random(),
                &spec,
                body,
                token,
            )
            .await;
        });
        SpendWorker {
            spend: self,
            flusher,
            shutdown,
        }
    }

    async fn drain(&self) {
        loop {
            match flush(&self.local, &self.shared, FLUSH_MAX_ENTRIES).await {
                Ok(FlushOutcome::Committed { .. }) => {}
                Ok(FlushOutcome::Empty) => return,
                Err(error) => {
                    tracing::warn!(%error, "flushing spend totals to Redis failed; the next flush retries");
                    return;
                }
            }
        }
    }

    async fn record(&self, request: Settled) {
        let metered = request.metered;
        let Some(plan) = self.prices.get(&metered.model_group) else {
            tracing::warn!(
                model = metered.model_group,
                "model has no custom pricing; its spend is not recorded"
            );
            return;
        };
        let cost = match plan.calculate(&priced(metered.usage)) {
            Ok(cost) => cost.total(),
            Err(error) => {
                tracing::warn!(
                    model = metered.model_group,
                    ?error,
                    "pricing a request failed; its spend is not recorded"
                );
                return;
            }
        };
        let event = SpendEvent {
            cost,
            attribution: Attribution {
                tags: request.tags,
                ..Attribution::default()
            },
            started_at: request.started_at,
            model: Some(metered.model),
            model_group: Some(metered.model_group),
            custom_llm_provider: metered.custom_llm_provider,
            endpoint: Some(request.endpoint),
            mcp_namespaced_tool_name: None,
            usage: spend_usage(metered.usage),
            outcome: match metered.completed {
                true => Outcome::Success,
                false => Outcome::Failure,
            },
            response_time_ms: u64::try_from(request.elapsed.as_millis()).ok(),
        };
        let Charges {
            counters, totals, ..
        } = charges(&event);
        if let Err(error) = self.counters.add(&counters).await {
            tracing::warn!(%error, "incrementing budget counters failed");
        }
        let Ok(()) = self.local.push(totals).await;
    }
}

impl SpendWorker {
    pub async fn stop(self) {
        self.spend.tracker.close();
        self.spend.tracker.wait().await;
        self.shutdown.cancel();
        if let Err(error) = self.flusher.await {
            tracing::warn!(%error, "the spend flush job ended abnormally");
        }
        self.spend.drain().await;
    }
}

struct Settled {
    metered: Metered,
    tags: Vec<String>,
    endpoint: String,
    started_at: OffsetDateTime,
    elapsed: Duration,
}

pub fn metered(router: Router, spend: Arc<Spend>) -> Router {
    router.layer(axum::middleware::from_fn_with_state(spend, meter))
}

async fn meter(State(spend): State<Arc<Spend>>, request: Request, next: Next) -> Response {
    let started_at = OffsetDateTime::now_utc();
    let started = Instant::now();
    let tags = tags(request.headers());
    let endpoint = request.uri().path().to_owned();
    let response = next.run(request).await;
    let Some(metering) = response.extensions().get::<Metering>().cloned() else {
        return response;
    };
    let tracker = spend.tracker.clone();
    tracker.spawn(async move {
        let Some(metered) = metering.settled().await else {
            return;
        };
        spend
            .record(Settled {
                metered,
                tags,
                endpoint,
                started_at,
                elapsed: started.elapsed(),
            })
            .await;
    });
    response
}

fn tags(headers: &HeaderMap) -> Vec<String> {
    let Some(header) = headers
        .get("x-litellm-tags")
        .and_then(|value| value.to_str().ok())
    else {
        return Vec::new();
    };
    header
        .split(',')
        .map(str::trim)
        .filter(|tag| !tag.is_empty())
        .map(str::to_owned)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn prices(config: &Config) -> Result<HashMap<String, PricingPlan>, Error> {
    config
        .model_list
        .iter()
        .filter(|model| {
            let params = &model.litellm_params;
            params.input_cost_per_token.is_some() || params.output_cost_per_token.is_some()
        })
        .map(|model| {
            let params = &model.litellm_params;
            let standard = Rates {
                input: rate(params.input_cost_per_token),
                output: rate(params.output_cost_per_token),
                cache_read: rate(params.cache_read_input_token_cost),
                cache_write: rate(params.cache_creation_input_token_cost),
                cache_write_1h: Rate::Missing,
            };
            let plan = litellm_cost::compile(&litellm_cost::Pricing {
                standard,
                tiers: &[],
                thresholds: &[],
                off_peak: None,
            })
            .map_err(|error| Error::Pricing {
                model: model.model_name.clone(),
                error,
            })?;
            Ok((model.model_name.clone(), plan))
        })
        .collect()
}

fn rate(value: Option<f64>) -> Rate {
    value.map_or(Rate::Missing, Rate::Value)
}

fn priced(usage: litellm_cost::Usage) -> litellm_cost::Request {
    litellm_cost::Request {
        usage,
        service_tier: ServiceTier::Standard,
        threshold_policy: ThresholdPolicy::Exclusive,
        region_multiplier: None,
        billed_at_utc_minute: None,
    }
}

fn spend_usage(usage: litellm_cost::Usage) -> Usage {
    let cached = usage
        .cache_read_tokens
        .saturating_add(usage.cache_write_tokens);
    Usage {
        prompt_tokens: match usage.prompt_convention {
            PromptConvention::IncludesCache => usage.prompt_tokens,
            PromptConvention::ExcludesCache => usage.prompt_tokens.saturating_add(cached),
        },
        completion_tokens: usage.completion_tokens,
        cache_read_input_tokens: usage.cache_read_tokens,
        cache_creation_input_tokens: usage.cache_write_tokens,
    }
}

pub fn redis_url_from_env() -> Option<String> {
    let set = |name: &str| std::env::var(name).ok().filter(|value| !value.is_empty());
    if let Some(url) = set("REDIS_URL") {
        return Some(url);
    }
    let host = set("REDIS_HOST")?;
    let port = set("REDIS_PORT").unwrap_or_else(|| "6379".to_owned());
    Some(match set("REDIS_PASSWORD") {
        Some(password) => format!(
            "redis://:{}@{host}:{port}",
            utf8_percent_encode(&password, NON_ALPHANUMERIC)
        ),
        None => format!("redis://{host}:{port}"),
    })
}
