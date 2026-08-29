use std::sync::Arc;
use std::time::Duration;

use reqwest::Client;

use crate::auth::circuit_breaker::CircuitBreakerRegistry;
use crate::hardening::GlobalRateLimiter;
use crate::io::realtime_pool::RealtimePool;
use crate::metrics::GatewayMetrics;
use litellm_core::auth::KeyCache;
use litellm_core::persistence::{PostgresStore, RedisStore};
use litellm_core::router::Router;
use litellm_core::spend_tracking::SpendWorker;

use crate::integrations::custom_logger::CustomLogger;

/// Configuration values read once from the environment at startup.
/// Avoids per-request `std::env::var()` allocations on the hot path.
#[derive(Clone, Debug)]
pub struct GatewayConfig {
    pub default_request_timeout_secs: f64,
    pub cache_ttl_secs: u64,
    pub team_budget: f64,
    pub org_budget: f64,
}

impl GatewayConfig {
    pub fn from_env() -> Self {
        Self {
            default_request_timeout_secs: std::env::var("LITELLM_REQUEST_TIMEOUT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(600.0),
            cache_ttl_secs: std::env::var("LITELLM_CACHE_TTL")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(300),
            team_budget: std::env::var("LITELLM_TEAM_BUDGET")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(f64::MAX),
            org_budget: std::env::var("LITELLM_ORG_BUDGET")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(f64::MAX),
        }
    }
}

/// Shared application state handed to every route handler.
#[derive(Clone)]
pub struct AppState {
    pub router: Arc<Router>,
    /// The gateway master key. Any caller presenting it as a bearer token may
    /// invoke the gateway. `None` → auth not configured (routes fail closed).
    pub master_key: Option<Arc<str>>,
    /// Logging callbacks fanned out at the end of each realtime session.
    pub loggers: Arc<Vec<Arc<dyn CustomLogger>>>,
    /// Pre-warmed upstream realtime connection pool. Disabled
    /// (`RealtimePool::disabled()`) when `REALTIME_POOL_SIZE=0`, in which case
    /// every realtime connect fresh-dials exactly as before.
    pub realtime_pool: Arc<RealtimePool>,
    /// In-memory cache for API key auth objects. Keyed by SHA-256 hashed token.
    pub key_cache: Arc<KeyCache>,
    /// Redis connection for spend counters and rate limiting. `None` if
    /// `REDIS_URL` is not set.
    pub redis: Option<Arc<RedisStore>>,
    /// PostgreSQL connection for spend logs and key lookups. `None` if
    /// `DATABASE_URL` is not set.
    pub postgres: Option<Arc<PostgresStore>>,
    /// Background spend tracking worker. Batches spend entries and flushes
    /// them to Redis + Postgres. `None` if neither Redis nor Postgres is
    /// configured (spend is logged to stdout only).
    pub spend_worker: Option<Arc<SpendWorker>>,
    /// HTTP client with connection pooling for upstream provider calls.
    /// Configured with pool idle timeout and max idle connections per host.
    pub http_client: Arc<Client>,
    /// Circuit breaker registry for upstream providers. Prevents cascading
    /// failures when providers are down or slow.
    pub circuit_breakers: Arc<CircuitBreakerRegistry>,
    /// Prometheus metrics for the gateway.
    pub metrics: Arc<GatewayMetrics>,
    /// Cached configuration from environment variables.
    pub config: GatewayConfig,
    /// Global rate limiter (in addition to per-key limits).
    pub global_rate_limiter: Arc<GlobalRateLimiter>,
}

impl AppState {
    /// Create a new HTTP client with connection pooling configured.
    pub fn new_http_client() -> Client {
        Client::builder()
            .pool_idle_timeout(Duration::from_secs(90))
            .pool_max_idle_per_host(100)
            .tcp_keepalive(Duration::from_secs(60))
            .timeout(Duration::from_secs(300))
            .build()
            .expect("failed to build HTTP client")
    }
}
