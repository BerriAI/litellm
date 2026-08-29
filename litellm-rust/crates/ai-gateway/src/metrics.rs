use prometheus::{
    CounterVec, Encoder, HistogramOpts, HistogramVec, IntCounterVec, IntGaugeVec, Opts, Registry,
    TextEncoder,
};

/// Gateway metrics exported via Prometheus text format.
#[derive(Clone)]
pub struct GatewayMetrics {
    pub registry: Registry,
    pub requests_total: IntCounterVec,
    pub request_duration_seconds: HistogramVec,
    pub active_requests: IntGaugeVec,
    pub tokens_total: IntCounterVec,
    pub spend_usd_total: CounterVec,
    pub circuit_breaker_state: IntGaugeVec,
    pub rate_limit_rejections: IntCounterVec,
    pub retry_attempts: IntCounterVec,
}

impl GatewayMetrics {
    pub fn new() -> Self {
        let registry = Registry::new();

        let requests_total = IntCounterVec::new(
            Opts::new("litellm_requests_total", "Total chat completion requests"),
            &["model", "status"],
        )
        .expect("counter");

        let request_duration_seconds = HistogramVec::new(
            HistogramOpts::new(
                "litellm_request_duration_seconds",
                "Request latency in seconds",
            )
            .buckets(vec![0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]),
            &["model"],
        )
        .expect("histogram");

        let active_requests = IntGaugeVec::new(
            Opts::new("litellm_active_requests", "Currently in-flight requests"),
            &["model"],
        )
        .expect("gauge");

        let tokens_total = IntCounterVec::new(
            Opts::new("litellm_tokens_total", "Total tokens processed"),
            &["model", "kind"],
        )
        .expect("counter");

        let spend_usd_total = CounterVec::new(
            Opts::new("litellm_spend_usd_total", "Total spend in USD"),
            &["model"],
        )
        .expect("counter");

        let circuit_breaker_state = IntGaugeVec::new(
            Opts::new(
                "litellm_circuit_breaker_state",
                "Circuit breaker state (0=closed, 1=open, 2=half_open)",
            ),
            &["provider"],
        )
        .expect("gauge");

        let rate_limit_rejections = IntCounterVec::new(
            Opts::new(
                "litellm_rate_limit_rejections_total",
                "Requests rejected by rate limiting",
            ),
            &["reason"],
        )
        .expect("counter");

        let retry_attempts = IntCounterVec::new(
            Opts::new("litellm_retry_attempts_total", "Total retry attempts"),
            &["model"],
        )
        .expect("counter");

        registry
            .register(Box::new(requests_total.clone()))
            .expect("register requests_total");
        registry
            .register(Box::new(request_duration_seconds.clone()))
            .expect("register request_duration");
        registry
            .register(Box::new(active_requests.clone()))
            .expect("register active_requests");
        registry
            .register(Box::new(tokens_total.clone()))
            .expect("register tokens_total");
        registry
            .register(Box::new(spend_usd_total.clone()))
            .expect("register spend_usd_total");
        registry
            .register(Box::new(circuit_breaker_state.clone()))
            .expect("register circuit_breaker_state");
        registry
            .register(Box::new(rate_limit_rejections.clone()))
            .expect("register rate_limit_rejections");
        registry
            .register(Box::new(retry_attempts.clone()))
            .expect("register retry_attempts");

        Self {
            registry,
            requests_total,
            request_duration_seconds,
            active_requests,
            tokens_total,
            spend_usd_total,
            circuit_breaker_state,
            rate_limit_rejections,
            retry_attempts,
        }
    }

    pub fn render(&self) -> String {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder
            .encode(&metric_families, &mut buffer)
            .expect("encode metrics");
        String::from_utf8(buffer).expect("metrics are utf8")
    }
}

impl Default for GatewayMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_creation() {
        let metrics = GatewayMetrics::new();
        metrics
            .requests_total
            .with_label_values(&["gpt-4", "success"])
            .inc();
        metrics
            .tokens_total
            .with_label_values(&["gpt-4", "prompt"])
            .inc_by(100);

        let output = metrics.render();
        assert!(output.contains("litellm_requests_total"));
        assert!(output.contains("litellm_tokens_total"));
        assert!(output.contains("gpt-4"));
    }

    #[test]
    fn test_metrics_render_empty() {
        let metrics = GatewayMetrics::new();
        let output = metrics.render();
        assert!(output.is_empty() || output.contains("litellm"));
    }

    #[test]
    fn test_spend_counter() {
        let metrics = GatewayMetrics::new();
        metrics
            .spend_usd_total
            .with_label_values(&["gpt-4"])
            .inc_by(0.05);
        metrics
            .spend_usd_total
            .with_label_values(&["gpt-4"])
            .inc_by(0.03);

        let output = metrics.render();
        assert!(output.contains("litellm_spend_usd_total"));
    }
}
