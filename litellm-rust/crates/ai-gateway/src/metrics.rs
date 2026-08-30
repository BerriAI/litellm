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
    // Enhanced metrics
    pub deployment_requests_total: IntCounterVec,
    pub deployment_duration_seconds: HistogramVec,
    pub cache_hits_total: IntCounterVec,
    pub cache_misses_total: IntCounterVec,
    pub errors_total: IntCounterVec,
    pub routing_strategy_selections: IntCounterVec,
    pub health_status: IntGaugeVec,
    pub load_tracker_load: IntGaugeVec,
    pub latency_tracker_avg_ms: HistogramVec,
    // Middleware metrics
    pub validation_failures_total: IntCounterVec,
    pub csrf_rejections_total: IntCounterVec,
    pub cors_preflight_requests_total: IntCounterVec,
    pub alerts_triggered_total: IntCounterVec,
    pub trace_spans_created_total: IntCounterVec,
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
            .buckets(vec![
                0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
            ]),
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

        // Enhanced metrics
        let deployment_requests_total = IntCounterVec::new(
            Opts::new(
                "litellm_deployment_requests_total",
                "Total requests per deployment",
            ),
            &["model_name", "provider", "status"],
        )
        .expect("counter");

        let deployment_duration_seconds = HistogramVec::new(
            HistogramOpts::new(
                "litellm_deployment_duration_seconds",
                "Request latency per deployment in seconds",
            )
            .buckets(vec![
                0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
            ]),
            &["model_name", "provider"],
        )
        .expect("histogram");

        let cache_hits_total = IntCounterVec::new(
            Opts::new("litellm_cache_hits_total", "Total cache hits"),
            &["model", "cache_type"],
        )
        .expect("counter");

        let cache_misses_total = IntCounterVec::new(
            Opts::new("litellm_cache_misses_total", "Total cache misses"),
            &["model", "cache_type"],
        )
        .expect("counter");

        let errors_total = IntCounterVec::new(
            Opts::new("litellm_errors_total", "Total errors by type"),
            &["model", "error_type"],
        )
        .expect("counter");

        let routing_strategy_selections = IntCounterVec::new(
            Opts::new(
                "litellm_routing_strategy_selections_total",
                "Total routing strategy selections",
            ),
            &["strategy", "model"],
        )
        .expect("counter");

        let health_status = IntGaugeVec::new(
            Opts::new(
                "litellm_health_status",
                "Deployment health status (0=unknown, 1=healthy, 2=degraded, 3=unhealthy)",
            ),
            &["model_name", "provider"],
        )
        .expect("gauge");

        let load_tracker_load = IntGaugeVec::new(
            Opts::new("litellm_load_tracker_load", "Current load per deployment"),
            &["model_name", "provider"],
        )
        .expect("gauge");

        let latency_tracker_avg_ms = HistogramVec::new(
            HistogramOpts::new(
                "litellm_latency_tracker_avg_ms",
                "Average latency per deployment in milliseconds",
            )
            .buckets(vec![
                10.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0,
            ]),
            &["model_name", "provider"],
        )
        .expect("histogram");

        // Middleware metrics
        let validation_failures_total = IntCounterVec::new(
            Opts::new(
                "litellm_validation_failures_total",
                "Total validation failures by type",
            ),
            &["validation_type"],
        )
        .expect("counter");

        let csrf_rejections_total = IntCounterVec::new(
            Opts::new(
                "litellm_csrf_rejections_total",
                "Total CSRF token rejections",
            ),
            &["reason"],
        )
        .expect("counter");

        let cors_preflight_requests_total = IntCounterVec::new(
            Opts::new(
                "litellm_cors_preflight_requests_total",
                "Total CORS preflight requests",
            ),
            &["origin", "method"],
        )
        .expect("counter");

        let alerts_triggered_total = IntCounterVec::new(
            Opts::new(
                "litellm_alerts_triggered_total",
                "Total alerts triggered by type",
            ),
            &["alert_type", "severity"],
        )
        .expect("counter");

        let trace_spans_created_total = IntCounterVec::new(
            Opts::new(
                "litellm_trace_spans_created_total",
                "Total trace spans created",
            ),
            &["service"],
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
        registry
            .register(Box::new(deployment_requests_total.clone()))
            .expect("register deployment_requests_total");
        registry
            .register(Box::new(deployment_duration_seconds.clone()))
            .expect("register deployment_duration_seconds");
        registry
            .register(Box::new(cache_hits_total.clone()))
            .expect("register cache_hits_total");
        registry
            .register(Box::new(cache_misses_total.clone()))
            .expect("register cache_misses_total");
        registry
            .register(Box::new(errors_total.clone()))
            .expect("register errors_total");
        registry
            .register(Box::new(routing_strategy_selections.clone()))
            .expect("register routing_strategy_selections");
        registry
            .register(Box::new(health_status.clone()))
            .expect("register health_status");
        registry
            .register(Box::new(load_tracker_load.clone()))
            .expect("register load_tracker_load");
        registry
            .register(Box::new(latency_tracker_avg_ms.clone()))
            .expect("register latency_tracker_avg_ms");
        registry
            .register(Box::new(validation_failures_total.clone()))
            .expect("register validation_failures_total");
        registry
            .register(Box::new(csrf_rejections_total.clone()))
            .expect("register csrf_rejections_total");
        registry
            .register(Box::new(cors_preflight_requests_total.clone()))
            .expect("register cors_preflight_requests_total");
        registry
            .register(Box::new(alerts_triggered_total.clone()))
            .expect("register alerts_triggered_total");
        registry
            .register(Box::new(trace_spans_created_total.clone()))
            .expect("register trace_spans_created_total");

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
            deployment_requests_total,
            deployment_duration_seconds,
            cache_hits_total,
            cache_misses_total,
            errors_total,
            routing_strategy_selections,
            health_status,
            load_tracker_load,
            latency_tracker_avg_ms,
            validation_failures_total,
            csrf_rejections_total,
            cors_preflight_requests_total,
            alerts_triggered_total,
            trace_spans_created_total,
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

    #[test]
    fn test_middleware_metrics() {
        let metrics = GatewayMetrics::new();

        metrics
            .validation_failures_total
            .with_label_values(&["model_name"])
            .inc();
        metrics
            .csrf_rejections_total
            .with_label_values(&["invalid_token"])
            .inc();
        metrics
            .cors_preflight_requests_total
            .with_label_values(&["https://example.com", "POST"])
            .inc();
        metrics
            .alerts_triggered_total
            .with_label_values(&["high_error_rate", "critical"])
            .inc();
        metrics
            .trace_spans_created_total
            .with_label_values(&["ai-gateway"])
            .inc();

        let output = metrics.render();
        assert!(output.contains("litellm_validation_failures_total"));
        assert!(output.contains("litellm_csrf_rejections_total"));
        assert!(output.contains("litellm_cors_preflight_requests_total"));
        assert!(output.contains("litellm_alerts_triggered_total"));
        assert!(output.contains("litellm_trace_spans_created_total"));
    }
}
