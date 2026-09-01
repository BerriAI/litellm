//! End-to-end tests for the AI gateway.
//!
//! Tests the full request pipeline including middleware integration.

#[cfg(test)]
#[allow(clippy::module_inception)]
mod e2e_tests {
    use crate::metrics::GatewayMetrics;

    #[test]
    fn test_metrics_comprehensive_tracking() {
        let metrics = GatewayMetrics::new();

        // Simulate various request scenarios
        metrics
            .requests_total
            .with_label_values(&["gpt-4", "success"])
            .inc_by(10);
        metrics
            .requests_total
            .with_label_values(&["gpt-4", "error"])
            .inc_by(2);
        metrics
            .requests_total
            .with_label_values(&["claude-3", "success"])
            .inc_by(5);

        metrics
            .tokens_total
            .with_label_values(&["gpt-4", "prompt"])
            .inc_by(1000);
        metrics
            .tokens_total
            .with_label_values(&["gpt-4", "completion"])
            .inc_by(500);

        metrics
            .spend_usd_total
            .with_label_values(&["gpt-4"])
            .inc_by(0.50);
        metrics
            .spend_usd_total
            .with_label_values(&["claude-3"])
            .inc_by(0.25);

        metrics
            .deployment_requests_total
            .with_label_values(&["gpt-4", "openai", "success"])
            .inc_by(10);
        metrics
            .deployment_duration_seconds
            .with_label_values(&["gpt-4", "openai"])
            .observe(0.5);

        metrics
            .cache_hits_total
            .with_label_values(&["gpt-4", "semantic"])
            .inc_by(5);
        metrics
            .cache_misses_total
            .with_label_values(&["gpt-4", "semantic"])
            .inc_by(3);

        metrics
            .errors_total
            .with_label_values(&["gpt-4", "timeout"])
            .inc();
        metrics
            .errors_total
            .with_label_values(&["gpt-4", "rate_limit"])
            .inc_by(2);

        let output = metrics.render();

        assert!(output.contains("litellm_requests_total"));
        assert!(output.contains("litellm_tokens_total"));
        assert!(output.contains("litellm_spend_usd_total"));
        assert!(output.contains("litellm_deployment_requests_total"));
        assert!(output.contains("litellm_cache_hits_total"));
        assert!(output.contains("litellm_errors_total"));
        assert!(output.contains("gpt-4"));
        assert!(output.contains("claude-3"));
    }

    #[test]
    fn test_middleware_metrics_integration() {
        let metrics = GatewayMetrics::new();

        // Simulate middleware-specific metrics
        metrics
            .validation_failures_total
            .with_label_values(&["payload_too_large"])
            .inc_by(3);
        metrics
            .validation_failures_total
            .with_label_values(&["invalid_model"])
            .inc_by(2);

        metrics
            .csrf_rejections_total
            .with_label_values(&["missing_token"])
            .inc_by(5);
        metrics
            .csrf_rejections_total
            .with_label_values(&["invalid_token"])
            .inc_by(1);

        metrics
            .cors_preflight_requests_total
            .with_label_values(&["https://example.com", "POST"])
            .inc_by(10);
        metrics
            .cors_preflight_requests_total
            .with_label_values(&["https://app.com", "GET"])
            .inc_by(5);

        metrics
            .alerts_triggered_total
            .with_label_values(&["high_error_rate", "critical"])
            .inc();
        metrics
            .alerts_triggered_total
            .with_label_values(&["high_latency", "warning"])
            .inc_by(2);

        metrics
            .trace_spans_created_total
            .with_label_values(&["ai-gateway"])
            .inc_by(100);

        let output = metrics.render();

        assert!(output.contains("litellm_validation_failures_total"));
        assert!(output.contains("litellm_csrf_rejections_total"));
        assert!(output.contains("litellm_cors_preflight_requests_total"));
        assert!(output.contains("litellm_alerts_triggered_total"));
        assert!(output.contains("litellm_trace_spans_created_total"));
    }

    #[test]
    fn test_routing_strategy_metrics() {
        let metrics = GatewayMetrics::new();

        // Simulate routing strategy selections
        metrics
            .routing_strategy_selections
            .with_label_values(&["latency-based", "gpt-4"])
            .inc_by(50);
        metrics
            .routing_strategy_selections
            .with_label_values(&["load-based", "gpt-4"])
            .inc_by(30);
        metrics
            .routing_strategy_selections
            .with_label_values(&["cost-based", "claude-3"])
            .inc_by(20);

        // Simulate health status tracking
        metrics
            .health_status
            .with_label_values(&["gpt-4", "openai"])
            .set(1); // healthy
        metrics
            .health_status
            .with_label_values(&["claude-3", "anthropic"])
            .set(2); // degraded

        // Simulate load tracking
        metrics
            .load_tracker_load
            .with_label_values(&["gpt-4", "openai"])
            .set(75);
        metrics
            .load_tracker_load
            .with_label_values(&["claude-3", "anthropic"])
            .set(45);

        let output = metrics.render();

        assert!(output.contains("litellm_routing_strategy_selections_total"));
        assert!(output.contains("litellm_health_status"));
        assert!(output.contains("litellm_load_tracker_load"));
    }

    #[test]
    fn test_circuit_breaker_metrics() {
        let metrics = GatewayMetrics::new();

        // Simulate circuit breaker state changes
        metrics
            .circuit_breaker_state
            .with_label_values(&["openai"])
            .set(0); // closed
        metrics
            .circuit_breaker_state
            .with_label_values(&["anthropic"])
            .set(1); // open
        metrics
            .circuit_breaker_state
            .with_label_values(&["bedrock"])
            .set(2); // half-open

        // Simulate retry attempts
        metrics
            .retry_attempts
            .with_label_values(&["gpt-4"])
            .inc_by(5);
        metrics
            .retry_attempts
            .with_label_values(&["claude-3"])
            .inc_by(3);

        let output = metrics.render();

        assert!(output.contains("litellm_circuit_breaker_state"));
        assert!(output.contains("litellm_retry_attempts_total"));
    }

    #[test]
    fn test_rate_limiting_metrics() {
        let metrics = GatewayMetrics::new();

        // Simulate rate limit rejections
        metrics
            .rate_limit_rejections
            .with_label_values(&["rpm_exceeded"])
            .inc_by(10);
        metrics
            .rate_limit_rejections
            .with_label_values(&["tpm_exceeded"])
            .inc_by(5);
        metrics
            .rate_limit_rejections
            .with_label_values(&["parallel_requests"])
            .inc_by(2);

        let output = metrics.render();

        assert!(output.contains("litellm_rate_limit_rejections_total"));
    }

    #[test]
    fn test_latency_tracking_metrics() {
        let metrics = GatewayMetrics::new();

        // Simulate latency tracking
        metrics
            .latency_tracker_avg_ms
            .with_label_values(&["gpt-4", "openai"])
            .observe(150.0);
        metrics
            .latency_tracker_avg_ms
            .with_label_values(&["gpt-4", "openai"])
            .observe(200.0);
        metrics
            .latency_tracker_avg_ms
            .with_label_values(&["claude-3", "anthropic"])
            .observe(300.0);

        // Simulate request duration
        metrics
            .request_duration_seconds
            .with_label_values(&["gpt-4"])
            .observe(0.5);
        metrics
            .request_duration_seconds
            .with_label_values(&["gpt-4"])
            .observe(1.2);
        metrics
            .request_duration_seconds
            .with_label_values(&["claude-3"])
            .observe(0.8);

        metrics
            .deployment_duration_seconds
            .with_label_values(&["gpt-4", "openai"])
            .observe(0.6);
        metrics
            .deployment_duration_seconds
            .with_label_values(&["claude-3", "anthropic"])
            .observe(0.9);

        let output = metrics.render();

        assert!(output.contains("litellm_latency_tracker_avg_ms"));
        assert!(output.contains("litellm_request_duration_seconds"));
        assert!(output.contains("litellm_deployment_duration_seconds"));
    }

    #[test]
    fn test_active_requests_tracking() {
        let metrics = GatewayMetrics::new();

        // Simulate active requests
        metrics
            .active_requests
            .with_label_values(&["gpt-4"])
            .set(10);
        metrics
            .active_requests
            .with_label_values(&["claude-3"])
            .set(5);

        let output = metrics.render();

        assert!(output.contains("litellm_active_requests"));
    }

    #[test]
    fn test_comprehensive_metrics_render() {
        let metrics = GatewayMetrics::new();

        // Exercise all metrics
        metrics
            .requests_total
            .with_label_values(&["gpt-4", "success"])
            .inc();
        metrics
            .tokens_total
            .with_label_values(&["gpt-4", "prompt"])
            .inc_by(100);
        metrics
            .spend_usd_total
            .with_label_values(&["gpt-4"])
            .inc_by(0.01);
        metrics.active_requests.with_label_values(&["gpt-4"]).set(1);
        metrics
            .circuit_breaker_state
            .with_label_values(&["openai"])
            .set(0);
        metrics
            .rate_limit_rejections
            .with_label_values(&["rpm"])
            .inc();
        metrics.retry_attempts.with_label_values(&["gpt-4"]).inc();
        metrics
            .deployment_requests_total
            .with_label_values(&["gpt-4", "openai", "success"])
            .inc();
        metrics
            .deployment_duration_seconds
            .with_label_values(&["gpt-4", "openai"])
            .observe(0.5);
        metrics
            .cache_hits_total
            .with_label_values(&["gpt-4", "semantic"])
            .inc();
        metrics
            .cache_misses_total
            .with_label_values(&["gpt-4", "semantic"])
            .inc();
        metrics
            .errors_total
            .with_label_values(&["gpt-4", "timeout"])
            .inc();
        metrics
            .routing_strategy_selections
            .with_label_values(&["latency", "gpt-4"])
            .inc();
        metrics
            .health_status
            .with_label_values(&["gpt-4", "openai"])
            .set(1);
        metrics
            .load_tracker_load
            .with_label_values(&["gpt-4", "openai"])
            .set(50);
        metrics
            .latency_tracker_avg_ms
            .with_label_values(&["gpt-4", "openai"])
            .observe(100.0);
        metrics
            .validation_failures_total
            .with_label_values(&["model"])
            .inc();
        metrics
            .csrf_rejections_total
            .with_label_values(&["token"])
            .inc();
        metrics
            .cors_preflight_requests_total
            .with_label_values(&["origin", "POST"])
            .inc();
        metrics
            .alerts_triggered_total
            .with_label_values(&["error", "critical"])
            .inc();
        metrics
            .trace_spans_created_total
            .with_label_values(&["gateway"])
            .inc();

        let output = metrics.render();

        // Verify all metrics are present
        assert!(output.contains("litellm_requests_total"));
        assert!(output.contains("litellm_tokens_total"));
        assert!(output.contains("litellm_spend_usd_total"));
        assert!(output.contains("litellm_active_requests"));
        assert!(output.contains("litellm_circuit_breaker_state"));
        assert!(output.contains("litellm_rate_limit_rejections_total"));
        assert!(output.contains("litellm_retry_attempts_total"));
        assert!(output.contains("litellm_deployment_requests_total"));
        assert!(output.contains("litellm_deployment_duration_seconds"));
        assert!(output.contains("litellm_cache_hits_total"));
        assert!(output.contains("litellm_cache_misses_total"));
        assert!(output.contains("litellm_errors_total"));
        assert!(output.contains("litellm_routing_strategy_selections_total"));
        assert!(output.contains("litellm_health_status"));
        assert!(output.contains("litellm_load_tracker_load"));
        assert!(output.contains("litellm_latency_tracker_avg_ms"));
        assert!(output.contains("litellm_validation_failures_total"));
        assert!(output.contains("litellm_csrf_rejections_total"));
        assert!(output.contains("litellm_cors_preflight_requests_total"));
        assert!(output.contains("litellm_alerts_triggered_total"));
        assert!(output.contains("litellm_trace_spans_created_total"));
    }
}
