//! Tests for advanced routing strategies.

#[cfg(test)]
mod tests {
    use litellm_core::router::{
        CostTracker, Deployment, HealthStatus, LiteLLMParams, Router, RoutingStrategy,
        WeightTracker,
    };
    use std::time::Duration;

    fn create_deployment(model_name: &str, model: &str) -> Deployment {
        Deployment {
            model_name: model_name.to_string(),
            litellm_params: LiteLLMParams {
                model: model.to_string(),
                api_key: None,
                api_base: None,
            },
            healthy: Some(true),
            weight: None,
            input_cost_per_token: None,
            output_cost_per_token: None,
        }
    }

    fn create_deployment_with_cost(
        model_name: &str,
        model: &str,
        input_cost: f64,
        output_cost: f64,
    ) -> Deployment {
        Deployment {
            model_name: model_name.to_string(),
            litellm_params: LiteLLMParams {
                model: model.to_string(),
                api_key: None,
                api_base: None,
            },
            healthy: Some(true),
            weight: None,
            input_cost_per_token: Some(input_cost),
            output_cost_per_token: Some(output_cost),
        }
    }

    fn create_deployment_with_weight(model_name: &str, model: &str, weight: u32) -> Deployment {
        Deployment {
            model_name: model_name.to_string(),
            litellm_params: LiteLLMParams {
                model: model.to_string(),
                api_key: None,
                api_base: None,
            },
            healthy: Some(true),
            weight: Some(weight),
            input_cost_per_token: None,
            output_cost_per_token: None,
        }
    }

    #[tokio::test]
    async fn test_router_with_latency_strategy() {
        let dep1 = create_deployment("gpt-4", "openai/gpt-4");
        let dep2 = create_deployment("gpt-4", "azure/gpt-4");
        let deployments = vec![dep1, dep2];

        let router = Router::with_strategy(deployments, RoutingStrategy::LatencyBased);
        assert_eq!(router.strategy(), RoutingStrategy::LatencyBased);
        assert!(router.latency_tracker().is_some());
    }

    #[tokio::test]
    async fn test_router_with_load_strategy() {
        let dep1 = create_deployment("gpt-4", "openai/gpt-4");
        let dep2 = create_deployment("gpt-4", "azure/gpt-4");
        let deployments = vec![dep1, dep2];

        let router = Router::with_strategy(deployments, RoutingStrategy::LoadBased);
        assert_eq!(router.strategy(), RoutingStrategy::LoadBased);
        assert!(router.load_tracker().is_some());
    }

    #[tokio::test]
    async fn test_router_with_cost_strategy() {
        let dep1 = create_deployment("gpt-4", "openai/gpt-4");
        let dep2 = create_deployment("gpt-4", "azure/gpt-4");
        let deployments = vec![dep1, dep2];

        let router = Router::with_strategy(deployments, RoutingStrategy::CostBased);
        assert_eq!(router.strategy(), RoutingStrategy::CostBased);
        assert!(router.cost_tracker().is_some());
    }

    #[tokio::test]
    async fn test_router_with_weighted_strategy() {
        let dep1 = create_deployment("gpt-4", "openai/gpt-4");
        let dep2 = create_deployment("gpt-4", "azure/gpt-4");
        let deployments = vec![dep1, dep2];

        let router = Router::with_strategy(deployments, RoutingStrategy::Weighted);
        assert_eq!(router.strategy(), RoutingStrategy::Weighted);
        assert!(router.weight_tracker().is_some());
    }

    #[tokio::test]
    async fn test_latency_based_routing_integration() {
        let dep1 = create_deployment("gpt-4", "openai/gpt-4");
        let dep2 = create_deployment("gpt-4", "azure/gpt-4");
        let deployments = vec![dep1.clone(), dep2.clone()];

        let router = Router::with_strategy(deployments, RoutingStrategy::LatencyBased);
        let tracker = router.latency_tracker().unwrap();

        // Record latencies: dep2 is faster
        tracker
            .record_latency(&dep1, Duration::from_millis(300))
            .await;
        tracker
            .record_latency(&dep2, Duration::from_millis(100))
            .await;

        let selected = router
            .get_available_deployment_with_latency("gpt-4")
            .await
            .unwrap();
        assert_eq!(selected.litellm_params.model, "azure/gpt-4");
    }

    #[tokio::test]
    async fn test_load_based_routing_integration() {
        let dep1 = create_deployment("gpt-4", "openai/gpt-4");
        let dep2 = create_deployment("gpt-4", "azure/gpt-4");
        let deployments = vec![dep1.clone(), dep2.clone()];

        let router = Router::with_strategy(deployments, RoutingStrategy::LoadBased);
        let tracker = router.load_tracker().unwrap();

        // Simulate load: dep1 has more concurrent requests
        tracker.increment_load(&dep1).await;
        tracker.increment_load(&dep1).await;
        tracker.increment_load(&dep1).await;
        tracker.increment_load(&dep2).await;

        let selected = router
            .get_available_deployment_with_load("gpt-4")
            .await
            .unwrap();
        assert_eq!(selected.litellm_params.model, "azure/gpt-4");
    }

    #[tokio::test]
    async fn test_cost_based_routing_integration() {
        let dep1 = create_deployment_with_cost("gpt-4", "openai/gpt-4", 0.00003, 0.00006);
        let dep2 = create_deployment_with_cost("gpt-4", "azure/gpt-4", 0.00002, 0.00004);

        // Test CostTracker directly
        let mut tracker = CostTracker::new();
        tracker.set_cost(&dep1, 0.00003, 0.00006);
        tracker.set_cost(&dep2, 0.00002, 0.00004);

        let candidates = vec![&dep1, &dep2];
        let selected = tracker.select(&candidates, 1000, 500).unwrap();
        assert_eq!(selected.litellm_params.model, "azure/gpt-4");
    }

    #[tokio::test]
    async fn test_weighted_routing_integration() {
        let dep1 = create_deployment_with_weight("gpt-4", "openai/gpt-4", 90);
        let dep2 = create_deployment_with_weight("gpt-4", "azure/gpt-4", 10);

        // Test WeightTracker directly
        let mut tracker = WeightTracker::new();
        tracker.set_weight(&dep1, 90);
        tracker.set_weight(&dep2, 10);

        // Run selection many times to verify distribution
        let mut dep1_count = 0;
        let candidates = vec![&dep1, &dep2];

        for _ in 0..1000 {
            let selected = tracker.select(&candidates).unwrap();
            if selected.litellm_params.model == "openai/gpt-4" {
                dep1_count += 1;
            }
        }

        // dep1 should be selected roughly 90% of the time
        let dep1_percentage = dep1_count as f64 / 1000.0 * 100.0;
        assert!(dep1_percentage > 80.0 && dep1_percentage < 95.0);
    }

    #[tokio::test]
    async fn test_health_monitor_integration() {
        let dep1 = create_deployment("gpt-4", "openai/gpt-4");
        let dep2 = create_deployment("gpt-4", "azure/gpt-4");
        let deployments = vec![dep1.clone(), dep2.clone()];

        let router = Router::with_strategy(deployments, RoutingStrategy::SimpleShuffle);
        let monitor = router.health_monitor();

        // Record some requests
        monitor.record_request_start(&dep1).await;
        monitor
            .record_request_success(&dep1, Duration::from_millis(100))
            .await;

        monitor.record_request_start(&dep2).await;
        monitor
            .record_request_failure(&dep2, Duration::from_millis(200))
            .await;

        // Check health status
        let status1 = monitor.get_health_status(&dep1).await;
        let status2 = monitor.get_health_status(&dep2).await;

        // Both should be Unknown initially (not enough data)
        assert_eq!(status1, HealthStatus::Unknown);
        assert_eq!(status2, HealthStatus::Unknown);
    }

    #[tokio::test]
    async fn test_routing_strategy_config_parsing() {
        use crate::config::RouterSettings;

        let settings = RouterSettings {
            routing_strategy: Some("latency-based".to_string()),
            num_retries: Some(3),
            timeout: Some(300),
            cooldown_seconds: Some(60),
            allowed_fails: Some(3),
        };

        assert_eq!(
            settings.to_routing_strategy(),
            RoutingStrategy::LatencyBased
        );

        let settings = RouterSettings {
            routing_strategy: Some("load-based".to_string()),
            ..Default::default()
        };
        assert_eq!(settings.to_routing_strategy(), RoutingStrategy::LoadBased);

        let settings = RouterSettings {
            routing_strategy: Some("cost-based".to_string()),
            ..Default::default()
        };
        assert_eq!(settings.to_routing_strategy(), RoutingStrategy::CostBased);

        let settings = RouterSettings {
            routing_strategy: Some("weighted".to_string()),
            ..Default::default()
        };
        assert_eq!(settings.to_routing_strategy(), RoutingStrategy::Weighted);

        let settings = RouterSettings {
            routing_strategy: Some("simple-shuffle".to_string()),
            ..Default::default()
        };
        assert_eq!(
            settings.to_routing_strategy(),
            RoutingStrategy::SimpleShuffle
        );

        let settings = RouterSettings {
            routing_strategy: None,
            ..Default::default()
        };
        assert_eq!(
            settings.to_routing_strategy(),
            RoutingStrategy::SimpleShuffle
        );
    }

    #[tokio::test]
    async fn test_config_loading_with_routing_strategy() {
        use crate::config::load_config_from_yaml;
        use std::io::Write;
        use tempfile::NamedTempFile;

        let yaml = r#"
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: test-key
  - model_name: gpt-4
    litellm_params:
      model: azure/gpt-4
      api_key: test-key

router_settings:
  routing_strategy: latency-based
  num_retries: 5
  timeout: 300
"#;

        let mut file = NamedTempFile::new().unwrap();
        file.write_all(yaml.as_bytes()).unwrap();

        let config = load_config_from_yaml(file.path().to_str().unwrap()).unwrap();
        assert_eq!(config.router.strategy(), RoutingStrategy::LatencyBased);
        assert!(config.router.latency_tracker().is_some());
    }
}
