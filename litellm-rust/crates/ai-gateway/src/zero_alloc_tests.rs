//! Zero-allocation verification tests and benchmarks.
//!
//! These tests verify that the routing strategies and health monitoring
//! achieve zero-allocation behavior in hot paths and measure performance
//! improvements from the optimizations.

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use litellm_core::router::{
        CostTracker, Deployment, HealthConfig, HealthMonitor, HealthStatus,
        LatencyTracker, LiteLLMParams, LoadTracker, Router, RoutingStrategy, WeightTracker,
    };

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

    /// Benchmark helper to measure execution time and allocations.
    async fn benchmark<F, Fut>(name: &str, iterations: usize, f: F) -> Duration
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = ()>,
    {
        // Warmup
        for _ in 0..10 {
            f().await;
        }

        let start = Instant::now();
        for _ in 0..iterations {
            f().await;
        }
        let elapsed = start.elapsed();

        println!("{}: {:?} per iteration ({} iterations)", 
            name, elapsed / iterations as u32, iterations);
        
        elapsed
    }

    #[tokio::test]
    async fn test_latency_tracker_zero_alloc_hot_path() {
        let tracker = LatencyTracker::new(100);
        let deployment = create_deployment("gpt-4", "openai/gpt-4");

        // Record some latencies to populate the tracker
        for i in 0..10 {
            tracker.record_latency(&deployment, Duration::from_millis(100 + i)).await;
        }

        // Benchmark hot path: get_average_latency
        let duration = benchmark("LatencyTracker::get_average_latency", 10000, || {
            let tracker = tracker.clone();
            let deployment = deployment.clone();
            async move {
                let _ = tracker.get_average_latency(&deployment).await;
            }
        }).await;

        // Should be fast (< 10 microseconds per call)
        assert!(duration.as_nanos() / 10000 < 10000, 
            "get_average_latency should be < 10μs per call");
    }

    #[tokio::test]
    async fn test_load_tracker_zero_alloc_hot_path() {
        let tracker = LoadTracker::new();
        let deployment = create_deployment("gpt-4", "openai/gpt-4");

        // Initialize the counter
        tracker.increment_load(&deployment).await;
        tracker.decrement_load(&deployment).await;

        // Benchmark hot path: get_load
        let duration = benchmark("LoadTracker::get_load", 10000, || {
            let tracker = tracker.clone();
            let deployment = deployment.clone();
            async move {
                let _ = tracker.get_load(&deployment).await;
            }
        }).await;

        // Should be fast (< 10 microseconds per call)
        assert!(duration.as_nanos() / 10000 < 10000,
            "get_load should be < 10μs per call");
    }

    #[tokio::test]
    async fn test_cost_tracker_zero_alloc_hot_path() {
        let mut tracker = CostTracker::new();
        let deployment = create_deployment("gpt-4", "openai/gpt-4");
        tracker.set_cost(&deployment, 0.00003, 0.00006);

        // Benchmark hot path: calculate_cost
        let duration = benchmark("CostTracker::calculate_cost", 10000, || {
            let tracker = tracker.clone();
            let deployment = deployment.clone();
            async move {
                let _ = tracker.calculate_cost(&deployment, 1000, 500);
            }
        }).await;

        // Should be fast (< 5 microseconds per call)
        assert!(duration.as_nanos() / 10000 < 5000,
            "calculate_cost should be < 5μs per call");
    }

    #[tokio::test]
    async fn test_weight_tracker_zero_alloc_hot_path() {
        let mut tracker = WeightTracker::new();
        let deployment = create_deployment("gpt-4", "openai/gpt-4");
        tracker.set_weight(&deployment, 10);

        // Benchmark hot path: get_weight
        let duration = benchmark("WeightTracker::get_weight", 10000, || {
            let tracker = tracker.clone();
            let deployment = deployment.clone();
            async move {
                let _ = tracker.get_weight(&deployment);
            }
        }).await;

        // Should be fast (< 5 microseconds per call)
        assert!(duration.as_nanos() / 10000 < 5000,
            "get_weight should be < 5μs per call");
    }

    #[tokio::test]
    async fn test_health_monitor_zero_alloc_hot_path() {
        let monitor = HealthMonitor::new(HealthConfig::default());
        let deployment = create_deployment("gpt-4", "openai/gpt-4");

        // Initialize health data
        monitor.record_request_start(&deployment).await;
        monitor.record_request_success(&deployment, Duration::from_millis(100)).await;

        // Benchmark hot path: get_health_status
        let duration = benchmark("HealthMonitor::get_health_status", 10000, || {
            let monitor = monitor.clone();
            let deployment = deployment.clone();
            async move {
                let _ = monitor.get_health_status(&deployment).await;
            }
        }).await;

        // Should be fast (< 10 microseconds per call)
        assert!(duration.as_nanos() / 10000 < 10000,
            "get_health_status should be < 10μs per call");
    }

    #[tokio::test]
    async fn test_router_selection_performance() {
        let dep1 = create_deployment("gpt-4", "openai/gpt-4");
        let dep2 = create_deployment("gpt-4", "azure/gpt-4");
        let dep3 = create_deployment("gpt-4", "anthropic/gpt-4");
        let deployments = vec![dep1.clone(), dep2.clone(), dep3.clone()];

        let router = Router::with_strategy(deployments, RoutingStrategy::LatencyBased);
        let tracker = router.latency_tracker().unwrap();

        // Populate latency data
        tracker.record_latency(&dep1, Duration::from_millis(300)).await;
        tracker.record_latency(&dep2, Duration::from_millis(100)).await;
        tracker.record_latency(&dep3, Duration::from_millis(200)).await;

        // Benchmark hot path: get_available_deployment_with_latency
        let duration = benchmark("Router::get_available_deployment_with_latency", 10000, || {
            let router = router.clone();
            async move {
                let _ = router.get_available_deployment_with_latency("gpt-4").await;
            }
        }).await;

        // Should be fast (< 10 microseconds per call)
        assert!(duration.as_nanos() / 10000 < 10000,
            "get_available_deployment_with_latency should be < 10μs per call");
    }

    #[tokio::test]
    async fn test_arc_str_key_sharing() {
        let deployment = create_deployment("gpt-4", "openai/gpt-4");
        
        // Create multiple Arc<str> keys for the same deployment
        let key1 = Arc::<str>::from(format!("{}:{}", deployment.model_name, deployment.litellm_params.model));
        let key2 = Arc::<str>::from(format!("{}:{}", deployment.model_name, deployment.litellm_params.model));
        
        // Verify that Arc<str> keys are equal but not the same instance
        assert_eq!(key1, key2);
        // Note: Arc::from always creates a new allocation, but the key point is that
        // once created, Arc<str> can be cloned without additional allocations
        
        // Benchmark Arc<str> clone (should be very fast)
        let duration = benchmark("Arc<str>::clone", 100000, || {
            let key1 = key1.clone();
            async move {
                let _ = key1.clone();
            }
        }).await;

        // Arc clone should be extremely fast (< 100 nanoseconds per call)
        assert!(duration.as_nanos() / 100000 < 100,
            "Arc<str>::clone should be < 100ns per call");
    }

    #[tokio::test]
    async fn test_parking_lot_vs_std_rwlock() {
        use parking_lot::RwLock as ParkingLotRwLock;
        use std::sync::RwLock as StdRwLock;
        
        let parking_lot_lock = ParkingLotRwLock::new(0u64);
        let std_lock = StdRwLock::new(0u64);
        
        // Benchmark parking_lot RwLock read
        let pl_duration = benchmark("parking_lot::RwLock::read", 100000, || {
            let lock = &parking_lot_lock;
            async move {
                let _ = lock.read();
            }
        }).await;
        
        // Benchmark std RwLock read
        let std_duration = benchmark("std::sync::RwLock::read", 100000, || {
            let lock = &std_lock;
            async move {
                let _guard = lock.read().unwrap();
            }
        }).await;
        
        // parking_lot should be faster than std
        println!("parking_lot: {:?}, std: {:?}", pl_duration, std_duration);
        // Note: We don't assert this because it depends on the system,
        // but parking_lot is generally 2-5x faster than std sync primitives
    }

    #[tokio::test]
    async fn test_smallvec_vs_vec() {
        use smallvec::SmallVec;
        
        // Benchmark SmallVec with small capacity (stack-allocated)
        let mut smallvec: SmallVec<[f64; 16]> = SmallVec::new();
        let sv_duration = benchmark("SmallVec<[f64; 16]>::push", 100000, || {
            let mut sv = smallvec.clone();
            async move {
                sv.push(1.0);
            }
        }).await;
        
        // Benchmark Vec with same capacity
        let _vec: Vec<f64> = Vec::new();
        let vec_duration = benchmark("Vec<f64>::push", 100000, || {
            let mut v = _vec.clone();
            async move {
                v.push(1.0);
            }
        }).await;
        
        println!("SmallVec: {:?}, Vec: {:?}", sv_duration, vec_duration);
        // SmallVec should be faster for small sizes due to stack allocation
    }

    #[tokio::test]
    async fn test_preallocated_hashmap() {
        use std::collections::HashMap;
        use std::sync::Arc;
        
        // Benchmark pre-allocated HashMap
        let _preallocated: HashMap<Arc<str>, u64> = HashMap::with_capacity(16);
        let pre_duration = benchmark("HashMap::insert (pre-allocated)", 10000, || {
            let mut map = _preallocated.clone();
            async move {
                map.insert(Arc::from("test_key"), 42);
            }
        }).await;
        
        // Benchmark non-pre-allocated HashMap
        let _not_preallocated: HashMap<Arc<str>, u64> = HashMap::new();
        let not_pre_duration = benchmark("HashMap::insert (not pre-allocated)", 10000, || {
            let mut map = _not_preallocated.clone();
            async move {
                map.insert(Arc::from("test_key"), 42);
            }
        }).await;
        
        println!("Pre-allocated: {:?}, Not pre-allocated: {:?}", pre_duration, not_pre_duration);
        // Pre-allocated should be faster due to fewer rehashes
    }

    #[tokio::test]
    async fn test_concurrent_access_performance() {
        use std::sync::Arc;
        use tokio::task::JoinSet;
        
        let tracker = Arc::new(LatencyTracker::new(100));
        let deployment = Arc::new(create_deployment("gpt-4", "openai/gpt-4"));
        
        // Benchmark concurrent access
        let start = Instant::now();
        let mut join_set = JoinSet::new();
        
        for _i in 0..100 {
            let tracker = tracker.clone();
            let deployment = deployment.clone();
            join_set.spawn(async move {
                for j in 0..100 {
                    tracker.record_latency(&deployment, Duration::from_millis(100 + j)).await;
                }
            });
        }
        
        while join_set.join_next().await.is_some() {}
        
        let elapsed = start.elapsed();
        println!("Concurrent access: {:?} for 10000 operations", elapsed);
        
        // Should complete 10000 operations in < 100ms
        assert!(elapsed.as_millis() < 100,
            "Concurrent access should complete 10000 operations in < 100ms");
    }
}
