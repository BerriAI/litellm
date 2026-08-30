//! Comprehensive load testing framework.
//!
//! Provides tools for load testing the gateway under various conditions
//! to identify performance limits and bottlenecks.

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use serde::{Deserialize, Serialize};

/// Load test configuration.
#[derive(Debug, Clone)]
pub struct LoadTestConfig {
    /// Number of concurrent clients.
    pub concurrency: usize,
    /// Total number of requests to send.
    pub total_requests: usize,
    /// Request timeout.
    pub timeout: Duration,
    /// Ramp-up duration.
    pub ramp_up: Duration,
}

impl Default for LoadTestConfig {
    fn default() -> Self {
        Self {
            concurrency: 100,
            total_requests: 10000,
            timeout: Duration::from_secs(30),
            ramp_up: Duration::from_secs(10),
        }
    }
}

/// Load test result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadTestResult {
    pub config: LoadTestConfigSummary,
    pub total_requests: usize,
    pub successful_requests: usize,
    pub failed_requests: usize,
    pub duration: Duration,
    pub requests_per_second: f64,
    pub latency_stats: LatencyStats,
    pub error_distribution: std::collections::HashMap<String, usize>,
}

/// Load test config summary (for serialization).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadTestConfigSummary {
    pub concurrency: usize,
    pub total_requests: usize,
    pub timeout_ms: u64,
    pub ramp_up_ms: u64,
}

/// Latency statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStats {
    pub min_ms: f64,
    pub max_ms: f64,
    pub mean_ms: f64,
    pub median_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
}

/// Load tester.
pub struct LoadTester {
    config: LoadTestConfig,
    results: Arc<Mutex<Vec<LoadTestResult>>>,
}

impl LoadTester {
    /// Create a new load tester.
    pub fn new(config: LoadTestConfig) -> Self {
        Self {
            config,
            results: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Run a load test.
    pub async fn run_test<F, Fut>(
        &self,
        request_fn: F,
    ) -> LoadTestResult
    where
        F: Fn() -> Fut + Send + Sync + 'static + Clone,
        Fut: Future<Output = Result<(), String>> + Send,
    {
        let start = Instant::now();
        let successful = Arc::new(Mutex::new(0usize));
        let failed = Arc::new(Mutex::new(0usize));
        let latencies = Arc::new(Mutex::new(Vec::<f64>::new()));
        let errors = Arc::new(Mutex::new(std::collections::HashMap::<String, usize>::new()));

        let mut handles = Vec::new();

        for _ in 0..self.config.concurrency {
            let request_fn = request_fn.clone();
            let successful = Arc::clone(&successful);
            let failed = Arc::clone(&failed);
            let latencies = Arc::clone(&latencies);
            let errors = Arc::clone(&errors);
            let requests_per_client = self.config.total_requests / self.config.concurrency;

            let handle = tokio::spawn(async move {
                for _ in 0..requests_per_client {
                    let start = Instant::now();
                    match request_fn().await {
                        Ok(_) => {
                            let latency = start.elapsed().as_secs_f64() * 1000.0;
                            latencies.lock().await.push(latency);
                            *successful.lock().await += 1;
                        }
                        Err(e) => {
                            let mut errors = errors.lock().await;
                            *errors.entry(e).or_insert(0) += 1;
                            *failed.lock().await += 1;
                        }
                    }
                }
            });

            handles.push(handle);
        }

        // Wait for all clients to complete
        for handle in handles {
            let _ = handle.await;
        }

        let duration = start.elapsed();
        let successful = *successful.lock().await;
        let failed = *failed.lock().await;
        let latencies = latencies.lock().await.clone();
        let error_distribution = errors.lock().await.clone();

        let latency_stats = Self::calculate_latency_stats(&latencies);
        let rps = successful as f64 / duration.as_secs_f64();

        let result = LoadTestResult {
            config: LoadTestConfigSummary {
                concurrency: self.config.concurrency,
                total_requests: self.config.total_requests,
                timeout_ms: self.config.timeout.as_millis() as u64,
                ramp_up_ms: self.config.ramp_up.as_millis() as u64,
            },
            total_requests: self.config.total_requests,
            successful_requests: successful,
            failed_requests: failed,
            duration,
            requests_per_second: rps,
            latency_stats,
            error_distribution,
        };

        self.results.lock().await.push(result.clone());
        result
    }

    /// Calculate latency statistics.
    fn calculate_latency_stats(latencies: &[f64]) -> LatencyStats {
        if latencies.is_empty() {
            return LatencyStats {
                min_ms: 0.0,
                max_ms: 0.0,
                mean_ms: 0.0,
                median_ms: 0.0,
                p95_ms: 0.0,
                p99_ms: 0.0,
            };
        }

        let mut sorted = latencies.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean = latencies.iter().sum::<f64>() / latencies.len() as f64;
        let median = sorted[sorted.len() / 2];
        let p95_index = (sorted.len() as f64 * 0.95) as usize;
        let p99_index = (sorted.len() as f64 * 0.99) as usize;

        LatencyStats {
            min_ms: sorted[0],
            max_ms: sorted[sorted.len() - 1],
            mean_ms: mean,
            median_ms: median,
            p95_ms: sorted[p95_index.min(sorted.len() - 1)],
            p99_ms: sorted[p99_index.min(sorted.len() - 1)],
        }
    }

    /// Get all test results.
    pub async fn get_results(&self) -> Vec<LoadTestResult> {
        self.results.lock().await.clone()
    }

    /// Generate a summary report.
    pub async fn generate_report(&self) -> String {
        let results = self.results.lock().await;
        let mut report = String::new();

        report.push_str("# Load Test Report\n\n");

        for (i, result) in results.iter().enumerate() {
            report.push_str(&format!("## Test {}\n\n", i + 1));
            report.push_str(&format!("- Concurrency: {}\n", result.config.concurrency));
            report.push_str(&format!("- Total Requests: {}\n", result.total_requests));
            report.push_str(&format!("- Successful: {}\n", result.successful_requests));
            report.push_str(&format!("- Failed: {}\n", result.failed_requests));
            report.push_str(&format!("- Duration: {:.2}s\n", result.duration.as_secs_f64()));
            report.push_str(&format!("- RPS: {:.2}\n\n", result.requests_per_second));

            report.push_str("### Latency Statistics\n\n");
            report.push_str(&format!("- Min: {:.2}ms\n", result.latency_stats.min_ms));
            report.push_str(&format!("- Max: {:.2}ms\n", result.latency_stats.max_ms));
            report.push_str(&format!("- Mean: {:.2}ms\n", result.latency_stats.mean_ms));
            report.push_str(&format!("- Median: {:.2}ms\n", result.latency_stats.median_ms));
            report.push_str(&format!("- P95: {:.2}ms\n", result.latency_stats.p95_ms));
            report.push_str(&format!("- P99: {:.2}ms\n\n", result.latency_stats.p99_ms));

            if !result.error_distribution.is_empty() {
                report.push_str("### Error Distribution\n\n");
                for (error, count) in &result.error_distribution {
                    report.push_str(&format!("- {}: {}\n", error, count));
                }
                report.push('\n');
            }
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_stats_empty() {
        let stats = LoadTester::calculate_latency_stats(&[]);
        assert_eq!(stats.min_ms, 0.0);
        assert_eq!(stats.max_ms, 0.0);
    }

    #[test]
    fn test_latency_stats_calculation() {
        let latencies = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0];
        let stats = LoadTester::calculate_latency_stats(&latencies);

        assert_eq!(stats.min_ms, 10.0);
        assert_eq!(stats.max_ms, 100.0);
        assert_eq!(stats.mean_ms, 55.0);
        assert_eq!(stats.median_ms, 60.0);
    }

    #[tokio::test]
    async fn test_load_tester_creation() {
        let config = LoadTestConfig::default();
        let tester = LoadTester::new(config);
        let results = tester.get_results().await;
        assert!(results.is_empty());
    }
}
