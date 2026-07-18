#![forbid(unsafe_code)]

use std::io::Write;
use std::time::Duration;

use hdrhistogram::Histogram;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("failed to serialize benchmark result: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("failed to write benchmark result: {0}")]
    Io(#[from] std::io::Error),
}

mod duration_millis {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(duration.as_millis() as u64)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let millis = u64::deserialize(deserializer)?;
        Ok(Duration::from_millis(millis))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scenario {
    pub name: String,
    pub target_rps: u32,
    pub concurrency: u32,
    pub streaming: bool,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    #[serde(with = "duration_millis")]
    pub duration: Duration,
    #[serde(with = "duration_millis")]
    pub warmup: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum Outcome {
    Success { status: u16 },
    Timeout,
    TransportError { message: String },
    UpstreamError { status: u16 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencySummary {
    pub count: u64,
    pub p50_ms: f64,
    pub p90_ms: f64,
    pub p99_ms: f64,
    pub p999_ms: f64,
    pub max_ms: f64,
}

impl LatencySummary {
    /// Converts histogram values recorded in microseconds to millisecond percentiles.
    pub fn from_histogram(histogram: &Histogram<u64>) -> Self {
        let micros_to_millis = |micros: u64| micros as f64 / 1_000.0;
        Self {
            count: histogram.len(),
            p50_ms: micros_to_millis(histogram.value_at_quantile(0.50)),
            p90_ms: micros_to_millis(histogram.value_at_quantile(0.90)),
            p99_ms: micros_to_millis(histogram.value_at_quantile(0.99)),
            p999_ms: micros_to_millis(histogram.value_at_quantile(0.999)),
            max_ms: micros_to_millis(histogram.max()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceSample {
    pub rss_bytes: u64,
    pub cpu_millis: u64,
    pub ts: u64,
}

impl ResourceSample {
    /// Stub resource sampler; real `/proc` and cgroup sampling will land later.
    pub fn sample() -> Result<Self, CoreError> {
        let ts = match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
            Ok(duration) => duration.as_millis() as u64,
            Err(_) => 0,
        };
        Ok(Self {
            rss_bytes: 0,
            cpu_millis: 0,
            ts,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchResult {
    pub gateway: String,
    pub scenario: Scenario,
    pub latency: LatencySummary,
    pub resources: ResourceSample,
}

impl BenchResult {
    pub fn append_json_line<W: Write>(&self, writer: &mut W) -> Result<(), CoreError> {
        serde_json::to_writer(&mut *writer, self)?;
        writer.write_all(b"\n")?;
        Ok(())
    }
}
