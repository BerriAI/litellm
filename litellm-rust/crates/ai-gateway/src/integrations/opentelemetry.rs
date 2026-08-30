//! OpenTelemetry callback integration for distributed tracing.
//!
//! This module provides OpenTelemetry integration for the AI gateway,
//! enabling distributed tracing across all LLM calls.

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// OpenTelemetry callback configuration.
#[derive(Debug, Clone)]
pub struct OpenTelemetryConfig {
    /// Service name for tracing.
    pub service_name: String,
    /// OTLP endpoint for exporting traces.
    pub otlp_endpoint: String,
    /// Whether to enable tracing.
    pub enabled: bool,
}

impl Default for OpenTelemetryConfig {
    fn default() -> Self {
        Self {
            service_name: "litellm-gateway".to_string(),
            otlp_endpoint: "http://localhost:4317".to_string(),
            enabled: true,
        }
    }
}

/// OpenTelemetry callback for distributed tracing.
pub struct OpenTelemetryCallback {
    config: OpenTelemetryConfig,
    /// Trace storage for testing and inspection.
    traces: Arc<RwLock<Vec<TraceData>>>,
}

/// Trace data structure for storing trace information.
#[derive(Debug, Clone)]
pub struct TraceData {
    /// Trace ID.
    pub trace_id: String,
    /// Span ID.
    pub span_id: String,
    /// Operation name.
    pub operation: String,
    /// Start time in nanoseconds.
    pub start_time: u64,
    /// End time in nanoseconds.
    pub end_time: u64,
    /// Tags/attributes for the span.
    pub tags: HashMap<String, String>,
    /// Logs/events for the span.
    pub logs: Vec<LogEntry>,
}

/// Log entry for a span.
#[derive(Debug, Clone)]
pub struct LogEntry {
    /// Timestamp in nanoseconds.
    pub timestamp: u64,
    /// Log message.
    pub message: String,
    /// Log level.
    pub level: String,
}

impl OpenTelemetryCallback {
    /// Create a new OpenTelemetry callback.
    pub fn new(config: OpenTelemetryConfig) -> Self {
        Self {
            config,
            traces: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Start a new trace span.
    pub fn start_span(&self, operation: &str) -> String {
        let trace_id = self.generate_trace_id();
        let span_id = self.generate_span_id();
        let start_time = self.current_time_nanos();

        let trace = TraceData {
            trace_id: trace_id.clone(),
            span_id: span_id.clone(),
            operation: operation.to_string(),
            start_time,
            end_time: 0,
            tags: HashMap::new(),
            logs: vec![],
        };

        self.traces.write().push(trace);
        span_id
    }

    /// End a trace span.
    pub fn end_span(&self, span_id: &str) {
        let end_time = self.current_time_nanos();
        let mut traces = self.traces.write();

        if let Some(trace) = traces.iter_mut().find(|t| t.span_id == span_id) {
            trace.end_time = end_time;
        }
    }

    /// Add a tag to a span.
    pub fn add_tag(&self, span_id: &str, key: &str, value: &str) {
        let mut traces = self.traces.write();

        if let Some(trace) = traces.iter_mut().find(|t| t.span_id == span_id) {
            trace.tags.insert(key.to_string(), value.to_string());
        }
    }

    /// Add a log entry to a span.
    pub fn add_log(&self, span_id: &str, message: &str, level: &str) {
        let timestamp = self.current_time_nanos();
        let mut traces = self.traces.write();

        if let Some(trace) = traces.iter_mut().find(|t| t.span_id == span_id) {
            trace.logs.push(LogEntry {
                timestamp,
                message: message.to_string(),
                level: level.to_string(),
            });
        }
    }

    /// Get all traces (for testing).
    pub fn get_traces(&self) -> Vec<TraceData> {
        self.traces.read().clone()
    }

    /// Generate a unique trace ID.
    fn generate_trace_id(&self) -> String {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let bytes: [u8; 16] = rng.r#gen();
        bytes.iter().map(|b| format!("{:02x}", b)).collect()
    }

    /// Generate a unique span ID.
    fn generate_span_id(&self) -> String {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let bytes: [u8; 8] = rng.r#gen();
        bytes.iter().map(|b| format!("{:02x}", b)).collect()
    }

    /// Get current time in nanoseconds.
    fn current_time_nanos(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }

    /// Record an LLM call trace.
    pub fn record_llm_call(
        &self,
        model: &str,
        provider: &str,
        prompt_tokens: u64,
        completion_tokens: u64,
        latency_ms: u64,
    ) {
        let span_id = self.start_span("llm_call");

        self.add_tag(&span_id, "model", model);
        self.add_tag(&span_id, "provider", provider);
        self.add_tag(&span_id, "prompt_tokens", &prompt_tokens.to_string());
        self.add_tag(
            &span_id,
            "completion_tokens",
            &completion_tokens.to_string(),
        );
        self.add_tag(&span_id, "latency_ms", &latency_ms.to_string());

        self.add_log(&span_id, "LLM call completed", "info");
        self.end_span(&span_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_otel_callback_creation() {
        let config = OpenTelemetryConfig::default();
        let callback = OpenTelemetryCallback::new(config);
        assert!(callback.get_traces().is_empty());
    }

    #[test]
    fn test_start_and_end_span() {
        let config = OpenTelemetryConfig::default();
        let callback = OpenTelemetryCallback::new(config);

        let span_id = callback.start_span("test_operation");
        assert_eq!(callback.get_traces().len(), 1);

        callback.end_span(&span_id);
        let traces = callback.get_traces();
        assert_eq!(traces.len(), 1);
        assert!(traces[0].end_time > 0);
    }

    #[test]
    fn test_add_tags() {
        let config = OpenTelemetryConfig::default();
        let callback = OpenTelemetryCallback::new(config);

        let span_id = callback.start_span("test_operation");
        callback.add_tag(&span_id, "key1", "value1");
        callback.add_tag(&span_id, "key2", "value2");

        let traces = callback.get_traces();
        assert_eq!(traces[0].tags.len(), 2);
        assert_eq!(traces[0].tags.get("key1"), Some(&"value1".to_string()));
        assert_eq!(traces[0].tags.get("key2"), Some(&"value2".to_string()));
    }

    #[test]
    fn test_add_logs() {
        let config = OpenTelemetryConfig::default();
        let callback = OpenTelemetryCallback::new(config);

        let span_id = callback.start_span("test_operation");
        callback.add_log(&span_id, "Test message", "info");

        let traces = callback.get_traces();
        assert_eq!(traces[0].logs.len(), 1);
        assert_eq!(traces[0].logs[0].message, "Test message");
        assert_eq!(traces[0].logs[0].level, "info");
    }

    #[test]
    fn test_record_llm_call() {
        let config = OpenTelemetryConfig::default();
        let callback = OpenTelemetryCallback::new(config);

        callback.record_llm_call("gpt-4", "openai", 100, 50, 150);

        let traces = callback.get_traces();
        assert_eq!(traces.len(), 1);
        assert_eq!(traces[0].tags.get("model"), Some(&"gpt-4".to_string()));
        assert_eq!(traces[0].tags.get("provider"), Some(&"openai".to_string()));
        assert_eq!(
            traces[0].tags.get("prompt_tokens"),
            Some(&"100".to_string())
        );
        assert_eq!(
            traces[0].tags.get("completion_tokens"),
            Some(&"50".to_string())
        );
        assert_eq!(traces[0].tags.get("latency_ms"), Some(&"150".to_string()));
    }
}
