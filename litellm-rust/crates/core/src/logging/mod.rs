mod redaction;

pub mod events;
pub mod http;
pub mod stream;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use crate::call_lifecycle::CallLifecycleContext;

pub trait LogSink: Send + Sync {
    fn emit(&self, event: &LogEvent);
}

pub struct CallLogger {
    context: CallLifecycleContext,
    sink: Option<Arc<dyn LogSink>>,
    started: Instant,
    bytes_received: AtomicUsize,
    frames_received: AtomicUsize,
    events_decoded: AtomicUsize,
}

impl CallLogger {
    pub fn new(context: &CallLifecycleContext, sink: Option<Arc<dyn LogSink>>) -> Self {
        Self {
            context: context.clone(),
            sink,
            started: Instant::now(),
            bytes_received: AtomicUsize::new(0),
            frames_received: AtomicUsize::new(0),
            events_decoded: AtomicUsize::new(0),
        }
    }

    pub(crate) fn call_id(&self) -> &str {
        &self.context.litellm_call_id
    }

    pub(crate) fn provider(&self) -> &str {
        &self.context.custom_llm_provider
    }

    pub fn request_about_to_be_sent(&self, mut input: events::RequestEventInput) {
        input.call_id = self.context.litellm_call_id.clone();
        input.provider = self.context.custom_llm_provider.clone();
        self.emit(events::request_event(input));
    }

    pub fn response_received(
        &self,
        status: u16,
        headers: Vec<(String, String)>,
        body: ResponseBody,
    ) {
        self.emit(events::response_event(events::ResponseEventInput {
            call_id: self.context.litellm_call_id.clone(),
            provider: self.context.custom_llm_provider.clone(),
            status,
            duration_ms: self.started.elapsed().as_millis(),
            headers,
            body,
        }));
    }

    pub fn failure(
        &self,
        status: Option<u16>,
        kind: &'static str,
        message: String,
        body: Option<ResponseBody>,
    ) {
        self.emit(events::error_event(events::ErrorEventInput {
            call_id: self.context.litellm_call_id.clone(),
            provider: self.context.custom_llm_provider.clone(),
            duration_ms: self.started.elapsed().as_millis(),
            status,
            kind,
            message,
            body,
        }));
    }

    pub fn stream_started(&self, status: u16, content_type: Option<String>) {
        self.emit(events::stream_started(
            self.context.litellm_call_id.clone(),
            self.context.custom_llm_provider.clone(),
            status,
            content_type,
        ));
    }

    pub fn stream_finished(&self) {
        self.emit(events::stream_completed(
            self.context.litellm_call_id.clone(),
            self.context.custom_llm_provider.clone(),
            self.started.elapsed().as_millis(),
            self.bytes_received.load(Ordering::Relaxed),
            self.frames_received.load(Ordering::Relaxed),
            self.events_decoded.load(Ordering::Relaxed),
        ));
    }

    pub fn stream_chunk_observed(&self, bytes: usize, events: usize) {
        self.bytes_received.fetch_add(bytes, Ordering::Relaxed);
        self.frames_received.fetch_add(1, Ordering::Relaxed);
        self.events_decoded.fetch_add(events, Ordering::Relaxed);
    }

    fn emit(&self, event: LogEvent) {
        if let Some(sink) = &self.sink {
            sink.emit(&event);
        }
    }
}

pub use events::{
    BodySnapshot, ErrorEventInput, LogEvent, ProviderErrorEvent, ProviderRequestEvent,
    ProviderResponseEvent, ProviderStreamCompletedEvent, ProviderStreamStartedEvent,
    RequestEventInput, ResponseBody, ResponseEventInput,
};

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    #[derive(Clone, Default)]
    struct RecordingSink(Arc<Mutex<Vec<LogEvent>>>);

    impl LogSink for RecordingSink {
        fn emit(&self, event: &LogEvent) {
            self.0.lock().expect("recording lock").push(event.clone());
        }
    }

    #[test]
    fn logger_uses_lifecycle_context_and_redacts_events() {
        let sink = RecordingSink::default();
        let context = CallLifecycleContext::new("messages", "claude", "anthropic", "req_123");
        let logger = CallLogger::new(&context, Some(Arc::new(sink.clone())));
        logger.request_about_to_be_sent(events::RequestEventInput {
            call_id: String::new(),
            provider: String::new(),
            model: "claude".to_string(),
            stream: false,
            url: "https://example.test/v1/messages".to_string(),
            headers: vec![("authorization".to_string(), "Bearer secret".to_string())],
            body: serde_json::json!({"token": "secret", "prompt": "visible"}),
        });

        let events = sink.0.lock().expect("recording lock");
        let serialized = serde_json::to_string(&events[0]).expect("event serializes");
        assert!(serialized.contains("\"call_id\":\"req_123\""));
        assert!(!serialized.contains("secret"));
        assert!(serialized.contains("visible"));
    }
}
