use std::{
    cell::{Cell, RefCell},
    fmt,
    future::{Future, poll_fn},
    pin::pin,
    sync::Arc,
};

use base64::{Engine, engine::general_purpose::STANDARD};
use serde_json::{Map, Value};
use tracing::{
    Dispatch,
    field::{Field, Visit},
};

pub mod analytics;
mod configuration;
mod error;
mod export;
mod layer;
#[cfg(feature = "posthog")]
mod posthog;
mod processing;
mod redaction;
mod runtime;
mod shape;

pub use configuration::{DestinationConfig, DiagnosticPolicy, DiagnosticsConfig};
pub use error::Error;
pub use export::{ExportPolicy, OtlpSink};
pub use layer::sink_layer;
#[cfg(feature = "posthog")]
pub use posthog::PostHogSink;
pub use processing::{DiagnosticInput, DiagnosticOutput, Policy, Processor};
pub use redaction::{REDACTED, SecretRedactor};
pub use runtime::Diagnostics;
pub use shape::{PayloadShape, ShapeLimits};
pub use tracing::{Level, Metadata, debug, error, info, trace, warn};

pub struct ByteChunk<'a>(&'a [u8]);

impl<'a> ByteChunk<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self(data)
    }

    pub fn encoding(&self) -> &'static str {
        if std::str::from_utf8(self.0).is_ok() {
            "utf8"
        } else {
            "base64"
        }
    }
}

impl fmt::Display for ByteChunk<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match std::str::from_utf8(self.0) {
            Ok(text) => formatter.write_str(text),
            Err(_) => formatter.write_str(&STANDARD.encode(self.0)),
        }
    }
}

pub trait Sink: Send + Sync + 'static {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool;
    fn emit(&self, record: &Record);
}

impl<S: Sink + ?Sized> Sink for Arc<S> {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        self.as_ref().enabled(metadata)
    }
    fn emit(&self, record: &Record) {
        self.as_ref().emit(record);
    }
}

pub trait ExportSink: Sink {
    fn force_flush(&self) -> Result<(), Error>;
    fn shutdown(&self) -> Result<(), Error>;
}

#[derive(Debug)]
pub struct Record {
    pub metadata: &'static Metadata<'static>,
    pub message: String,
    pub fields: Map<String, Value>,
}

#[derive(Clone, Default)]
pub struct Logger {
    dispatch: Dispatch,
    fields: Arc<Map<String, Value>>,
}

impl Logger {
    pub fn new(sink: impl Sink) -> Self {
        Self {
            dispatch: layer::dispatch(sink),
            fields: Arc::default(),
        }
    }

    pub fn current() -> Self {
        Self {
            dispatch: tracing::dispatcher::get_default(Clone::clone),
            fields: CONTEXT.with(|fields| fields.borrow().clone()),
        }
    }

    pub fn with_fields(&self, fields: Map<String, Value>) -> Self {
        Self {
            dispatch: self.dispatch.clone(),
            fields: Arc::new(fields),
        }
    }

    pub fn emit(&self, level: Level, message: &str, fields: Map<String, Value>) {
        let fields = Value::Object(fields);
        self.scope(|| match level {
            Level::ERROR => {
                tracing::error!(target: "litellm.diagnostics", diagnostic_fields = %fields, message)
            }
            Level::WARN => {
                tracing::warn!(target: "litellm.diagnostics", diagnostic_fields = %fields, message)
            }
            Level::INFO => {
                tracing::info!(target: "litellm.diagnostics", diagnostic_fields = %fields, message)
            }
            Level::DEBUG => {
                tracing::debug!(target: "litellm.diagnostics", diagnostic_fields = %fields, message)
            }
            Level::TRACE => {
                tracing::trace!(target: "litellm.diagnostics", diagnostic_fields = %fields, message)
            }
        });
    }

    pub fn install_global(&self) -> Result<(), tracing::dispatcher::SetGlobalDefaultError> {
        tracing::dispatcher::set_global_default(self.dispatch.clone())
    }

    pub fn scope<T>(&self, operation: impl FnOnce() -> T) -> T {
        if EMITTING.get() {
            return operation();
        }
        let _context = DiagnosticContext::enter(self.fields.clone());
        tracing::dispatcher::with_default(&self.dispatch, operation)
    }

    pub fn instrument<F: Future>(&self, future: F) -> impl Future<Output = F::Output> + use<F> {
        let logger = self.clone();
        async move {
            let mut future = pin!(future);
            poll_fn(|context| logger.scope(|| future.as_mut().poll(context))).await
        }
    }
}

thread_local! {
    static EMITTING: Cell<bool> = const { Cell::new(false) };
    static CONTEXT: RefCell<Arc<Map<String, Value>>> = RefCell::new(Arc::default());
}

struct DiagnosticContext(Arc<Map<String, Value>>);

impl DiagnosticContext {
    fn enter(fields: Arc<Map<String, Value>>) -> Self {
        Self(CONTEXT.with(|current| current.replace(fields)))
    }
}

impl Drop for DiagnosticContext {
    fn drop(&mut self) {
        CONTEXT.with(|fields| fields.replace(self.0.clone()));
    }
}

struct Emitting;

impl Emitting {
    fn enter() -> Option<Self> {
        EMITTING.with(|active| {
            if active.replace(true) {
                None
            } else {
                Some(Self)
            }
        })
    }
}

impl Drop for Emitting {
    fn drop(&mut self) {
        EMITTING.set(false);
    }
}

impl Record {
    fn field(&mut self, field: &Field, value: Value) {
        if field.name() == "diagnostic_fields"
            && let Some(text) = value.as_str()
            && let Ok(Value::Object(fields)) = serde_json::from_str(text)
        {
            self.fields.extend(fields);
        } else if field.name() == "message" && self.metadata.is_event() {
            self.message = match value {
                Value::String(message) => message,
                value => value.to_string(),
            };
        } else {
            self.fields.insert(field.name().to_owned(), value);
        }
    }
}

impl Visit for Record {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.field(field, format!("{value:?}").into());
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.field(field, value.into());
    }

    fn record_bool(&mut self, field: &Field, value: bool) {
        self.field(field, value.into());
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.field(field, value.into());
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.field(field, value.into());
    }

    fn record_f64(&mut self, field: &Field, value: f64) {
        self.field(field, value.into());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rstest::rstest]
    fn rejected_reentrant_entries_preserve_the_outer_guard() {
        let guard = Emitting::enter().unwrap();
        assert!(Emitting::enter().is_none());
        assert!(EMITTING.get());
        assert!(Emitting::enter().is_none());
        assert!(EMITTING.get());
        drop(guard);
        assert!(!EMITTING.get());
    }
}
