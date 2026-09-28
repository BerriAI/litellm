use std::{
    cell::Cell,
    fmt,
    future::{Future, poll_fn},
    pin::pin,
};

use base64::{Engine, engine::general_purpose::STANDARD};
use serde_json::{Map, Value};
use tracing::{
    Dispatch,
    field::{Field, Visit},
};

mod layer;
mod processing;
mod redaction;

pub use layer::sink_layer;
pub use processing::{DiagnosticInput, DiagnosticOutput, Policy, Processor};
pub use redaction::{REDACTED, SecretRedactor};
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

#[derive(Debug)]
pub struct Record {
    pub metadata: &'static Metadata<'static>,
    pub message: String,
    pub fields: Map<String, Value>,
}

#[derive(Clone, Default)]
pub struct Logger {
    dispatch: Dispatch,
}

impl Logger {
    pub fn new(sink: impl Sink) -> Self {
        Self {
            dispatch: layer::dispatch(sink),
        }
    }

    pub fn current() -> Self {
        Self {
            dispatch: tracing::dispatcher::get_default(Clone::clone),
        }
    }

    pub fn install_global(&self) -> Result<(), tracing::dispatcher::SetGlobalDefaultError> {
        tracing::dispatcher::set_global_default(self.dispatch.clone())
    }

    pub fn scope<T>(&self, operation: impl FnOnce() -> T) -> T {
        if EMITTING.get() {
            return operation();
        }
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
}

struct Emitting;

impl Emitting {
    fn enter() -> Option<Self> {
        EMITTING.with(|active| (!active.replace(true)).then_some(Self))
    }
}

impl Drop for Emitting {
    fn drop(&mut self) {
        EMITTING.set(false);
    }
}

impl Record {
    fn field(&mut self, field: &Field, value: Value) {
        if field.name() == "message" && self.metadata.is_event() {
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
