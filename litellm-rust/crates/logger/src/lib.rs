use std::{
    cell::Cell,
    fmt,
    future::{Future, poll_fn},
    pin::pin,
};

use serde_json::{Map, Value};
use tracing::{
    Dispatch, Event, Subscriber,
    field::{Field, Visit},
    subscriber::Interest,
};
use tracing_subscriber::{Layer, Registry, layer::Context, prelude::*};

mod processing;
mod redaction;

pub use processing::{DiagnosticInput, DiagnosticOutput, Policy, Processor};
pub use redaction::{REDACTED, SecretRedactor};
pub use tracing::{Level, Metadata, debug, error, info, trace, warn};

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
            dispatch: Dispatch::new(Registry::default().with(Output(sink))),
        }
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

struct Output<S>(S);

impl<S: Sink, R: Subscriber> Layer<R> for Output<S> {
    fn register_callsite(&self, _: &'static Metadata<'static>) -> Interest {
        Interest::sometimes()
    }

    fn enabled(&self, metadata: &Metadata<'_>, _: Context<'_, R>) -> bool {
        let Some(_guard) = Emitting::enter() else {
            return false;
        };
        self.0.enabled(metadata)
    }

    fn on_event(&self, event: &Event<'_>, _: Context<'_, R>) {
        let Some(_guard) = Emitting::enter() else {
            return;
        };
        let mut record = Record {
            metadata: event.metadata(),
            message: String::new(),
            fields: Map::new(),
        };
        event.record(&mut record);
        self.0.emit(&record);
    }
}

impl Record {
    fn field(&mut self, field: &Field, value: Value) {
        if field.name() == "message" {
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
