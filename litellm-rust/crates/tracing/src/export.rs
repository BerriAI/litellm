use std::{
    collections::HashMap,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use opentelemetry::logs::{AnyValue, LogRecord, Logger as _, LoggerProvider as _, Severity};
use opentelemetry_otlp::{WithExportConfig, WithHttpConfig};
use opentelemetry_sdk::{
    Resource,
    logs::{SdkLogger, SdkLoggerProvider},
};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::{Error, ExportSink, Level, Metadata, Processor, Record, Sink};

pub struct ExportPolicy {
    minimum_level: Level,
    target_prefixes: Vec<String>,
    sample_rate: f64,
}

pub(crate) fn source_timestamp(record: &Record) -> Option<SystemTime> {
    let created = record.fields.get("source.timestamp")?.as_f64()?;
    UNIX_EPOCH.checked_add(Duration::try_from_secs_f64(created).ok()?)
}

impl ExportPolicy {
    pub(crate) fn minimum_level_accepts(&self, metadata: &Metadata<'_>) -> bool {
        metadata.is_span() || metadata.level() <= &self.minimum_level
    }
    pub fn new(
        minimum_level: Level,
        target_prefixes: Vec<String>,
        sample_rate: f64,
    ) -> Result<Self, Error> {
        if !sample_rate.is_finite() || !(0.0..=1.0).contains(&sample_rate) {
            return Err(Error::InvalidSampleRate);
        }
        Ok(Self {
            minimum_level,
            target_prefixes,
            sample_rate,
        })
    }

    pub(crate) fn accepts(&self, record: &Record) -> bool {
        let target = record
            .fields
            .get("source.target")
            .and_then(Value::as_str)
            .unwrap_or(record.metadata.target());
        if record.metadata.level() > &self.minimum_level
            || (!self.target_prefixes.is_empty()
                && !self
                    .target_prefixes
                    .iter()
                    .any(|prefix| target.starts_with(prefix)))
        {
            return false;
        }
        if record.metadata.level() == &Level::ERROR || self.sample_rate == 1.0 {
            return true;
        }
        let correlation = record
            .fields
            .get("trace_id")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty());
        let fraction = correlation.map_or_else(rand::random::<f64>, |trace| {
            let digest = Sha256::digest(trace.as_bytes());
            let number =
                u64::from_be_bytes(digest[..8].try_into().expect("SHA256 contains eight bytes"));
            (number >> 11) as f64 / (1_u64 << 53) as f64
        });
        fraction < self.sample_rate
    }
}

#[cfg(feature = "posthog")]
pub(crate) fn scrub(
    processor: &Processor,
    key: Option<&str>,
    value: &Value,
) -> fancy_regex::Result<Value> {
    Ok(match value {
        Value::String(value) => processor.redact_structured_text(key, value)?.into(),
        Value::Array(values) => Value::Array(
            values
                .iter()
                .map(|value| scrub(processor, key, value))
                .collect::<fancy_regex::Result<_>>()?,
        ),
        Value::Object(values) => Value::Object(
            values
                .iter()
                .map(|(key, value)| {
                    scrub(processor, Some(key), value).map(|value| (key.clone(), value))
                })
                .collect::<fancy_regex::Result<_>>()?,
        ),
        value => value.clone(),
    })
}

pub struct OtlpSink {
    provider: SdkLoggerProvider,
    logger: SdkLogger,
    policy: ExportPolicy,
    processor: Processor,
}

impl OtlpSink {
    pub fn new(
        endpoint: String,
        headers: HashMap<String, String>,
        service_name: String,
        policy: ExportPolicy,
    ) -> Result<Self, Error> {
        let exporter = opentelemetry_otlp::LogExporter::builder()
            .with_http()
            .with_protocol(opentelemetry_otlp::Protocol::HttpBinary)
            .with_endpoint(endpoint)
            .with_headers(headers)
            .with_timeout(Duration::from_secs(5))
            .build()?;
        let provider = SdkLoggerProvider::builder()
            .with_resource(
                Resource::builder_empty()
                    .with_service_name(service_name)
                    .build(),
            )
            .with_batch_exporter(exporter)
            .build();
        let logger = provider.logger("litellm.diagnostics");
        Ok(Self {
            provider,
            logger,
            policy,
            processor: Processor::new(16),
        })
    }

    fn value(&self, key: Option<&str>, value: &Value) -> fancy_regex::Result<AnyValue> {
        Ok(match value {
            Value::Null => AnyValue::String("null".into()),
            Value::Bool(value) => AnyValue::Boolean(*value),
            Value::Number(value) => value
                .as_i64()
                .map(AnyValue::Int)
                .unwrap_or_else(|| AnyValue::Double(value.as_f64().unwrap_or_default())),
            Value::String(value) => {
                AnyValue::String(self.processor.redact_structured_text(key, value)?.into())
            }
            Value::Array(values) => AnyValue::ListAny(Box::new(
                values
                    .iter()
                    .map(|value| self.value(key, value))
                    .collect::<fancy_regex::Result<_>>()?,
            )),
            Value::Object(values) => AnyValue::Map(Box::new(
                values
                    .iter()
                    .map(|(key, value)| {
                        self.value(Some(key), value)
                            .map(|value| (key.clone().into(), value))
                    })
                    .collect::<fancy_regex::Result<_>>()?,
            )),
        })
    }
}

impl ExportSink for OtlpSink {
    fn force_flush(&self) -> Result<(), Error> {
        self.provider.force_flush().map_err(Error::from)
    }

    fn shutdown(&self) -> Result<(), Error> {
        self.provider
            .shutdown_with_timeout(Duration::from_secs(5))
            .map_err(Error::from)
    }
}

impl Sink for OtlpSink {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        self.policy.minimum_level_accepts(metadata)
    }

    fn emit(&self, record: &Record) {
        if !self.policy.accepts(record) {
            return;
        }
        let Ok(message) = self.processor.redact_text(&record.message) else {
            return;
        };
        let Ok(attributes) = record
            .fields
            .iter()
            .filter(|(_, value)| !value.is_null())
            .map(|(key, value)| {
                self.value(Some(key), value)
                    .map(|value| (key.clone(), value))
            })
            .collect::<fancy_regex::Result<Vec<_>>>()
        else {
            return;
        };
        let mut log = self.logger.create_log_record();
        let (severity, text) = match *record.metadata.level() {
            Level::ERROR => (Severity::Error, "ERROR"),
            Level::WARN => (Severity::Warn, "WARN"),
            Level::INFO => (Severity::Info, "INFO"),
            Level::DEBUG => (Severity::Debug, "DEBUG"),
            Level::TRACE => (Severity::Trace, "TRACE"),
        };
        log.set_severity_number(severity);
        log.set_severity_text(text);
        log.set_body(message.into());
        log.set_target(record.metadata.target().to_owned());
        if let Some(timestamp) = source_timestamp(record) {
            log.set_timestamp(timestamp);
        }
        log.add_attributes(attributes);
        self.logger.emit(log);
    }
}
