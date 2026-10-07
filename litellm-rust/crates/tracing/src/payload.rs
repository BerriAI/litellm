use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::{PayloadShape, ShapeLimits};

pub const TARGET: &str = "litellm_payload_shapes";

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub enum PayloadStage {
    #[serde(rename = "litellm.request.received")]
    RequestReceived,
    #[serde(rename = "provider.request.transformed")]
    RequestTransformed,
    #[serde(rename = "provider.request.sent")]
    RequestSent,
    #[serde(rename = "provider.response.received")]
    ResponseReceived,
    #[serde(rename = "litellm.response.normalized")]
    ResponseNormalized,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PayloadEvent {
    pub stage: PayloadStage,
    pub shape: PayloadShape,
    pub capture_id: String,
    pub outcome: Option<PayloadOutcome>,
}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PayloadOutcome {
    Success,
    Failure,
    Cancelled,
}

pub fn enabled() -> bool {
    tracing::enabled!(target: "litellm_payload_shapes", tracing::Level::INFO)
}

impl PayloadEvent {
    pub fn emit(&self) {
        let limits = ShapeLimits::default();
        if !enabled()
            || self.capture_id.is_empty()
            || self.capture_id.len() > 128
            || self.shape.field_paths.len() > limits.paths
            || self
                .shape
                .field_paths
                .iter()
                .map(String::len)
                .sum::<usize>()
                > limits.bytes
        {
            return;
        }
        let fields = json!({
            "event.name": "llm.payload.shape",
            "schema_version": 1,
            "trace_id": self.capture_id,
            "payload.stage": self.stage,
            "payload.field_paths": self.shape.field_paths,
            "payload.shape_truncated": self.shape.truncated,
            "payload.outcome": self.outcome,
        });
        tracing::info!(target: "litellm_payload_shapes", diagnostic_fields = %fields, "payload shape");
    }
}

pub(crate) fn project(record: &mut crate::Record) {
    if record.metadata.target() != TARGET {
        return;
    }
    record.message = "payload shape".to_owned();
    record.fields.retain(|key, _| {
        matches!(
            key.as_str(),
            "event.name"
                | "schema_version"
                | "trace_id"
                | "payload.stage"
                | "payload.field_paths"
                | "payload.shape_truncated"
                | "payload.outcome"
        )
    });
}

pub async fn capture<F: std::future::Future>(future: F) -> F::Output {
    if !enabled() {
        return future.await;
    }
    let mut fields = crate::CONTEXT.with(|fields| fields.borrow().as_ref().clone());
    let id = fields
        .get("_payload_capture_id")
        .or_else(|| fields.get("trace_id"))
        .and_then(serde_json::Value::as_str)
        .filter(|id| !id.is_empty() && id.len() <= 128)
        .map(str::to_owned)
        .unwrap_or_else(|| format!("{:032x}", rand::random::<u128>()));
    fields.insert("_payload_capture_id".to_owned(), id.into());
    crate::Logger::current()
        .with_fields(fields)
        .instrument(future)
        .await
}

pub fn record(stage: PayloadStage, value: &serde_json::Value) {
    if !enabled() {
        return;
    }
    let id = crate::CONTEXT.with(|fields| {
        fields
            .borrow()
            .get("_payload_capture_id")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
    });
    if let Some(capture_id) = id {
        PayloadEvent {
            stage,
            shape: PayloadShape::extract(value, ShapeLimits::default()),
            capture_id,
            outcome: None,
        }
        .emit();
    }
}

pub fn record_serialized(stage: PayloadStage, value: &impl Serialize) {
    if enabled()
        && let Ok(value) = serde_json::to_value(value)
    {
        record(stage, &value);
    }
}
