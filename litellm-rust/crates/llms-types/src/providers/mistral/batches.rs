use serde_json::{Map, Value};

use crate::recognized::Recognized;

#[macro_rules_attribute::apply(wire_type)]
#[derive(Copy, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum MistralBatchStatus {
    Queued,
    Running,
    Success,
    Failed,
    TimeoutExceeded,
    CancellationRequested,
    Cancelled,
}

#[macro_rules_attribute::apply(wire_type)]
#[derive(Copy, Eq)]
pub enum MistralBatchEndpoint {
    #[serde(rename = "/v1/chat/completions")]
    ChatCompletions,
    #[serde(rename = "/v1/embeddings")]
    Embeddings,
    #[serde(rename = "/v1/fim/completions")]
    FimCompletions,
    #[serde(rename = "/v1/moderations")]
    Moderations,
    #[serde(rename = "/v1/chat/moderations")]
    ChatModerations,
    #[serde(rename = "/v1/ocr")]
    Ocr,
    #[serde(rename = "/v1/classifications")]
    Classifications,
    #[serde(rename = "/v1/chat/classifications")]
    ChatClassifications,
    #[serde(rename = "/v1/conversations")]
    Conversations,
    #[serde(rename = "/v1/audio/transcriptions")]
    AudioTranscriptions,
}

#[macro_rules_attribute::apply(wire_type)]
pub struct MistralBatchRequest {
    pub body: Map<String, Value>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub custom_id: Option<Option<String>>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
pub struct MistralCreateBatchJobRequest {
    pub endpoint: Recognized<MistralBatchEndpoint>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub input_files: Option<Option<Vec<String>>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub requests: Option<Option<Vec<MistralBatchRequest>>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub model: Option<Option<String>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub agent_id: Option<Option<String>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub metadata: Option<Option<std::collections::BTreeMap<String, String>>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub timeout_hours: Option<Option<u64>>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
pub struct MistralBatchError {
    pub message: String,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub count: Option<Option<u64>>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
pub struct MistralBatchJob {
    pub id: String,
    pub input_files: Vec<String>,
    pub endpoint: Recognized<MistralBatchEndpoint>,
    pub status: Recognized<MistralBatchStatus>,
    pub created_at: i64,
    pub total_requests: u64,
    pub completed_requests: u64,
    pub succeeded_requests: u64,
    pub failed_requests: u64,
    pub errors: Vec<MistralBatchError>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub object: Option<Option<String>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub model: Option<Option<String>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub agent_id: Option<Option<String>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub output_file: Option<Option<String>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub error_file: Option<Option<String>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub outputs: Option<Option<Vec<Map<String, Value>>>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub started_at: Option<Option<i64>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub completed_at: Option<Option<i64>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub metadata: Option<Option<Map<String, Value>>>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[macro_rules_attribute::apply(wire_type)]
pub struct MistralBatchJobList {
    pub total: u64,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub object: Option<Option<String>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub data: Option<Option<Vec<MistralBatchJob>>>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::{
        MistralBatchEndpoint, MistralBatchJob, MistralBatchStatus, MistralCreateBatchJobRequest,
    };
    use crate::recognized::Recognized;
    use rstest::rstest;
    use serde_json::{Value, json};

    #[rstest]
    #[case::files(json!({"input_files":["file-id"]}))]
    #[case::inline(json!({"requests":[{"custom_id":"request-id","body":{"messages":[{"role":"user","content":"text"}]}}]}))]
    fn creation_preserves_input_sources_and_metadata(#[case] source: Value) {
        let wire = Value::Object(
            json!({"model":"model", "endpoint":"/v1/chat/completions", "metadata":{"tag":"value"}})
                .as_object()
                .unwrap()
                .iter()
                .chain(source.as_object().unwrap())
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        );
        let request: MistralCreateBatchJobRequest = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(
            request.endpoint,
            Recognized::Known(MistralBatchEndpoint::ChatCompletions)
        );
        assert_eq!(serde_json::to_value(request).unwrap(), wire);
    }

    #[rstest]
    #[case::queued(MistralBatchStatus::Queued)]
    #[case::running(MistralBatchStatus::Running)]
    #[case::success(MistralBatchStatus::Success)]
    #[case::failed(MistralBatchStatus::Failed)]
    #[case::timeout(MistralBatchStatus::TimeoutExceeded)]
    #[case::cancellation_requested(MistralBatchStatus::CancellationRequested)]
    #[case::cancelled(MistralBatchStatus::Cancelled)]
    fn native_status_and_counters_are_not_normalized(#[case] status: MistralBatchStatus) {
        let wire = json!({
            "id":"job-id", "input_files":["file-id"], "endpoint":"/v1/ocr", "status":status,
            "created_at":1, "total_requests":5, "completed_requests":3, "succeeded_requests":2,
            "failed_requests":1, "errors":[{"message":"failed", "count":1}], "future_field":null
        });
        let job: MistralBatchJob = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(job.status, Recognized::Known(status));
        assert_eq!(job.completed_requests, 3);
        assert_eq!(job.succeeded_requests, 2);
        let encoded = serde_json::to_value(job).unwrap();
        for (key, value) in wire.as_object().unwrap() {
            assert_eq!(&encoded[key], value);
        }
    }

    #[rstest]
    fn future_status_and_endpoint_are_preserved() {
        let wire = json!({
            "id":"job-id", "input_files":[], "endpoint":"/v1/future", "status":"FUTURE",
            "created_at":1, "total_requests":0, "completed_requests":0, "succeeded_requests":0,
            "failed_requests":0, "errors":[]
        });
        let job: MistralBatchJob = serde_json::from_value(wire.clone()).unwrap();
        assert!(job.status.known().is_none());
        assert!(job.endpoint.known().is_none());
        let encoded = serde_json::to_value(job).unwrap();
        assert_eq!(encoded["status"], wire["status"]);
        assert_eq!(encoded["endpoint"], wire["endpoint"]);
    }

    #[rstest]
    fn creation_rejects_malformed_inline_body() {
        assert!(
            serde_json::from_value::<MistralCreateBatchJobRequest>(json!({
                "endpoint":"/v1/chat/completions", "requests":[{"body":false}]
            }))
            .is_err()
        );
    }
}
