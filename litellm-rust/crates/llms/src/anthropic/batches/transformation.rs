use litellm_llms_types::batches::{BatchRequestCounts, BatchResponse, BatchStatus};
use litellm_llms_types::messages::MessagesResponse;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use time::OffsetDateTime;
use url::Url;

use crate::{Error, anthropic::common_utils::resolve_anthropic_api_base};

const BATCHES_PATH_SUFFIX: &str = "/v1/messages/batches";

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnthropicBatchRequestCounts {
    #[serde(default)]
    pub processing: u64,
    #[serde(default)]
    pub succeeded: u64,
    #[serde(default)]
    pub errored: u64,
    #[serde(default)]
    pub canceled: u64,
    #[serde(default)]
    pub expired: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnthropicMessageBatch {
    #[serde(default)]
    pub id: String,
    #[serde(default = "default_processing_status")]
    pub processing_status: String,
    pub created_at: Option<String>,
    pub ended_at: Option<String>,
    pub expires_at: Option<String>,
    pub cancel_initiated_at: Option<String>,
    pub archived_at: Option<String>,
    #[serde(default)]
    pub request_counts: AnthropicBatchRequestCounts,
}

#[derive(Deserialize)]
struct BatchResultRecord {
    result: BatchResult,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum BatchResult {
    Succeeded { message: Box<MessagesResponse> },
    Errored { error: Value },
}

pub trait AnthropicBatchesConfig {
    fn create_batch_url(
        &self,
        api_base: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error>;

    fn transform_create_batch_request(&self) -> Result<Value, Error>;

    fn transform_create_batch_response(
        &self,
        response: AnthropicMessageBatch,
        now: i64,
    ) -> Result<BatchResponse, Error>;

    fn retrieve_batch_url(
        &self,
        api_base: Option<&str>,
        batch_id: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error>;

    fn transform_retrieve_batch_request(&self) -> Value;

    fn transform_retrieve_batch_response(
        &self,
        response: AnthropicMessageBatch,
        now: i64,
    ) -> BatchResponse;

    fn transform_batch_results(&self, body: &str) -> Result<Vec<MessagesResponse>, Error>;
}

pub struct AnthropicBatchesTransformation;

pub const ANTHROPIC_BATCHES_TRANSFORMATION: AnthropicBatchesTransformation =
    AnthropicBatchesTransformation;

fn default_processing_status() -> String {
    "in_progress".into()
}

fn timestamp(value: Option<&str>) -> Option<i64> {
    value
        .and_then(|value| {
            OffsetDateTime::parse(value, &time::format_description::well_known::Rfc3339).ok()
        })
        .map(OffsetDateTime::unix_timestamp)
}

fn batches_base_url(
    api_base: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<Url, Error> {
    let api_base = resolve_anthropic_api_base(api_base, env_lookup);
    let api_base = api_base.trim_end_matches('/');
    let complete_url = if api_base.ends_with(BATCHES_PATH_SUFFIX) {
        api_base.to_string()
    } else if let Some(base) = api_base.strip_suffix("/v1/messages") {
        format!("{base}{BATCHES_PATH_SUFFIX}")
    } else {
        format!("{api_base}{BATCHES_PATH_SUFFIX}")
    };
    Url::parse(&complete_url).map_err(|error| {
        Error::InvalidRequest(crate::ErrorDetail::invalid("Anthropic API base", error))
    })
}

impl AnthropicBatchesConfig for AnthropicBatchesTransformation {
    fn create_batch_url(
        &self,
        api_base: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        Ok(batches_base_url(api_base, env_lookup)?.into())
    }

    fn transform_create_batch_request(&self) -> Result<Value, Error> {
        Err(Error::Unsupported("Anthropic message batch creation"))
    }

    fn transform_create_batch_response(
        &self,
        _response: AnthropicMessageBatch,
        _now: i64,
    ) -> Result<BatchResponse, Error> {
        Err(Error::Unsupported("Anthropic message batch creation"))
    }

    fn retrieve_batch_url(
        &self,
        api_base: Option<&str>,
        batch_id: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        if batch_id.is_empty() {
            return Err(Error::MissingField("batch_id"));
        }
        let mut url = batches_base_url(api_base, env_lookup)?;
        url.path_segments_mut()
            .map_err(|_| Error::InvalidRequest("Anthropic API base cannot be a base URL".into()))?
            .push(batch_id);
        Ok(url.into())
    }

    fn transform_retrieve_batch_request(&self) -> Value {
        Value::Object(Default::default())
    }

    fn transform_retrieve_batch_response(
        &self,
        response: AnthropicMessageBatch,
        now: i64,
    ) -> BatchResponse {
        let created_at = timestamp(response.created_at.as_deref());
        let ended_at = timestamp(response.ended_at.as_deref());
        let expires_at = timestamp(response.expires_at.as_deref());
        let cancel_initiated_at = timestamp(response.cancel_initiated_at.as_deref());
        let archived_at = timestamp(response.archived_at.as_deref());
        let status = match response.processing_status.as_str() {
            "canceling" => BatchStatus::Cancelling,
            "ended" => BatchStatus::Completed,
            _ => BatchStatus::InProgress,
        };
        let request_counts = BatchRequestCounts {
            total: response.request_counts.processing
                + response.request_counts.succeeded
                + response.request_counts.errored
                + response.request_counts.canceled
                + response.request_counts.expired,
            completed: response.request_counts.succeeded,
            failed: response.request_counts.errored,
        };

        BatchResponse {
            id: response.id.clone(),
            object: "batch".into(),
            endpoint: "/v1/messages".into(),
            input_file_id: "None".into(),
            completion_window: "24h".into(),
            status,
            output_file_id: response.id,
            created_at: created_at.unwrap_or(now),
            in_progress_at: (response.processing_status == "in_progress")
                .then_some(created_at)
                .flatten(),
            expires_at,
            completed_at: (response.processing_status == "ended")
                .then_some(ended_at)
                .flatten(),
            expired_at: archived_at,
            cancelling_at: (response.processing_status == "canceling")
                .then_some(cancel_initiated_at)
                .flatten(),
            cancelled_at: (response.processing_status == "canceling")
                .then_some(ended_at)
                .flatten(),
            request_counts,
        }
    }

    fn transform_batch_results(&self, body: &str) -> Result<Vec<MessagesResponse>, Error> {
        body.lines()
            .filter(|line| !line.trim().is_empty())
            .enumerate()
            .map(|(index, line)| {
                let record: BatchResultRecord =
                    serde_json::from_str(line.trim()).map_err(|error| {
                        Error::InvalidResponse(crate::ErrorDetail::InvalidLine {
                            subject: "Anthropic batch result",
                            line: index + 1,
                            source: crate::ErrorSource::new(error),
                        })
                    })?;
                match record.result {
                    BatchResult::Succeeded { message } => Ok(*message),
                    BatchResult::Errored { error } => {
                        Err(Error::InvalidResponse(crate::ErrorDetail::RemoteFailure {
                            operation: "Anthropic batch request",
                            detail: error,
                        }))
                    }
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn builds_and_encodes_message_batch_urls() {
        assert_eq!(
            ANTHROPIC_BATCHES_TRANSFORMATION
                .create_batch_url(None, &|_| None)
                .unwrap(),
            "https://api.anthropic.com/v1/messages/batches"
        );
        assert_eq!(
            ANTHROPIC_BATCHES_TRANSFORMATION
                .create_batch_url(Some("https://proxy.test/v1/messages/batches"), &|_| None)
                .unwrap(),
            "https://proxy.test/v1/messages/batches"
        );
        assert_eq!(
            ANTHROPIC_BATCHES_TRANSFORMATION
                .retrieve_batch_url(Some("https://proxy.test"), "batch/id ?", &|_| None)
                .unwrap(),
            "https://proxy.test/v1/messages/batches/batch%2Fid%20%3F"
        );
        assert_eq!(
            ANTHROPIC_BATCHES_TRANSFORMATION.transform_retrieve_batch_request(),
            json!({})
        );
    }

    #[test]
    fn maps_retrieved_batch_status_counts_and_timestamps_like_python() {
        let response: AnthropicMessageBatch = serde_json::from_value(json!({
            "id": "msgbatch_1",
            "processing_status": "ended",
            "created_at": "2025-01-01T00:00:00Z",
            "ended_at": "2025-01-01T00:01:00Z",
            "expires_at": "not-a-timestamp",
            "request_counts": {
                "processing": 1,
                "succeeded": 2,
                "errored": 3,
                "canceled": 4,
                "expired": 5
            }
        }))
        .unwrap();

        let batch = ANTHROPIC_BATCHES_TRANSFORMATION.transform_retrieve_batch_response(response, 7);
        assert_eq!(batch.status, BatchStatus::Completed);
        assert_eq!(batch.created_at, 1_735_689_600);
        assert_eq!(batch.completed_at, Some(1_735_689_660));
        assert_eq!(batch.expires_at, None);
        assert_eq!(
            batch.request_counts,
            BatchRequestCounts {
                total: 15,
                completed: 2,
                failed: 3
            }
        );
    }

    #[test]
    fn extracts_successful_message_responses_from_ndjson() {
        let body = r#"{"result":{"type":"succeeded","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":"end_turn","stop_sequence":null}}}
"#;
        let messages = ANTHROPIC_BATCHES_TRANSFORMATION
            .transform_batch_results(body)
            .unwrap();

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].id, "msg_1");
    }

    #[test]
    fn reports_malformed_and_unsuccessful_batch_results() {
        assert!(matches!(
            ANTHROPIC_BATCHES_TRANSFORMATION.transform_batch_results("not-json"),
            Err(Error::InvalidResponse(message)) if message.to_string().contains("line 1")
        ));
        assert!(matches!(
            ANTHROPIC_BATCHES_TRANSFORMATION.transform_batch_results(
                r#"{"result":{"type":"errored","error":{"type":"invalid_request_error"}}}"#
            ),
            Err(Error::InvalidResponse(message)) if message.to_string().contains("request failed")
        ));
    }

    #[test]
    fn preserves_python_placeholder_for_batch_creation() {
        assert!(matches!(
            ANTHROPIC_BATCHES_TRANSFORMATION.transform_create_batch_request(),
            Err(Error::Unsupported("Anthropic message batch creation"))
        ));
        let response: AnthropicMessageBatch = serde_json::from_value(json!({})).unwrap();
        assert!(matches!(
            ANTHROPIC_BATCHES_TRANSFORMATION.transform_create_batch_response(response, 0),
            Err(Error::Unsupported("Anthropic message batch creation"))
        ));
    }
}
