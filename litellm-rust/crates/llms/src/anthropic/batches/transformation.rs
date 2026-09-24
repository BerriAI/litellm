use litellm_types::llms::anthropic_messages::anthropic_response::AnthropicMessagesResponse;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use time::OffsetDateTime;
use url::Url;

use crate::{
    anthropic::{
        common_utils::is_anthropic_oauth_key,
        experimental_pass_through::messages::transformation::resolve_anthropic_api_base,
    },
    base_llm::{anthropic_messages::transformation::Headers, chat::transformation::Error},
};

const BATCHES_PATH_SUFFIX: &str = "/v1/messages/batches";
const BATCHES_BETA: &str = "message-batches-2024-09-24";
const BETA_HEADER: &str = "anthropic-beta";
const ANTHROPIC_API_KEY_ENV: &str = "ANTHROPIC_API_KEY";
const ANTHROPIC_AUTH_TOKEN_ENV: &str = "ANTHROPIC_AUTH_TOKEN";

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
    #[serde(default)]
    pub processing_status: String,
    pub created_at: Option<String>,
    pub ended_at: Option<String>,
    pub expires_at: Option<String>,
    pub cancel_initiated_at: Option<String>,
    pub archived_at: Option<String>,
    #[serde(default)]
    pub request_counts: AnthropicBatchRequestCounts,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchStatus {
    InProgress,
    Cancelling,
    Completed,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchRequestCounts {
    pub total: u64,
    pub completed: u64,
    pub failed: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LiteLlmMessageBatch {
    pub id: String,
    pub object: String,
    pub endpoint: String,
    pub input_file_id: String,
    pub completion_window: String,
    pub status: BatchStatus,
    pub output_file_id: String,
    pub created_at: i64,
    pub in_progress_at: Option<i64>,
    pub expires_at: Option<i64>,
    pub completed_at: Option<i64>,
    pub expired_at: Option<i64>,
    pub cancelling_at: Option<i64>,
    pub cancelled_at: Option<i64>,
    pub request_counts: BatchRequestCounts,
}

pub trait AnthropicBatchesConfig {
    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<Headers, Error>;

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
    ) -> Result<LiteLlmMessageBatch, Error>;

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
    ) -> LiteLlmMessageBatch;

    fn transform_batch_results(&self, body: &str) -> Result<Vec<AnthropicMessagesResponse>, Error>;
}

pub struct AnthropicBatchesTransformation;

pub const ANTHROPIC_BATCHES_TRANSFORMATION: AnthropicBatchesTransformation =
    AnthropicBatchesTransformation;

fn timestamp(value: Option<&str>) -> Option<i64> {
    value
        .and_then(|value| {
            OffsetDateTime::parse(value, &time::format_description::well_known::Rfc3339).ok()
        })
        .map(OffsetDateTime::unix_timestamp)
}

fn auth_header(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<(String, String)> {
    let env = |name: &str| env_lookup(name).filter(|value| !value.is_empty());
    let api_key = api_key
        .filter(|key| !key.is_empty())
        .map(str::to_string)
        .or_else(|| env(ANTHROPIC_API_KEY_ENV));
    match api_key {
        Some(key) if is_anthropic_oauth_key(&key) => {
            Some(("authorization".into(), format!("Bearer {key}")))
        }
        Some(key) => Some(("x-api-key".into(), key)),
        None => env(ANTHROPIC_AUTH_TOKEN_ENV)
            .map(|token| ("authorization".into(), format!("Bearer {token}"))),
    }
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
    Url::parse(&complete_url)
        .map_err(|error| Error::InvalidRequest(format!("invalid Anthropic API base: {error}")))
}

impl AnthropicBatchesConfig for AnthropicBatchesTransformation {
    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<Headers, Error> {
        let auth = auth_header(api_key, env_lookup).ok_or(litellm_auth::Error::MissingApiKey {
            provider: "Anthropic",
            environment_variable: ANTHROPIC_API_KEY_ENV,
        })?;
        let fixed = [
            ("accept", "application/json"),
            ("anthropic-version", "2023-06-01"),
            ("content-type", "application/json"),
        ]
        .map(|(name, value)| (name.to_string(), value.to_string()))
        .into_iter()
        .chain([auth])
        .collect::<Vec<_>>();
        let has_beta = headers
            .iter()
            .any(|(name, _)| name.eq_ignore_ascii_case(BETA_HEADER));
        let default_beta = (!has_beta).then(|| (BETA_HEADER.to_string(), BATCHES_BETA.to_string()));
        let caller = headers
            .into_iter()
            .filter(|(name, _)| {
                !fixed
                    .iter()
                    .any(|(fixed_name, _)| fixed_name.eq_ignore_ascii_case(name))
            })
            .collect::<Vec<_>>();
        Ok(caller
            .into_iter()
            .chain(default_beta)
            .chain(fixed)
            .collect())
    }

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
    ) -> Result<LiteLlmMessageBatch, Error> {
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
        if matches!(batch_id, "." | "..") {
            return Err(Error::InvalidRequest(
                "batch_id cannot be a dot path segment".into(),
            ));
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
    ) -> LiteLlmMessageBatch {
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

        LiteLlmMessageBatch {
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

    fn transform_batch_results(&self, body: &str) -> Result<Vec<AnthropicMessagesResponse>, Error> {
        body.lines()
            .filter(|line| !line.trim().is_empty())
            .filter_map(|line| serde_json::from_str::<Value>(line.trim()).ok())
            .map(|record| {
                serde_json::from_value(record["result"]["message"].clone()).map_err(|error| {
                    Error::InvalidResponse(format!("invalid Anthropic batch result: {error}"))
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    const NOW: i64 = 7;
    const SEP_24_10_00: i64 = 1_727_172_000;

    type Env = &'static [(&'static str, &'static str)];

    fn lookup(env: Env) -> impl Fn(&str) -> Option<String> {
        move |name| {
            env.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        }
    }

    fn headers(pairs: &[(&str, &str)]) -> Headers {
        pairs
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect()
    }

    fn retrieve(body: Value) -> LiteLlmMessageBatch {
        ANTHROPIC_BATCHES_TRANSFORMATION
            .transform_retrieve_batch_response(serde_json::from_value(body).unwrap(), NOW)
    }

    fn batch(id: &str, status: BatchStatus, created_at: i64) -> LiteLlmMessageBatch {
        LiteLlmMessageBatch {
            id: id.into(),
            object: "batch".into(),
            endpoint: "/v1/messages".into(),
            input_file_id: "None".into(),
            completion_window: "24h".into(),
            status,
            output_file_id: id.into(),
            created_at,
            in_progress_at: None,
            expires_at: None,
            completed_at: None,
            expired_at: None,
            cancelling_at: None,
            cancelled_at: None,
            request_counts: BatchRequestCounts {
                total: 0,
                completed: 0,
                failed: 0,
            },
        }
    }

    fn counts(total: u64, completed: u64, failed: u64) -> BatchRequestCounts {
        BatchRequestCounts {
            total,
            completed,
            failed,
        }
    }

    const JSON_HEADERS: [(&str, &str); 3] = [
        ("accept", "application/json"),
        ("anthropic-version", "2023-06-01"),
        ("content-type", "application/json"),
    ];

    #[rstest]
    #[case::api_key_param(&[], Some("sk-ant-test"), &[], &[("x-api-key", "sk-ant-test")])]
    #[case::oauth_key_param_uses_bearer(
        &[],
        Some("sk-ant-oat-abc123"),
        &[],
        &[("authorization", "Bearer sk-ant-oat-abc123")]
    )]
    #[case::empty_key_param_falls_back_to_env(
        &[],
        Some(""),
        &[("ANTHROPIC_API_KEY", "sk-env")],
        &[("x-api-key", "sk-env")]
    )]
    #[case::env_oauth_key_uses_bearer(
        &[],
        None,
        &[("ANTHROPIC_API_KEY", "sk-ant-oat01-env")],
        &[("authorization", "Bearer sk-ant-oat01-env")]
    )]
    #[case::key_param_beats_auth_token(
        &[],
        Some("sk-param"),
        &[("ANTHROPIC_AUTH_TOKEN", "token")],
        &[("x-api-key", "sk-param")]
    )]
    #[case::auth_token_when_no_key(
        &[],
        None,
        &[("ANTHROPIC_AUTH_TOKEN", "token")],
        &[("authorization", "Bearer token")]
    )]
    #[case::caller_json_headers_are_overridden(
        &[("Content-Type", "text/plain"), ("x-trace", "1")],
        Some("sk"),
        &[],
        &[("x-trace", "1"), ("x-api-key", "sk")]
    )]
    fn validate_environment_adds_json_headers_auth_and_default_beta(
        #[case] caller: &[(&str, &str)],
        #[case] api_key: Option<&str>,
        #[case] env: Env,
        #[case] expected_extra: &[(&str, &str)],
    ) {
        let actual = ANTHROPIC_BATCHES_TRANSFORMATION
            .validate_environment(headers(caller), api_key, &lookup(env))
            .unwrap();

        let (caller_kept, auth) = expected_extra.split_at(expected_extra.len() - 1);
        let expected = headers(
            &[
                caller_kept,
                &[("anthropic-beta", BATCHES_BETA)],
                &JSON_HEADERS,
                auth,
            ]
            .concat(),
        );
        assert_eq!(actual, expected);
    }

    #[rstest]
    #[case::lowercase("anthropic-beta")]
    #[case::mixed_case("Anthropic-Beta")]
    fn validate_environment_keeps_caller_beta(#[case] name: &str) {
        let actual = ANTHROPIC_BATCHES_TRANSFORMATION
            .validate_environment(headers(&[(name, "custom-beta-value")]), Some("sk"), &|_| {
                None
            })
            .unwrap();

        let betas = actual
            .iter()
            .filter(|(header, _)| header.eq_ignore_ascii_case(BETA_HEADER))
            .map(|(_, value)| value.as_str())
            .collect::<Vec<_>>();
        assert_eq!(betas, ["custom-beta-value"]);
    }

    #[rstest]
    #[case::nothing_set(None, &[])]
    #[case::blank_everything(Some(""), &[("ANTHROPIC_API_KEY", ""), ("ANTHROPIC_AUTH_TOKEN", "")])]
    fn validate_environment_without_credentials_is_missing_api_key(
        #[case] api_key: Option<&str>,
        #[case] env: Env,
    ) {
        let error = ANTHROPIC_BATCHES_TRANSFORMATION
            .validate_environment(Vec::new(), api_key, &lookup(env))
            .unwrap_err();

        assert!(matches!(
            error,
            Error::Auth(litellm_auth::Error::MissingApiKey {
                provider: "Anthropic",
                environment_variable: "ANTHROPIC_API_KEY",
            })
        ));
    }

    #[rstest]
    #[case::appends_path(Some("https://api.anthropic.com"), &[], "https://api.anthropic.com/v1/messages/batches")]
    #[case::strips_trailing_slash(Some("https://api.anthropic.com/"), &[], "https://api.anthropic.com/v1/messages/batches")]
    #[case::already_complete(Some("https://proxy.internal/v1/messages/batches"), &[], "https://proxy.internal/v1/messages/batches")]
    #[case::messages_endpoint_base(Some("https://proxy.internal/v1/messages"), &[], "https://proxy.internal/v1/messages/batches")]
    #[case::default_base(None, &[], "https://api.anthropic.com/v1/messages/batches")]
    #[case::env_base(None, &[("ANTHROPIC_API_BASE", "https://env.test")], "https://env.test/v1/messages/batches")]
    fn create_batch_url_points_at_batches_collection(
        #[case] api_base: Option<&str>,
        #[case] env: Env,
        #[case] expected: &str,
    ) {
        assert_eq!(
            ANTHROPIC_BATCHES_TRANSFORMATION
                .create_batch_url(api_base, &lookup(env))
                .unwrap(),
            expected
        );
    }

    #[rstest]
    #[case::happy_path(
        Some("https://api.anthropic.com"),
        "msgbatch_123",
        "https://api.anthropic.com/v1/messages/batches/msgbatch_123"
    )]
    #[case::strips_trailing_slash(
        Some("https://api.anthropic.com/"),
        "msgbatch_123",
        "https://api.anthropic.com/v1/messages/batches/msgbatch_123"
    )]
    #[case::complete_base(
        Some("https://proxy.test/v1/messages/batches"),
        "msgbatch_123",
        "https://proxy.test/v1/messages/batches/msgbatch_123"
    )]
    #[case::default_base(
        None,
        "msgbatch_123",
        "https://api.anthropic.com/v1/messages/batches/msgbatch_123"
    )]
    #[case::encodes_slash_and_space(
        Some("https://api.anthropic.com"),
        "a/b id",
        "https://api.anthropic.com/v1/messages/batches/a%2Fb%20id"
    )]
    #[case::encodes_query(
        Some("https://proxy.test"),
        "batch/id ?",
        "https://proxy.test/v1/messages/batches/batch%2Fid%20%3F"
    )]
    #[case::dots_inside_id_are_kept(
        Some("https://proxy.test"),
        "a..b",
        "https://proxy.test/v1/messages/batches/a..b"
    )]
    fn retrieve_batch_url_appends_one_encoded_segment(
        #[case] api_base: Option<&str>,
        #[case] batch_id: &str,
        #[case] expected: &str,
    ) {
        assert_eq!(
            ANTHROPIC_BATCHES_TRANSFORMATION
                .retrieve_batch_url(api_base, batch_id, &|_| None)
                .unwrap(),
            expected
        );
    }

    #[rstest]
    #[case::empty("", "missing required field: batch_id")]
    #[case::dot(".", "invalid request: batch_id cannot be a dot path segment")]
    #[case::dot_dot("..", "invalid request: batch_id cannot be a dot path segment")]
    fn retrieve_batch_url_rejects_ids_that_leave_the_batch_path(
        #[case] batch_id: &str,
        #[case] message: &str,
    ) {
        let error = ANTHROPIC_BATCHES_TRANSFORMATION
            .retrieve_batch_url(Some("https://api.anthropic.com"), batch_id, &|_| None)
            .unwrap_err();

        assert_eq!(error.to_string(), message);
    }

    #[test]
    fn retrieve_batch_request_has_no_body_params() {
        assert_eq!(
            ANTHROPIC_BATCHES_TRANSFORMATION.transform_retrieve_batch_request(),
            json!({})
        );
    }

    #[test]
    fn batch_creation_is_unsupported() {
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

    #[rstest]
    #[case::in_progress(
        json!({
            "id": "msgbatch_abc",
            "type": "message_batch",
            "processing_status": "in_progress",
            "created_at": "2024-09-24T10:00:00Z",
            "expires_at": "2024-09-25T10:00:00Z",
            "ended_at": null,
            "cancel_initiated_at": null,
            "archived_at": null,
            "results_url": null,
            "request_counts": {"processing": 3, "succeeded": 2, "errored": 1, "canceled": 0, "expired": 0}
        }),
        LiteLlmMessageBatch {
            in_progress_at: Some(SEP_24_10_00),
            expires_at: Some(SEP_24_10_00 + 86_400),
            request_counts: counts(6, 2, 1),
            ..batch("msgbatch_abc", BatchStatus::InProgress, SEP_24_10_00)
        }
    )]
    #[case::ended_maps_to_completed(
        json!({
            "id": "msgbatch_done",
            "processing_status": "ended",
            "created_at": "2024-09-24T10:00:00Z",
            "ended_at": "2024-09-24T11:00:00Z",
            "request_counts": {"succeeded": 5, "errored": 0}
        }),
        LiteLlmMessageBatch {
            completed_at: Some(SEP_24_10_00 + 3_600),
            request_counts: counts(5, 5, 0),
            ..batch("msgbatch_done", BatchStatus::Completed, SEP_24_10_00)
        }
    )]
    #[case::canceling_maps_to_cancelling(
        json!({
            "id": "msgbatch_cancel",
            "processing_status": "canceling",
            "created_at": "2024-09-24T10:00:00Z",
            "cancel_initiated_at": "2024-09-24T10:30:00Z",
            "ended_at": "2024-09-24T10:45:00Z",
            "request_counts": {"processing": 0, "succeeded": 5, "errored": 0, "canceled": 3, "expired": 0}
        }),
        LiteLlmMessageBatch {
            cancelling_at: Some(SEP_24_10_00 + 1_800),
            cancelled_at: Some(SEP_24_10_00 + 2_700),
            request_counts: counts(8, 5, 0),
            ..batch("msgbatch_cancel", BatchStatus::Cancelling, SEP_24_10_00)
        }
    )]
    #[case::canceling_without_end(
        json!({
            "id": "msgbatch_cancel",
            "processing_status": "canceling",
            "created_at": "2024-09-24T10:00:00Z",
            "cancel_initiated_at": "2024-09-24T10:30:00Z"
        }),
        LiteLlmMessageBatch {
            cancelling_at: Some(SEP_24_10_00 + 1_800),
            ..batch("msgbatch_cancel", BatchStatus::Cancelling, SEP_24_10_00)
        }
    )]
    #[case::unknown_status_defaults_to_in_progress(
        json!({
            "id": "msgbatch_x",
            "processing_status": "some_future_status",
            "created_at": "2024-09-24T10:00:00Z",
            "ended_at": "2024-09-24T11:00:00Z",
            "request_counts": {}
        }),
        batch("msgbatch_x", BatchStatus::InProgress, SEP_24_10_00)
    )]
    #[case::empty_body_defaults(json!({}), batch("", BatchStatus::InProgress, NOW))]
    #[case::archived_sets_expired_at(
        json!({
            "id": "msgbatch_arch",
            "processing_status": "ended",
            "created_at": "2024-09-24T10:00:00Z",
            "ended_at": "2024-09-24T11:00:00Z",
            "archived_at": "2024-09-26T10:00:00Z",
            "request_counts": {}
        }),
        LiteLlmMessageBatch {
            completed_at: Some(SEP_24_10_00 + 3_600),
            expired_at: Some(SEP_24_10_00 + 2 * 86_400),
            ..batch("msgbatch_arch", BatchStatus::Completed, SEP_24_10_00)
        }
    )]
    #[case::bad_timestamps_are_dropped(
        json!({
            "id": "msgbatch_bad",
            "processing_status": "in_progress",
            "created_at": "not-a-real-timestamp",
            "expires_at": "",
            "request_counts": {"processing": 1, "succeeded": 2, "errored": 3, "canceled": 4, "expired": 5}
        }),
        LiteLlmMessageBatch {
            request_counts: counts(15, 2, 3),
            ..batch("msgbatch_bad", BatchStatus::InProgress, NOW)
        }
    )]
    #[case::offset_timestamp(
        json!({
            "id": "msgbatch_tz",
            "processing_status": "ended",
            "created_at": "2024-09-24T19:00:00+09:00",
            "ended_at": "2024-09-24T10:00:00.123456Z"
        }),
        LiteLlmMessageBatch {
            completed_at: Some(SEP_24_10_00),
            ..batch("msgbatch_tz", BatchStatus::Completed, SEP_24_10_00)
        }
    )]
    fn retrieve_batch_response_maps_to_openai_batch(
        #[case] body: Value,
        #[case] expected: LiteLlmMessageBatch,
    ) {
        assert_eq!(retrieve(body), expected);
    }

    fn result_line(id: &str, input_tokens: u64) -> String {
        json!({
            "custom_id": id,
            "result": {
                "type": "succeeded",
                "message": {
                    "id": id,
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-test",
                    "content": [{"type": "text", "text": "a"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": null,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 5}
                }
            }
        })
        .to_string()
    }

    #[rstest]
    #[case::two_lines(format!("{}\n{}\n", result_line("msg_1", 10), result_line("msg_2", 20)), &[("msg_1", 10), ("msg_2", 20)])]
    #[case::skips_blank_and_non_json(format!("not-json\n\n  \n{}\n", result_line("msg_1", 7)), &[("msg_1", 7)])]
    #[case::trims_padded_lines(format!("  {}  \r\n", result_line("msg_1", 3)), &[("msg_1", 3)])]
    #[case::empty_body(String::new(), &[])]
    fn batch_results_yield_each_message_with_its_usage(
        #[case] body: String,
        #[case] expected: &[(&str, u64)],
    ) {
        let messages = ANTHROPIC_BATCHES_TRANSFORMATION
            .transform_batch_results(&body)
            .unwrap();

        let actual = messages
            .iter()
            .map(|message| {
                (
                    message.id.as_str(),
                    message.usage.as_ref().unwrap()["input_tokens"]
                        .as_u64()
                        .unwrap(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[rstest]
    #[case::errored_result(json!({"custom_id": "a", "result": {"type": "errored", "error": {"type": "invalid_request_error"}}}))]
    #[case::malformed_message(json!({"custom_id": "a", "result": {"type": "succeeded", "message": {"id": "msg_1"}}}))]
    fn batch_results_reject_json_lines_without_a_message(#[case] line: Value) {
        assert!(matches!(
            ANTHROPIC_BATCHES_TRANSFORMATION.transform_batch_results(&line.to_string()),
            Err(Error::InvalidResponse(_))
        ));
    }
}
