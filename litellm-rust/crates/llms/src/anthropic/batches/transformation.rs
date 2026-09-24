use litellm_types::llms::anthropic_messages::anthropic_response::AnthropicMessagesResponse;
use litellm_types::llms::openai::ChatMessage;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use time::OffsetDateTime;
use url::Url;

use crate::{
    anthropic::{
        chat::transformation::ANTHROPIC_CHAT_COMPLETIONS_CONFIG,
        common_utils::is_anthropic_oauth_key,
        experimental_pass_through::messages::transformation::resolve_anthropic_api_base,
    },
    base_llm::{
        anthropic_messages::transformation::Headers,
        chat::transformation::{BaseConfig, Error},
    },
};

const BATCHES_PATH_SUFFIX: &str = "/v1/messages/batches";
const BATCHES_BETA: &str = "message-batches-2024-09-24";
const BETA_HEADER: &str = "anthropic-beta";
const ANTHROPIC_API_KEY_ENV: &str = "ANTHROPIC_API_KEY";
const ANTHROPIC_AUTH_TOKEN_ENV: &str = "ANTHROPIC_AUTH_TOKEN";
const CHAT_COMPLETIONS_URL: &str = "/v1/chat/completions";
const MODEL_PREFIX: &str = "anthropic/";

#[derive(Deserialize)]
struct BatchInputLine {
    custom_id: String,
    method: String,
    url: String,
    body: Map<String, Value>,
}

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

    fn transform_create_batch_request(
        &self,
        model: Option<&str>,
        input_jsonl: &str,
    ) -> Result<Value, Error>;

    fn transform_create_batch_response(
        &self,
        response: AnthropicMessageBatch,
        now: i64,
    ) -> LiteLlmMessageBatch;

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

fn batch_request(model: Option<&str>, line: &str) -> Result<Value, Error> {
    let line: BatchInputLine = serde_json::from_str(line)
        .map_err(|error| Error::InvalidRequest(format!("invalid batch input line: {error}")))?;
    let invalid = |reason: &str| {
        Error::InvalidRequest(format!("batch request {}: {reason}", line.custom_id))
    };
    if line.method != "POST" || line.url != CHAT_COMPLETIONS_URL {
        return Err(invalid(&format!(
            "{} {} is not supported, only POST {CHAT_COMPLETIONS_URL}",
            line.method, line.url
        )));
    }
    let model = model
        .or_else(|| line.body.get("model").and_then(Value::as_str))
        .ok_or_else(|| invalid("model is required"))?;
    let model = model.strip_prefix(MODEL_PREFIX).unwrap_or(model);
    let messages: Vec<ChatMessage> = line
        .body
        .get("messages")
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .map_err(|error| invalid(&format!("invalid messages: {error}")))?
        .filter(|messages: &Vec<ChatMessage>| !messages.is_empty())
        .ok_or_else(|| invalid("messages is required"))?;
    let config = &ANTHROPIC_CHAT_COMPLETIONS_CONFIG;
    let params = line
        .body
        .iter()
        .filter(|(name, _)| !matches!(name.as_str(), "model" | "messages"))
        .map(|(name, value)| {
            config
                .supported_openai_param_mappings()
                .iter()
                .find(|(openai, _)| openai == name)
                .map(|(_, anthropic)| ((*anthropic).to_string(), value.clone()))
                .ok_or_else(|| invalid(&format!("parameter {name} is not supported")))
        })
        .collect::<Result<Map<_, _>, _>>()?;
    if !params.contains_key("max_tokens") {
        return Err(invalid("max_tokens is required"));
    }
    if let Some(reason) = config.unsupported_reason(&messages, &params) {
        return Err(invalid(reason.0));
    }
    let params = config.transform_request(model, messages, params)?.body;
    Ok(json!({ "custom_id": line.custom_id, "params": params }))
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

    fn transform_create_batch_request(
        &self,
        model: Option<&str>,
        input_jsonl: &str,
    ) -> Result<Value, Error> {
        let requests = input_jsonl
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(|line| batch_request(model, line))
            .collect::<Result<Vec<_>, _>>()?;
        if requests.is_empty() {
            return Err(Error::InvalidRequest(
                "batch input file has no requests".into(),
            ));
        }
        Ok(json!({ "requests": requests }))
    }

    fn transform_create_batch_response(
        &self,
        response: AnthropicMessageBatch,
        now: i64,
    ) -> LiteLlmMessageBatch {
        self.transform_retrieve_batch_response(response, now)
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

    fn line(custom_id: &str, body: Value) -> String {
        json!({"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": body})
            .to_string()
    }

    fn user_turn(text: &str) -> Value {
        json!({"role": "user", "content": [{"type": "text", "text": text}]})
    }

    #[rstest]
    #[case::maps_openai_params_and_folds_system(
        None,
        line("r1", json!({
            "model": "anthropic/claude-sonnet",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"},
            ],
            "max_tokens": 16,
            "stop": ["END"],
            "temperature": 0.5,
            "top_p": 0.9,
        })),
        json!({"requests": [{"custom_id": "r1", "params": {
            "model": "claude-sonnet",
            "messages": [user_turn("hi")],
            "system": [{"type": "text", "text": "be brief"}],
            "max_tokens": 16,
            "stop_sequences": ["END"],
            "temperature": 0.5,
            "top_p": 0.9,
        }}]}),
    )]
    #[case::deployment_model_overrides_the_line_model(
        Some("claude-deployed"),
        line("r1", json!({"model": "alias", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8})),
        json!({"requests": [{"custom_id": "r1", "params": {
            "model": "claude-deployed", "messages": [user_turn("hi")], "max_tokens": 8,
        }}]}),
    )]
    #[case::keeps_line_order_and_skips_blank_lines(
        None,
        format!(
            "\n{}\r\n   \n{}\n",
            line("b", json!({"model": "m", "messages": [{"role": "user", "content": "two"}], "max_tokens": 2})),
            line("a", json!({"model": "m", "messages": [{"role": "user", "content": "one"}], "max_tokens": 1})),
        ),
        json!({"requests": [
            {"custom_id": "b", "params": {"model": "m", "messages": [user_turn("two")], "max_tokens": 2}},
            {"custom_id": "a", "params": {"model": "m", "messages": [user_turn("one")], "max_tokens": 1}},
        ]}),
    )]
    fn create_batch_request_translates_each_chat_line(
        #[case] model: Option<&str>,
        #[case] input: String,
        #[case] expected: Value,
    ) {
        assert_eq!(
            ANTHROPIC_BATCHES_TRANSFORMATION
                .transform_create_batch_request(model, &input)
                .unwrap(),
            expected
        );
    }

    #[rstest]
    #[case::empty_input(String::new(), "invalid request: batch input file has no requests")]
    #[case::blank_input("\n  \n".to_string(), "invalid request: batch input file has no requests")]
    #[case::not_json(
        "not-json".to_string(),
        "invalid request: invalid batch input line: expected ident at line 1 column 2",
    )]
    #[case::other_endpoint(
        json!({"custom_id": "r1", "method": "POST", "url": "/v1/embeddings", "body": {}}).to_string(),
        "invalid request: batch request r1: POST /v1/embeddings is not supported, only POST /v1/chat/completions",
    )]
    #[case::other_method(
        json!({"custom_id": "r1", "method": "GET", "url": "/v1/chat/completions", "body": {}}).to_string(),
        "invalid request: batch request r1: GET /v1/chat/completions is not supported, only POST /v1/chat/completions",
    )]
    #[case::no_model(
        line("r1", json!({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1})),
        "invalid request: batch request r1: model is required",
    )]
    #[case::no_messages(
        line("r1", json!({"model": "m", "max_tokens": 1})),
        "invalid request: batch request r1: messages is required",
    )]
    #[case::empty_messages(
        line("r1", json!({"model": "m", "messages": [], "max_tokens": 1})),
        "invalid request: batch request r1: messages is required",
    )]
    #[case::malformed_messages(
        line("r1", json!({"model": "m", "messages": "hi", "max_tokens": 1})),
        "invalid request: batch request r1: invalid messages: invalid type: string \"hi\", expected a sequence",
    )]
    #[case::unsupported_param(
        line("r1", json!({"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "tools": []})),
        "invalid request: batch request r1: parameter tools is not supported",
    )]
    #[case::no_max_tokens(
        line("r1", json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]})),
        "invalid request: batch request r1: max_tokens is required",
    )]
    #[case::streaming(
        line("r1", json!({"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "stream": true})),
        "invalid request: batch request r1: parameter stream is not supported",
    )]
    #[case::tool_turn(
        line("r1", json!({"model": "m", "messages": [{"role": "tool", "content": "x", "tool_call_id": "t"}], "max_tokens": 1})),
        "invalid request: batch request r1: unrecognized message field",
    )]
    #[case::opens_on_assistant_turn(
        line("r1", json!({"model": "m", "messages": [{"role": "assistant", "content": "hi"}], "max_tokens": 1})),
        "invalid request: batch request r1: conversation does not open on a user turn",
    )]
    #[case::one_bad_line_fails_the_batch(
        format!(
            "{}\n{}",
            line("ok", json!({"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1})),
            line("bad", json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]})),
        ),
        "invalid request: batch request bad: max_tokens is required",
    )]
    fn create_batch_request_rejects_lines_it_cannot_translate(
        #[case] input: String,
        #[case] message: &str,
    ) {
        assert_eq!(
            ANTHROPIC_BATCHES_TRANSFORMATION
                .transform_create_batch_request(None, &input)
                .unwrap_err()
                .to_string(),
            message
        );
    }

    #[test]
    fn create_batch_response_is_the_retrieved_batch() {
        let body = json!({
            "id": "msgbatch_new",
            "processing_status": "in_progress",
            "created_at": "2024-09-24T10:00:00Z",
            "request_counts": {"processing": 2},
        });
        let response: AnthropicMessageBatch = serde_json::from_value(body.clone()).unwrap();
        assert_eq!(
            ANTHROPIC_BATCHES_TRANSFORMATION.transform_create_batch_response(response, NOW),
            retrieve(body)
        );
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
