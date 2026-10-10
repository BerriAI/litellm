use litellm_auth::{CredentialPlacement, SecretValue};
use litellm_core_utils::settings::resolve_non_empty;
use litellm_llms_types::formats::messages::{MessagesOptionalParams, MessagesRequest};
use serde_json::{Map, Value};

use crate::{
    Error,
    anthropic::common_utils::{filter_billing_headers_from_system, has_anthropic_credential},
    base_llm::{
        auth::AuthScheme,
        messages::transformation::{Headers, ValidatedEnvironment},
    },
};

pub fn complete_messages_url(base: &str) -> String {
    let base = base.trim_end_matches('/');
    if base.ends_with("/v1/messages") {
        return base.into();
    }
    format!("{}/v1/messages", base.strip_suffix("/v1").unwrap_or(base))
}

pub fn portable_cache_control(body: Value) -> Value {
    let Value::Object(fields) = body else {
        return body;
    };
    Value::Object(
        portable_block(fields)
            .into_iter()
            .map(|(key, value)| {
                let value = match key.as_str() {
                    "system" | "tools" => map_blocks(value, portable_value),
                    "messages" => map_blocks(value, portable_message),
                    _ => value,
                };
                (key, value)
            })
            .collect(),
    )
}

fn portable_block(fields: Map<String, Value>) -> Map<String, Value> {
    fields
        .into_iter()
        .filter_map(|(key, value)| {
            if key != "cache_control" {
                return Some((key, value));
            }
            let Value::Object(cache) = value else {
                return None;
            };
            let kind = cache
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("ephemeral");
            Some((key, serde_json::json!({"type": kind})))
        })
        .collect()
}

fn portable_value(value: Value) -> Value {
    match value {
        Value::Object(fields) => Value::Object(portable_block(fields)),
        other => other,
    }
}

fn map_blocks(value: Value, map: fn(Value) -> Value) -> Value {
    match value {
        Value::Array(blocks) => Value::Array(blocks.into_iter().map(map).collect()),
        other => other,
    }
}

fn portable_message(value: Value) -> Value {
    let Value::Object(fields) = value else {
        return value;
    };
    Value::Object(
        fields
            .into_iter()
            .map(|(key, value)| {
                let value = if key == "content" {
                    map_blocks(value, portable_content)
                } else {
                    value
                };
                (key, value)
            })
            .collect(),
    )
}

fn portable_content(value: Value) -> Value {
    let Value::Object(fields) = value else {
        return value;
    };
    let tool_result = fields.get("type").and_then(Value::as_str) == Some("tool_result");
    Value::Object(
        portable_block(fields)
            .into_iter()
            .map(|(key, value)| {
                let value = if tool_result && key == "content" {
                    map_blocks(value, portable_value)
                } else {
                    value
                };
                (key, value)
            })
            .collect(),
    )
}

/// Python's `get_complete_url` for a host whose Messages endpoint is `<base>/v1/messages`:
/// the explicit base, then the host's environment variable, then its default.
pub fn compatible_host_url(
    api_base: Option<&str>,
    api_base_env: &'static str,
    default_api_base: &str,
    env: &dyn Fn(&str) -> Option<String>,
) -> String {
    let base = resolve_non_empty(api_base, env, &[api_base_env])
        .unwrap_or_else(|| default_api_base.into());
    complete_messages_url(&base)
}

/// Python's `validate_environment` for a host behind its own bearer key: the key must
/// resolve, and a caller's own Anthropic credential is forwarded in its place.
pub fn compatible_host_environment(
    headers: Headers,
    api_key: Option<&str>,
    provider: &'static str,
    api_key_env: &'static str,
    env: &dyn Fn(&str) -> Option<String>,
) -> Result<ValidatedEnvironment, Error> {
    let key = resolve_non_empty(api_key, env, &[api_key_env]).ok_or(Error::Auth(
        litellm_auth::Error::MissingApiKey {
            provider,
            environment_variable: api_key_env,
        },
    ))?;
    if has_anthropic_credential(&headers) {
        return Ok(ValidatedEnvironment {
            headers,
            auth: AuthScheme::Forwarded,
        });
    }
    Ok(ValidatedEnvironment {
        headers,
        auth: AuthScheme::Credential {
            placement: CredentialPlacement::Bearer,
            secret: SecretValue::new(key),
        },
    })
}

/// First-party billing system blocks never leave the gateway for another host.
pub fn without_billing_blocks(request: MessagesRequest) -> MessagesRequest {
    MessagesRequest {
        params: MessagesOptionalParams {
            system: request
                .params
                .system
                .and_then(filter_billing_headers_from_system),
            ..request.params
        },
        ..request
    }
}
