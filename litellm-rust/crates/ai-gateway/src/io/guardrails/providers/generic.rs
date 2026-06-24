use std::collections::HashMap;
use std::time::Instant;

use litellm_core::guardrails::{
    Detection, GenericApiConfig, GuardrailInput, GuardrailOutcome, InputType, Message,
    ProviderError, RequestContext, UnreachableFallback, Verdict,
};
use serde::{Deserialize, Serialize};

use super::map_send_error;
use crate::io::guardrails::headers;
use crate::io::guardrails::http;
use crate::io::guardrails::provider::Guardrail;

const ENDPOINT_SUFFIX: &str = "/beta/litellm_basic_guardrail_api";

fn normalize_api_base(raw: &str) -> String {
    if raw.ends_with(ENDPOINT_SUFFIX) {
        raw.to_owned()
    } else {
        format!("{}{ENDPOINT_SUFFIX}", raw.trim_end_matches('/'))
    }
}

pub struct GenericGuardrailApi {
    config: GenericApiConfig,
    api_base: String,
}

impl GenericGuardrailApi {
    pub fn new(config: GenericApiConfig) -> Self {
        let api_base = normalize_api_base(&config.api_base);
        Self { config, api_base }
    }

    /// Map a transport/upstream failure to either fail-open (Ok, caller passes)
    /// or fail-closed (propagate the error).
    fn classify_error(&self, err: &ProviderError) -> Result<(), ProviderError> {
        let fallback = self
            .config
            .unreachable_fallback
            .unwrap_or(UnreachableFallback::FailClosed);

        if err.is_unreachable() && fallback == UnreachableFallback::FailOpen {
            return Ok(());
        }
        Err(err.clone())
    }

    fn fail_open(err: &ProviderError, start: Instant) -> GuardrailOutcome {
        GuardrailOutcome {
            verdict: Verdict::Pass,
            provider_response: serde_json::json!({"fail_open": true, "error": err.to_string()}),
            duration_ms: start.elapsed().as_millis() as u64,
        }
    }
}

#[derive(Serialize)]
struct GenericRequest<'a> {
    input_type: InputType,
    texts: &'a [String],
    images: &'a [String],
    structured_messages: &'a [Message],
    tools: &'a [serde_json::Value],
    tool_calls: &'a [serde_json::Value],
    model: &'a Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    litellm_call_id: &'a Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    litellm_trace_id: &'a Option<String>,
    request_data: &'a serde_json::Map<String, serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    request_headers: Option<HashMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    litellm_version: &'a Option<String>,
    #[serde(flatten)]
    additional_provider_specific_params: serde_json::Map<String, serde_json::Value>,
}

#[derive(Deserialize)]
struct GenericResponse {
    action: GenericAction,
    #[serde(default)]
    blocked_reason: Option<String>,
    #[serde(default)]
    texts: Option<Vec<String>>,
}

#[derive(Deserialize, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum GenericAction {
    Blocked,
    None,
    GuardrailIntervened,
}

impl Guardrail for GenericGuardrailApi {
    fn apply(
        &self,
        input: &GuardrailInput,
        input_type: InputType,
        ctx: &RequestContext,
    ) -> Result<GuardrailOutcome, ProviderError> {
        let start = Instant::now();

        let sanitized_headers = ctx
            .request_headers
            .as_ref()
            .map(|raw| headers::sanitize_headers(raw, &ctx.extra_header_allowlist));

        let mut merged_params = self.config.additional_provider_specific_params.clone();
        for (k, v) in &ctx.dynamic_params {
            merged_params.insert(k.clone(), v.clone());
        }

        let body = GenericRequest {
            input_type,
            texts: &input.texts,
            images: &input.images,
            structured_messages: &input.structured_messages,
            tools: &input.tools,
            tool_calls: &input.tool_calls,
            model: &input.model,
            litellm_call_id: &ctx.litellm_call_id,
            litellm_trace_id: &ctx.litellm_trace_id,
            request_data: &ctx.user_api_key_metadata,
            request_headers: sanitized_headers,
            litellm_version: &ctx.litellm_version,
            additional_provider_specific_params: merged_params,
        };

        let mut req = http::client().post(&self.api_base).json(&body);
        if let Some(key) = &self.config.api_key {
            req = req.header("x-api-key", key);
        }
        if let Some(hdrs) = &self.config.headers {
            for (k, v) in hdrs {
                req = req.header(k, v);
            }
        }

        let resp = match req.send().map_err(map_send_error(start)) {
            Ok(r) => r,
            Err(e) => {
                self.classify_error(&e)?;
                return Ok(Self::fail_open(&e, start));
            }
        };

        let status = resp.status().as_u16();
        if !resp.status().is_success() {
            let body_text = resp.text().unwrap_or_default();
            let err = ProviderError::Upstream {
                status,
                body: http::truncate_body(&body_text),
            };
            self.classify_error(&err)?;
            return Ok(Self::fail_open(&err, start));
        }

        let raw: serde_json::Value = resp.json().map_err(|e| ProviderError::InvalidResponse {
            message: e.to_string(),
        })?;

        let parsed: GenericResponse =
            serde_json::from_value(raw.clone()).map_err(|e| ProviderError::InvalidResponse {
                message: e.to_string(),
            })?;

        let verdict = match parsed.action {
            GenericAction::Blocked => Verdict::Block {
                violation_message: parsed
                    .blocked_reason
                    .unwrap_or_else(|| "Content violates policy".to_owned()),
                detections: vec![Detection {
                    category: "blocked".to_owned(),
                    label: None,
                    score: None,
                    action: Some("BLOCKED".to_owned()),
                }],
            },
            GenericAction::GuardrailIntervened => match parsed.texts {
                Some(texts) => Verdict::Mask {
                    texts,
                    masked_entity_count: HashMap::new(),
                    detections: vec![],
                },
                None => Verdict::Block {
                    violation_message: parsed
                        .blocked_reason
                        .unwrap_or_else(|| "Guardrail intervened".to_owned()),
                    detections: vec![],
                },
            },
            GenericAction::None => Verdict::Pass,
        };

        Ok(GuardrailOutcome {
            verdict,
            provider_response: raw,
            duration_ms: start.elapsed().as_millis() as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_normalization_appends_suffix() {
        assert_eq!(
            normalize_api_base("http://localhost:8888"),
            "http://localhost:8888/beta/litellm_basic_guardrail_api"
        );
    }

    #[test]
    fn url_normalization_strips_trailing_slash() {
        assert_eq!(
            normalize_api_base("http://localhost:8888/"),
            "http://localhost:8888/beta/litellm_basic_guardrail_api"
        );
    }

    #[test]
    fn url_normalization_preserves_existing_suffix() {
        let url = "http://localhost:8888/beta/litellm_basic_guardrail_api";
        assert_eq!(normalize_api_base(url), url);
    }
}
