use std::sync::Arc;
use std::time::Instant;

use guardrails_core::{
    Detection, GenericApiConfig, Guardrail, GuardrailInput, GuardrailOutcome, InputType,
    ProviderError, RequestContext, UnreachableFallback, Verdict,
};
use guardrails_http::HttpClient;
use serde::{Deserialize, Serialize};

pub struct GenericGuardrailApi {
    config: GenericApiConfig,
    http: Arc<HttpClient>,
}

impl GenericGuardrailApi {
    pub fn new(config: GenericApiConfig, http: Arc<HttpClient>) -> Self {
        Self { config, http }
    }

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
}

#[derive(Serialize)]
struct GenericRequest<'a> {
    input_type: InputType,
    texts: &'a [String],
    images: &'a [String],
    structured_messages: &'a [guardrails_core::Message],
    tools: &'a [serde_json::Value],
    tool_calls: &'a [serde_json::Value],
    model: &'a Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    litellm_call_id: &'a Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    litellm_trace_id: &'a Option<String>,
    #[serde(flatten)]
    additional_provider_specific_params: &'a serde_json::Map<String, serde_json::Value>,
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

#[async_trait::async_trait]
impl Guardrail for GenericGuardrailApi {
    fn provider_name(&self) -> &'static str {
        "generic_guardrail_api"
    }

    async fn apply(
        &self,
        input: &GuardrailInput,
        input_type: InputType,
        ctx: &RequestContext,
    ) -> Result<GuardrailOutcome, ProviderError> {
        let start = Instant::now();

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
            additional_provider_specific_params: &self.config.additional_provider_specific_params,
        };

        let mut req = self.http.inner().post(&self.config.api_base).json(&body);

        if let Some(key) = &self.config.api_key {
            req = req.header("x-api-key", key);
        }
        if let Some(headers) = &self.config.headers {
            for (k, v) in headers {
                req = req.header(k, v);
            }
        }

        let resp = req.send().await.map_err(|e| {
            if e.is_timeout() {
                ProviderError::Timeout {
                    ms: start.elapsed().as_millis() as u64,
                }
            } else {
                ProviderError::Network {
                    message: e.to_string(),
                }
            }
        });

        let resp = match resp {
            Ok(r) => r,
            Err(e) => {
                self.classify_error(&e)?;
                return Ok(GuardrailOutcome {
                    verdict: Verdict::Pass,
                    provider_response: serde_json::json!({"fail_open": true, "error": e.to_string()}),
                    duration_ms: start.elapsed().as_millis() as u64,
                });
            }
        };

        let status = resp.status().as_u16();
        if !resp.status().is_success() {
            let body_text = resp.text().await.unwrap_or_default();
            let err = ProviderError::Upstream {
                status,
                body: body_text,
            };
            self.classify_error(&err)?;
            return Ok(GuardrailOutcome {
                verdict: Verdict::Pass,
                provider_response: serde_json::json!({"fail_open": true, "error": err.to_string()}),
                duration_ms: start.elapsed().as_millis() as u64,
            });
        }

        let raw: serde_json::Value =
            resp.json()
                .await
                .map_err(|e| ProviderError::InvalidResponse {
                    message: e.to_string(),
                })?;

        let parsed: GenericResponse =
            serde_json::from_value(raw.clone()).map_err(|e| ProviderError::InvalidResponse {
                message: e.to_string(),
            })?;

        let duration_ms = start.elapsed().as_millis() as u64;

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
                    masked_entity_count: std::collections::HashMap::new(),
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
            duration_ms,
        })
    }
}
