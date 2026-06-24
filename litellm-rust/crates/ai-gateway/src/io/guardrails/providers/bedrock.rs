use std::collections::HashMap;
use std::time::{Instant, SystemTime};

use aws_credential_types::Credentials;
use aws_sigv4::http_request::{sign, SignableBody, SignableRequest, SigningSettings};
use aws_sigv4::sign::v4;
use litellm_core::guardrails::{
    BedrockConfig, Detection, GuardrailInput, GuardrailOutcome, InputType, ProviderError,
    RequestContext, Verdict,
};
use serde::{Deserialize, Serialize};

use super::{map_send_error, read_success_json};
use crate::io::guardrails::http;
use crate::io::guardrails::provider::Guardrail;

const SERVICE: &str = "bedrock";

pub struct BedrockGuardrail {
    config: BedrockConfig,
}

impl BedrockGuardrail {
    pub fn new(config: BedrockConfig) -> Self {
        Self { config }
    }

    /// The host resolves the region and static credentials into the config
    /// before dispatch; absence is a config-builder bug, so treat it as an
    /// invalid config rather than a silent pass.
    fn region(&self) -> Result<&str, ProviderError> {
        self.config
            .aws_region_name
            .as_deref()
            .ok_or_else(|| ProviderError::InvalidConfig {
                message: "bedrock requires aws_region_name".to_owned(),
            })
    }

    fn url(&self, region: &str) -> String {
        let base = match &self.config.aws_bedrock_runtime_endpoint {
            Some(endpoint) => endpoint.trim_end_matches('/').to_owned(),
            None => format!("https://bedrock-runtime.{region}.amazonaws.com"),
        };
        format!(
            "{base}/guardrail/{}/version/{}/apply",
            self.config.guardrail_identifier, self.config.guardrail_version
        )
    }

    fn credentials(&self) -> Result<Credentials, ProviderError> {
        match (
            &self.config.aws_access_key_id,
            &self.config.aws_secret_access_key,
        ) {
            (Some(access_key), Some(secret_key)) => Ok(Credentials::new(
                access_key.clone(),
                secret_key.clone(),
                self.config.aws_session_token.clone(),
                None,
                "guardrails-config",
            )),
            _ => Err(ProviderError::InvalidConfig {
                message: "bedrock requires aws_access_key_id and aws_secret_access_key".to_owned(),
            }),
        }
    }

    fn sign_request(
        &self,
        url: &str,
        body: &[u8],
        credentials: &Credentials,
        region: &str,
    ) -> Result<Vec<(String, String)>, ProviderError> {
        let identity = credentials.clone().into();
        let signing_params = v4::SigningParams::builder()
            .identity(&identity)
            .region(region)
            .name(SERVICE)
            .time(SystemTime::now())
            .settings(SigningSettings::default())
            .build()
            .map_err(|e| ProviderError::InvalidConfig {
                message: format!("failed to build SigV4 params: {e}"),
            })?;

        let signable = SignableRequest::new(
            "POST",
            url,
            [("content-type", "application/json")].into_iter(),
            SignableBody::Bytes(body),
        )
        .map_err(|e| ProviderError::InvalidConfig {
            message: format!("failed to build signable request: {e}"),
        })?;

        let (instructions, _signature) = sign(signable, &signing_params.into())
            .map_err(|e| ProviderError::InvalidConfig {
                message: format!("SigV4 signing failed: {e}"),
            })?
            .into_parts();

        Ok(instructions
            .headers()
            .map(|(k, v)| (k.to_owned(), v.to_owned()))
            .collect())
    }
}

#[derive(Serialize)]
struct ApplyGuardrailRequest {
    source: &'static str,
    content: Vec<ContentItem>,
}

#[derive(Serialize)]
struct ContentItem {
    text: TextBlock,
}

#[derive(Serialize)]
struct TextBlock {
    text: String,
}

#[derive(Deserialize)]
struct ApplyGuardrailResponse {
    #[serde(default)]
    action: Option<String>,
    #[serde(default)]
    output: Vec<OutputItem>,
    #[serde(default)]
    outputs: Vec<OutputItem>,
    #[serde(default)]
    assessments: Vec<serde_json::Value>,
}

#[derive(Deserialize)]
struct OutputItem {
    #[serde(default)]
    text: Option<String>,
}

fn has_blocked_action(assessments: &[serde_json::Value]) -> bool {
    fn any_blocked(items: Option<&serde_json::Value>, key: &str) -> bool {
        items
            .and_then(|p| p.get(key))
            .and_then(|v| v.as_array())
            .is_some_and(|arr| {
                arr.iter()
                    .any(|item| item.get("action").and_then(|a| a.as_str()) == Some("BLOCKED"))
            })
    }

    assessments.iter().any(|assessment| {
        any_blocked(assessment.get("topicPolicy"), "topics")
            || any_blocked(assessment.get("contentPolicy"), "filters")
            || any_blocked(assessment.get("wordPolicy"), "customWords")
            || any_blocked(assessment.get("wordPolicy"), "managedWordLists")
            || any_blocked(assessment.get("sensitiveInformationPolicy"), "piiEntities")
            || any_blocked(assessment.get("sensitiveInformationPolicy"), "regexes")
            || any_blocked(assessment.get("contextualGroundingPolicy"), "filters")
    })
}

fn blocked_detections(assessments: &[serde_json::Value]) -> Vec<Detection> {
    let policies: &[(&str, &[&str])] = &[
        ("topicPolicy", &["topics"]),
        ("contentPolicy", &["filters"]),
        ("wordPolicy", &["customWords", "managedWordLists"]),
        ("sensitiveInformationPolicy", &["piiEntities", "regexes"]),
        ("contextualGroundingPolicy", &["filters"]),
    ];

    let mut detections = vec![];
    for assessment in assessments {
        for (policy, keys) in policies {
            for key in *keys {
                let Some(items) = assessment
                    .get(policy)
                    .and_then(|p| p.get(key))
                    .and_then(|v| v.as_array())
                else {
                    continue;
                };
                for item in items {
                    let action = item.get("action").and_then(|a| a.as_str());
                    if action != Some("BLOCKED") {
                        continue;
                    }
                    let label = item
                        .get("name")
                        .or_else(|| item.get("type"))
                        .and_then(|v| v.as_str())
                        .map(str::to_owned);
                    detections.push(Detection {
                        category: (*policy).to_owned(),
                        label,
                        score: None,
                        action: Some("BLOCKED".to_owned()),
                    });
                }
            }
        }
    }
    detections
}

impl Guardrail for BedrockGuardrail {
    fn apply(
        &self,
        input: &GuardrailInput,
        input_type: InputType,
        _ctx: &RequestContext,
    ) -> Result<GuardrailOutcome, ProviderError> {
        let start = Instant::now();

        let region = self.region()?.to_owned();
        let url = self.url(&region);
        let credentials = self.credentials()?;

        let body = ApplyGuardrailRequest {
            source: match input_type {
                InputType::Request => "INPUT",
                InputType::Response => "OUTPUT",
            },
            content: input
                .texts
                .iter()
                .map(|t| ContentItem {
                    text: TextBlock { text: t.clone() },
                })
                .collect(),
        };
        let body_bytes = serde_json::to_vec(&body).map_err(|e| ProviderError::InvalidConfig {
            message: format!("failed to serialize bedrock request: {e}"),
        })?;

        let signed_headers = self.sign_request(&url, &body_bytes, &credentials, &region)?;

        let mut req = http::client()
            .post(&url)
            .header("content-type", "application/json")
            .body(body_bytes);
        for (name, value) in signed_headers {
            req = req.header(name, value);
        }

        let resp = req.send().map_err(map_send_error(start))?;
        let raw = read_success_json(resp)?;

        let parsed: ApplyGuardrailResponse =
            serde_json::from_value(raw.clone()).map_err(|e| ProviderError::InvalidResponse {
                message: e.to_string(),
            })?;

        let duration_ms = start.elapsed().as_millis() as u64;

        if parsed.action.as_deref() != Some("GUARDRAIL_INTERVENED") {
            return Ok(GuardrailOutcome {
                verdict: Verdict::Pass,
                provider_response: raw,
                duration_ms,
            });
        }

        let output_texts: Vec<String> = if parsed.output.is_empty() {
            &parsed.outputs
        } else {
            &parsed.output
        }
        .iter()
        .filter_map(|o| o.text.clone())
        .collect();

        if has_blocked_action(&parsed.assessments) {
            let violation_message = output_texts
                .first()
                .cloned()
                .unwrap_or_else(|| "Violated guardrail policy".to_owned());
            return Ok(GuardrailOutcome {
                verdict: Verdict::Block {
                    violation_message,
                    detections: blocked_detections(&parsed.assessments),
                },
                provider_response: raw,
                duration_ms,
            });
        }

        if !output_texts.is_empty() {
            return Ok(GuardrailOutcome {
                verdict: Verdict::Mask {
                    texts: output_texts,
                    masked_entity_count: HashMap::new(),
                    detections: vec![],
                },
                provider_response: raw,
                duration_ms,
            });
        }

        Ok(GuardrailOutcome {
            verdict: Verdict::Pass,
            provider_response: raw,
            duration_ms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocked_assessment_detected() {
        let assessments = vec![serde_json::json!({
            "contentPolicy": {
                "filters": [{"type": "VIOLENCE", "action": "BLOCKED"}]
            }
        })];
        assert!(has_blocked_action(&assessments));
        let detections = blocked_detections(&assessments);
        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].category, "contentPolicy");
        assert_eq!(detections[0].label.as_deref(), Some("VIOLENCE"));
    }

    #[test]
    fn anonymized_only_is_not_blocked() {
        let assessments = vec![serde_json::json!({
            "sensitiveInformationPolicy": {
                "piiEntities": [{"type": "EMAIL", "action": "ANONYMIZED"}]
            }
        })];
        assert!(!has_blocked_action(&assessments));
    }
}
