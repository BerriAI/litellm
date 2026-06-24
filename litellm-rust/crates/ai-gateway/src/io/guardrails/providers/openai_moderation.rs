use std::collections::HashMap;
use std::time::Instant;

use litellm_core::guardrails::{
    Detection, GuardrailInput, GuardrailOutcome, InputType, OpenaiModerationConfig, ProviderError,
    RequestContext, Verdict,
};
use serde::{Deserialize, Serialize};

use super::{map_send_error, read_success_json};
use crate::io::guardrails::http;
use crate::io::guardrails::provider::Guardrail;

const DEFAULT_API_BASE: &str = "https://api.openai.com/v1";
const DEFAULT_MODEL: &str = "omni-moderation-latest";

pub struct OpenaiModeration {
    config: OpenaiModerationConfig,
}

impl OpenaiModeration {
    pub fn new(config: OpenaiModerationConfig) -> Self {
        Self { config }
    }

    fn url(&self) -> String {
        let base = self
            .config
            .api_base
            .as_deref()
            .unwrap_or(DEFAULT_API_BASE)
            .trim_end_matches('/');
        format!("{base}/moderations")
    }
}

#[derive(Serialize)]
struct ModerationRequest<'a> {
    model: &'a str,
    input: String,
}

#[derive(Deserialize)]
struct ModerationResponse {
    results: Vec<ModerationResult>,
}

#[derive(Deserialize)]
struct ModerationResult {
    flagged: bool,
    #[serde(default)]
    categories: HashMap<String, bool>,
    #[serde(default)]
    category_scores: HashMap<String, f64>,
}

impl Guardrail for OpenaiModeration {
    fn apply(
        &self,
        input: &GuardrailInput,
        _input_type: InputType,
        _ctx: &RequestContext,
    ) -> Result<GuardrailOutcome, ProviderError> {
        let start = Instant::now();
        let api_key =
            self.config
                .api_key
                .as_deref()
                .ok_or_else(|| ProviderError::InvalidConfig {
                    message: "openai_moderation requires api_key or OPENAI_API_KEY".to_owned(),
                })?;

        let body = ModerationRequest {
            model: self.config.model.as_deref().unwrap_or(DEFAULT_MODEL),
            input: input.texts.join("\n"),
        };

        let resp = http::client()
            .post(self.url())
            .bearer_auth(api_key)
            .json(&body)
            .send()
            .map_err(map_send_error(start))?;

        let raw = read_success_json(resp)?;

        let parsed: ModerationResponse =
            serde_json::from_value(raw.clone()).map_err(|e| ProviderError::InvalidResponse {
                message: e.to_string(),
            })?;

        let flagged = parsed.results.iter().any(|r| r.flagged);
        let verdict = if flagged {
            let detections: Vec<Detection> = parsed
                .results
                .iter()
                .flat_map(|r| {
                    r.categories
                        .iter()
                        .filter(|(_, hit)| **hit)
                        .map(|(category, _)| Detection {
                            category: category.clone(),
                            label: None,
                            score: r.category_scores.get(category).copied(),
                            action: Some("BLOCKED".to_owned()),
                        })
                })
                .collect();

            let violated: Vec<&str> = detections.iter().map(|d| d.category.as_str()).collect();
            Verdict::Block {
                violation_message: format!(
                    "Violated OpenAI moderation policy: {}",
                    violated.join(", ")
                ),
                detections,
            }
        } else {
            Verdict::Pass
        };

        Ok(GuardrailOutcome {
            verdict,
            provider_response: raw,
            duration_ms: start.elapsed().as_millis() as u64,
        })
    }
}
