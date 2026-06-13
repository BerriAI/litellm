use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::{
    Detection, Guardrail, GuardrailInput, GuardrailOutcome, InputType, OpenaiModerationConfig,
    ProviderError, RequestContext, Verdict,
};
use crate::HttpClient;
use serde::{Deserialize, Serialize};

const DEFAULT_API_BASE: &str = "https://api.openai.com/v1";
const DEFAULT_MODEL: &str = "omni-moderation-latest";

pub struct OpenaiModeration {
    config: OpenaiModerationConfig,
    http: Arc<HttpClient>,
}

impl OpenaiModeration {
    pub fn new(config: OpenaiModerationConfig, http: Arc<HttpClient>) -> Self {
        Self { config, http }
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

    fn api_key(&self) -> Result<String, ProviderError> {
        if let Some(key) = &self.config.api_key {
            return Ok(key.clone());
        }
        std::env::var("OPENAI_API_KEY").map_err(|_| ProviderError::InvalidConfig {
            message: "openai_moderation requires api_key or OPENAI_API_KEY".to_owned(),
        })
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

#[async_trait::async_trait]
impl Guardrail for OpenaiModeration {
    fn provider_name(&self) -> &'static str {
        "openai_moderation"
    }

    async fn apply(
        &self,
        input: &GuardrailInput,
        _input_type: InputType,
        _ctx: &RequestContext,
    ) -> Result<GuardrailOutcome, ProviderError> {
        let start = Instant::now();
        let api_key = self.api_key()?;

        let body = ModerationRequest {
            model: self.config.model.as_deref().unwrap_or(DEFAULT_MODEL),
            input: input.texts.join("\n"),
        };

        let resp = self
            .http
            .inner()
            .post(self.url())
            .bearer_auth(api_key)
            .json(&body)
            .send()
            .await
            .map_err(super::map_send_error(start))?;

        let raw = super::read_success_json(resp).await?;

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
