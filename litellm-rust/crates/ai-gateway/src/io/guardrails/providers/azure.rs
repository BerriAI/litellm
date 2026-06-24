use std::time::Instant;

use litellm_core::guardrails::{
    AzurePromptShieldConfig, AzureTextModerationConfig, Detection, GuardrailInput,
    GuardrailOutcome, InputType, ProviderError, RequestContext, Verdict,
};
use serde::{Deserialize, Serialize};

use super::{map_send_error, read_success_json};
use crate::io::guardrails::http;
use crate::io::guardrails::provider::Guardrail;

const DEFAULT_API_VERSION: &str = "2024-09-01";
const MAX_CHUNK_CHARS: usize = 10_000;
const DEFAULT_SEVERITY_THRESHOLD: u8 = 2;
const DEFAULT_CATEGORIES: &[&str] = &["Hate", "Sexual", "SelfHarm", "Violence"];

fn split_text_by_chars(text: &str, max_chars: usize) -> Vec<String> {
    if text.chars().count() <= max_chars {
        return vec![text.to_owned()];
    }
    let chars: Vec<char> = text.chars().collect();
    chars
        .chunks(max_chars)
        .map(|c| c.iter().collect())
        .collect()
}

pub struct AzurePromptShield {
    config: AzurePromptShieldConfig,
}

impl AzurePromptShield {
    pub fn new(config: AzurePromptShieldConfig) -> Self {
        Self { config }
    }

    fn url(&self) -> String {
        let base = self.config.api_base.trim_end_matches('/');
        let version = self
            .config
            .api_version
            .as_deref()
            .unwrap_or(DEFAULT_API_VERSION);
        format!("{base}/contentsafety/text:shieldPrompt?api-version={version}")
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ShieldPromptRequest<'a> {
    user_prompt: &'a str,
    documents: Vec<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ShieldPromptResponse {
    user_prompt_analysis: PromptAnalysis,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PromptAnalysis {
    attack_detected: bool,
}

impl Guardrail for AzurePromptShield {
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
                    message: "azure/prompt_shield requires api_key".to_owned(),
                })?;

        let mut last_raw = serde_json::Value::Null;
        for text in &input.texts {
            for chunk in split_text_by_chars(text, MAX_CHUNK_CHARS) {
                let resp = http::client()
                    .post(self.url())
                    .header("Ocp-Apim-Subscription-Key", api_key)
                    .json(&ShieldPromptRequest {
                        user_prompt: &chunk,
                        documents: vec![],
                    })
                    .send()
                    .map_err(map_send_error(start))?;

                let raw = read_success_json(resp)?;
                let parsed: ShieldPromptResponse =
                    serde_json::from_value(raw.clone()).map_err(|e| {
                        ProviderError::InvalidResponse {
                            message: e.to_string(),
                        }
                    })?;

                if parsed.user_prompt_analysis.attack_detected {
                    return Ok(GuardrailOutcome {
                        verdict: Verdict::Block {
                            violation_message:
                                "Violated Azure Prompt Shield guardrail policy: attack detected"
                                    .to_owned(),
                            detections: vec![Detection {
                                category: "prompt_attack".to_owned(),
                                label: None,
                                score: None,
                                action: Some("BLOCKED".to_owned()),
                            }],
                        },
                        provider_response: raw,
                        duration_ms: start.elapsed().as_millis() as u64,
                    });
                }
                last_raw = raw;
            }
        }

        Ok(GuardrailOutcome {
            verdict: Verdict::Pass,
            provider_response: last_raw,
            duration_ms: start.elapsed().as_millis() as u64,
        })
    }
}

pub struct AzureTextModeration {
    config: AzureTextModerationConfig,
}

impl AzureTextModeration {
    pub fn new(config: AzureTextModerationConfig) -> Self {
        Self { config }
    }

    fn url(&self) -> String {
        let base = self.config.api_base.trim_end_matches('/');
        let version = self
            .config
            .api_version
            .as_deref()
            .unwrap_or(DEFAULT_API_VERSION);
        format!("{base}/contentsafety/text:analyze?api-version={version}")
    }

    fn threshold_for(&self, category: &str) -> u8 {
        if !self.config.severity_threshold_by_category.is_empty() {
            return self
                .config
                .severity_threshold_by_category
                .get(category)
                .copied()
                .unwrap_or(
                    self.config
                        .severity_threshold
                        .unwrap_or(DEFAULT_SEVERITY_THRESHOLD),
                );
        }
        self.config
            .severity_threshold
            .unwrap_or(DEFAULT_SEVERITY_THRESHOLD)
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct TextAnalyzeRequest<'a> {
    text: &'a str,
    categories: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    blocklist_names: Option<&'a [String]>,
    halt_on_blocklist_hit: bool,
    output_type: &'a str,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TextAnalyzeResponse {
    #[serde(default)]
    categories_analysis: Vec<CategoryAnalysis>,
}

#[derive(Deserialize)]
struct CategoryAnalysis {
    category: String,
    severity: u8,
}

impl Guardrail for AzureTextModeration {
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
                    message: "azure/text_moderations requires api_key".to_owned(),
                })?;

        let categories: Vec<String> = self
            .config
            .categories
            .clone()
            .unwrap_or_else(|| DEFAULT_CATEGORIES.iter().map(|c| (*c).to_owned()).collect());
        let blocklist_names = if self.config.blocklist_names.is_empty() {
            None
        } else {
            Some(self.config.blocklist_names.as_slice())
        };
        let output_type = self
            .config
            .output_type
            .as_deref()
            .unwrap_or("FourSeverityLevels");

        let mut last_raw = serde_json::Value::Null;
        for text in &input.texts {
            for chunk in split_text_by_chars(text, MAX_CHUNK_CHARS) {
                let resp = http::client()
                    .post(self.url())
                    .header("Ocp-Apim-Subscription-Key", api_key)
                    .json(&TextAnalyzeRequest {
                        text: &chunk,
                        categories: categories.clone(),
                        blocklist_names,
                        halt_on_blocklist_hit: self.config.halt_on_blocklist_hit,
                        output_type,
                    })
                    .send()
                    .map_err(map_send_error(start))?;

                let raw = read_success_json(resp)?;
                let parsed: TextAnalyzeResponse =
                    serde_json::from_value(raw.clone()).map_err(|e| {
                        ProviderError::InvalidResponse {
                            message: e.to_string(),
                        }
                    })?;

                for analysis in &parsed.categories_analysis {
                    let threshold = self.threshold_for(&analysis.category);
                    if analysis.severity >= threshold {
                        return Ok(GuardrailOutcome {
                            verdict: Verdict::Block {
                                violation_message: format!(
                                    "Azure Content Safety Guardrail: {} crossed severity {}, Got severity: {}",
                                    analysis.category, threshold, analysis.severity
                                ),
                                detections: vec![Detection {
                                    category: analysis.category.clone(),
                                    label: None,
                                    score: Some(f64::from(analysis.severity)),
                                    action: Some("BLOCKED".to_owned()),
                                }],
                            },
                            provider_response: raw,
                            duration_ms: start.elapsed().as_millis() as u64,
                        });
                    }
                }
                last_raw = raw;
            }
        }

        Ok(GuardrailOutcome {
            verdict: Verdict::Pass,
            provider_response: last_raw,
            duration_ms: start.elapsed().as_millis() as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunking_splits_long_text() {
        let text = "a".repeat(25_000);
        let chunks = split_text_by_chars(&text, MAX_CHUNK_CHARS);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].len(), 10_000);
        assert_eq!(chunks[2].len(), 5_000);
    }

    #[test]
    fn chunking_keeps_short_text_whole() {
        let chunks = split_text_by_chars("hello", MAX_CHUNK_CHARS);
        assert_eq!(chunks, vec!["hello"]);
    }
}
