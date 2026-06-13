use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::{
    Detection, Guardrail, GuardrailInput, GuardrailOutcome, InputType, PiiAction, PresidioConfig,
    ProviderError, RequestContext, Verdict,
};
use crate::HttpClient;
use serde::{Deserialize, Serialize};

const ALL_THRESHOLD_KEY: &str = "ALL";

fn normalize_base(raw: &str) -> String {
    let with_scheme = if raw.starts_with("http://") || raw.starts_with("https://") {
        raw.to_owned()
    } else {
        format!("http://{raw}")
    };
    format!("{}/", with_scheme.trim_end_matches('/'))
}

pub struct Presidio {
    config: PresidioConfig,
    analyzer_base: String,
    anonymizer_base: Option<String>,
    http: Arc<HttpClient>,
}

impl Presidio {
    pub fn new(config: PresidioConfig, http: Arc<HttpClient>) -> Self {
        let analyzer_base = normalize_base(&config.presidio_analyzer_api_base);
        let anonymizer_base = config
            .presidio_anonymizer_api_base
            .as_deref()
            .map(normalize_base);
        Self {
            config,
            analyzer_base,
            anonymizer_base,
            http,
        }
    }

    fn score_threshold_for(&self, entity_type: &str) -> Option<f64> {
        self.config
            .presidio_score_thresholds
            .get(entity_type)
            .or_else(|| self.config.presidio_score_thresholds.get(ALL_THRESHOLD_KEY))
            .copied()
    }

    fn action_for(&self, entity_type: &str) -> Option<PiiAction> {
        self.config.pii_entities_config.get(entity_type).copied()
    }
}

#[derive(Serialize)]
struct AnalyzeRequest<'a> {
    text: &'a str,
    language: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    entities: Option<Vec<String>>,
    #[serde(skip_serializing_if = "serde_json::Value::is_null")]
    ad_hoc_recognizers: &'a serde_json::Value,
}

#[derive(Deserialize, Serialize, Clone)]
struct AnalyzeResult {
    entity_type: String,
    start: usize,
    end: usize,
    score: f64,
}

#[derive(Serialize)]
struct AnonymizeRequest<'a> {
    text: &'a str,
    analyzer_results: &'a [AnalyzeResult],
}

#[derive(Deserialize)]
struct AnonymizeResponse {
    text: String,
    #[serde(default)]
    items: Vec<AnonymizeItem>,
}

#[derive(Deserialize)]
struct AnonymizeItem {
    entity_type: String,
}

#[async_trait::async_trait]
impl Guardrail for Presidio {
    fn provider_name(&self) -> &'static str {
        "presidio"
    }

    async fn apply(
        &self,
        input: &GuardrailInput,
        _input_type: InputType,
        _ctx: &RequestContext,
    ) -> Result<GuardrailOutcome, ProviderError> {
        let start = Instant::now();

        let language = self.config.presidio_language.as_deref().unwrap_or("en");
        let entities: Option<Vec<String>> = if self.config.pii_entities_config.is_empty() {
            None
        } else {
            Some(self.config.pii_entities_config.keys().cloned().collect())
        };

        let mut masked_texts: Vec<String> = Vec::with_capacity(input.texts.len());
        let mut masked_entity_count: HashMap<String, u32> = HashMap::new();
        let mut any_masked = false;
        let mut last_raw = serde_json::Value::Null;

        for text in &input.texts {
            let analyze_url = format!("{}analyze", self.analyzer_base);
            let resp = self
                .http
                .inner()
                .post(&analyze_url)
                .json(&AnalyzeRequest {
                    text,
                    language,
                    entities: entities.clone(),
                    ad_hoc_recognizers: &self.config.presidio_ad_hoc_recognizers,
                })
                .send()
                .await
                .map_err(super::map_send_error(start))?;

            let raw = super::read_success_json(resp).await?;
            let results: Vec<AnalyzeResult> = serde_json::from_value(raw.clone()).map_err(|e| {
                ProviderError::InvalidResponse {
                    message: format!("presidio analyzer returned invalid response: {e}"),
                }
            })?;
            last_raw = raw;

            let filtered: Vec<AnalyzeResult> = results
                .into_iter()
                .filter(|r| {
                    !self
                        .config
                        .presidio_entities_deny_list
                        .contains(&r.entity_type)
                })
                .filter(|r| match self.score_threshold_for(&r.entity_type) {
                    Some(threshold) => r.score >= threshold,
                    None => true,
                })
                .collect();

            for result in &filtered {
                if self.action_for(&result.entity_type) == Some(PiiAction::Block) {
                    return Ok(GuardrailOutcome {
                        verdict: Verdict::Block {
                            violation_message: format!(
                                "Blocked entity detected: {}. This entity is not allowed to be used in this request.",
                                result.entity_type
                            ),
                            detections: vec![Detection {
                                category: result.entity_type.clone(),
                                label: None,
                                score: Some(result.score),
                                action: Some("BLOCKED".to_owned()),
                            }],
                        },
                        provider_response: last_raw,
                        duration_ms: start.elapsed().as_millis() as u64,
                    });
                }
            }

            if filtered.is_empty() {
                masked_texts.push(text.clone());
                continue;
            }

            let anonymizer_base =
                self.anonymizer_base
                    .as_deref()
                    .ok_or_else(|| ProviderError::InvalidConfig {
                        message: "presidio requires presidio_anonymizer_api_base to mask entities"
                            .to_owned(),
                    })?;
            let anonymize_url = format!("{anonymizer_base}anonymize");

            let resp = self
                .http
                .inner()
                .post(&anonymize_url)
                .json(&AnonymizeRequest {
                    text,
                    analyzer_results: &filtered,
                })
                .send()
                .await
                .map_err(super::map_send_error(start))?;

            let raw = super::read_success_json(resp).await?;
            let anonymized: AnonymizeResponse =
                serde_json::from_value(raw.clone()).map_err(|e| ProviderError::InvalidResponse {
                    message: format!("presidio anonymizer returned invalid response: {e}"),
                })?;

            for item in &anonymized.items {
                *masked_entity_count.entry(item.entity_type.clone()).or_insert(0) += 1;
            }
            any_masked = true;
            masked_texts.push(anonymized.text);
            last_raw = raw;
        }

        let duration_ms = start.elapsed().as_millis() as u64;

        if any_masked {
            return Ok(GuardrailOutcome {
                verdict: Verdict::Mask {
                    texts: masked_texts,
                    masked_entity_count,
                    detections: vec![],
                },
                provider_response: last_raw,
                duration_ms,
            });
        }

        Ok(GuardrailOutcome {
            verdict: Verdict::Pass,
            provider_response: last_raw,
            duration_ms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_url_gets_scheme_and_trailing_slash() {
        assert_eq!(normalize_base("localhost:5002"), "http://localhost:5002/");
        assert_eq!(
            normalize_base("https://presidio.example.com"),
            "https://presidio.example.com/"
        );
        assert_eq!(
            normalize_base("http://localhost:5002/"),
            "http://localhost:5002/"
        );
    }
}
