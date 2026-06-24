use std::collections::HashMap;
use std::time::Instant;

use litellm_core::guardrails::{
    Detection, GuardrailInput, GuardrailOutcome, InputType, LakeraV2Config, OnFlagged,
    ProviderError, RequestContext, Verdict,
};
use serde::{Deserialize, Serialize};

use super::{map_send_error, read_success_json};
use crate::io::guardrails::http;
use crate::io::guardrails::provider::Guardrail;

const DEFAULT_API_BASE: &str = "https://api.lakera.ai";
const PII_PREFIX: &str = "pii/";

pub struct LakeraV2 {
    config: LakeraV2Config,
}

impl LakeraV2 {
    pub fn new(config: LakeraV2Config) -> Self {
        Self { config }
    }

    fn url(&self) -> String {
        let base = self
            .config
            .api_base
            .as_deref()
            .unwrap_or(DEFAULT_API_BASE)
            .trim_end_matches('/');
        format!("{base}/v2/guard")
    }

    fn api_key(&self) -> Result<&str, ProviderError> {
        self.config
            .api_key
            .as_deref()
            .ok_or_else(|| ProviderError::InvalidConfig {
                message: "lakera_v2 requires api_key or LAKERA_API_KEY".to_owned(),
            })
    }
}

#[derive(Serialize)]
struct GuardRequest<'a> {
    messages: Vec<GuardMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    project_id: &'a Option<String>,
    payload: bool,
    breakdown: bool,
    #[serde(skip_serializing_if = "serde_json::Value::is_null")]
    metadata: &'a serde_json::Value,
    dev_info: bool,
}

#[derive(Serialize)]
struct GuardMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize)]
struct GuardResponse {
    #[serde(default)]
    flagged: bool,
    #[serde(default)]
    payload: Vec<PayloadItem>,
    #[serde(default)]
    breakdown: Vec<BreakdownItem>,
}

#[derive(Deserialize)]
struct PayloadItem {
    start: usize,
    end: usize,
    #[serde(default)]
    detector_type: String,
    #[serde(default)]
    message_id: usize,
}

#[derive(Deserialize)]
struct BreakdownItem {
    #[serde(default)]
    detector_type: String,
    #[serde(default)]
    detected: bool,
}

fn is_only_pii_violation(breakdown: &[BreakdownItem]) -> bool {
    let detected: Vec<&BreakdownItem> = breakdown.iter().filter(|b| b.detected).collect();
    !detected.is_empty()
        && detected
            .iter()
            .all(|b| b.detector_type.starts_with(PII_PREFIX))
}

fn mask_label(detector_type: &str) -> String {
    let entity = detector_type
        .strip_prefix(PII_PREFIX)
        .unwrap_or(detector_type)
        .to_ascii_uppercase();
    format!("[MASKED {entity}]")
}

fn mask_texts(texts: &[String], payload: &[PayloadItem]) -> (Vec<String>, HashMap<String, u32>) {
    let mut masked: Vec<String> = texts.to_vec();
    let mut counts: HashMap<String, u32> = HashMap::new();

    let mut by_message: HashMap<usize, Vec<&PayloadItem>> = HashMap::new();
    for item in payload {
        if item.detector_type.starts_with(PII_PREFIX) {
            by_message.entry(item.message_id).or_default().push(item);
        }
    }

    for (msg_id, mut items) in by_message {
        let Some(text) = masked.get(msg_id) else {
            continue;
        };
        items.sort_by_key(|item| std::cmp::Reverse(item.start));

        let chars: Vec<char> = text.chars().collect();
        let mut out = chars.clone();
        for item in items {
            if item.start >= item.end || item.end > out.len() {
                continue;
            }
            let label: Vec<char> = mask_label(&item.detector_type).chars().collect();
            out.splice(item.start..item.end, label);
            *counts.entry(item.detector_type.clone()).or_insert(0) += 1;
        }
        masked[msg_id] = out.into_iter().collect();
    }

    (masked, counts)
}

fn detections_from_breakdown(breakdown: &[BreakdownItem]) -> Vec<Detection> {
    breakdown
        .iter()
        .filter(|b| b.detected)
        .map(|b| Detection {
            category: b.detector_type.clone(),
            label: None,
            score: None,
            action: Some("DETECTED".to_owned()),
        })
        .collect()
}

impl Guardrail for LakeraV2 {
    fn apply(
        &self,
        input: &GuardrailInput,
        _input_type: InputType,
        _ctx: &RequestContext,
    ) -> Result<GuardrailOutcome, ProviderError> {
        let start = Instant::now();
        let api_key = self.api_key()?;

        let messages: Vec<GuardMessage> = input
            .texts
            .iter()
            .map(|t| GuardMessage {
                role: "user",
                content: t,
            })
            .collect();

        let body = GuardRequest {
            messages,
            project_id: &self.config.project_id,
            payload: self.config.payload.unwrap_or(true),
            breakdown: self.config.breakdown.unwrap_or(true),
            metadata: &self.config.metadata,
            dev_info: self.config.dev_info.unwrap_or(true),
        };

        let resp = http::client()
            .post(self.url())
            .bearer_auth(api_key)
            .json(&body)
            .send()
            .map_err(map_send_error(start))?;

        let raw = read_success_json(resp)?;
        let parsed: GuardResponse =
            serde_json::from_value(raw.clone()).map_err(|e| ProviderError::InvalidResponse {
                message: e.to_string(),
            })?;

        let duration_ms = start.elapsed().as_millis() as u64;

        if !parsed.flagged {
            return Ok(GuardrailOutcome {
                verdict: Verdict::Pass,
                provider_response: raw,
                duration_ms,
            });
        }

        if is_only_pii_violation(&parsed.breakdown) {
            let (texts, masked_entity_count) = mask_texts(&input.texts, &parsed.payload);
            return Ok(GuardrailOutcome {
                verdict: Verdict::Mask {
                    texts,
                    masked_entity_count,
                    detections: detections_from_breakdown(&parsed.breakdown),
                },
                provider_response: raw,
                duration_ms,
            });
        }

        if self.config.on_flagged == Some(OnFlagged::Monitor) {
            return Ok(GuardrailOutcome {
                verdict: Verdict::Pass,
                provider_response: raw,
                duration_ms,
            });
        }

        Ok(GuardrailOutcome {
            verdict: Verdict::Block {
                violation_message: "Violated guardrail policy".to_owned(),
                detections: detections_from_breakdown(&parsed.breakdown),
            },
            provider_response: raw,
            duration_ms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masks_pii_span_with_label() {
        let texts = vec!["my card is 4111111111111111 ok".to_owned()];
        let payload = vec![PayloadItem {
            start: 11,
            end: 27,
            detector_type: "pii/credit_card".to_owned(),
            message_id: 0,
        }];
        let (masked, counts) = mask_texts(&texts, &payload);
        assert_eq!(masked[0], "my card is [MASKED CREDIT_CARD] ok");
        assert_eq!(counts.get("pii/credit_card"), Some(&1));
    }

    #[test]
    fn pii_only_detection_logic() {
        let only_pii = vec![BreakdownItem {
            detector_type: "pii/ssn".to_owned(),
            detected: true,
        }];
        assert!(is_only_pii_violation(&only_pii));

        let mixed = vec![
            BreakdownItem {
                detector_type: "pii/ssn".to_owned(),
                detected: true,
            },
            BreakdownItem {
                detector_type: "prompt_attack/jailbreak".to_owned(),
                detected: true,
            },
        ];
        assert!(!is_only_pii_violation(&mixed));

        let none_detected = vec![BreakdownItem {
            detector_type: "pii/ssn".to_owned(),
            detected: false,
        }];
        assert!(!is_only_pii_violation(&none_detected));
    }
}
