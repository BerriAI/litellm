use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;
use std::time::Instant;

use crate::{
    Detection, Guardrail, GuardrailInput, GuardrailOutcome, InputType, LocalPiiConfig, PiiAction,
    ProviderError, RequestContext, Verdict,
};
use regex::Regex;

const CUSTOM_ENTITY: &str = "CUSTOM";
const DEFAULT_MASK_TOKEN: &str = "<{entity}>";

struct Recognizer {
    entity_type: &'static str,
    regex: Regex,
    validator: Option<fn(&str) -> bool>,
}

static BUILTINS: LazyLock<Vec<Recognizer>> = LazyLock::new(|| {
    vec![
        Recognizer {
            entity_type: "EMAIL_ADDRESS",
            regex: Regex::new(r"(?i)\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b").unwrap(),
            validator: None,
        },
        Recognizer {
            entity_type: "US_SSN",
            regex: Regex::new(r"\b\d{3}-\d{2}-\d{4}\b").unwrap(),
            validator: None,
        },
        Recognizer {
            entity_type: "CREDIT_CARD",
            regex: Regex::new(r"\b\d(?:[ -]?\d){12,18}\b").unwrap(),
            validator: Some(luhn_valid),
        },
        Recognizer {
            entity_type: "PHONE_NUMBER",
            regex: Regex::new(r"(?:\+?\d{1,3}[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}")
                .unwrap(),
            validator: None,
        },
        Recognizer {
            entity_type: "IP_ADDRESS",
            regex: Regex::new(
                r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b",
            )
            .unwrap(),
            validator: None,
        },
    ]
});

fn luhn_valid(candidate: &str) -> bool {
    let digits: Vec<u32> = candidate
        .bytes()
        .filter(u8::is_ascii_digit)
        .map(|b| u32::from(b - b'0'))
        .collect();
    if !(13..=19).contains(&digits.len()) {
        return false;
    }
    let sum: u32 = digits
        .iter()
        .rev()
        .enumerate()
        .map(|(i, &d)| {
            if i % 2 == 0 {
                d
            } else if d * 2 > 9 {
                d * 2 - 9
            } else {
                d * 2
            }
        })
        .sum();
    sum % 10 == 0
}

fn build_deny_regex(terms: &[String]) -> Option<Regex> {
    let alternation = terms
        .iter()
        .filter(|t| !t.is_empty())
        .map(|t| regex::escape(t))
        .collect::<Vec<_>>()
        .join("|");
    if alternation.is_empty() {
        return None;
    }
    Regex::new(&format!("(?i)({alternation})")).ok()
}

struct Span {
    start: usize,
    end: usize,
    entity: String,
    action: PiiAction,
}

fn resolve_overlaps(mut spans: Vec<Span>) -> Vec<Span> {
    spans.sort_by(|a, b| a.start.cmp(&b.start).then(b.end.cmp(&a.end)));
    let mut kept: Vec<Span> = Vec::with_capacity(spans.len());
    let mut last_end = 0usize;
    for span in spans {
        if span.start >= last_end {
            last_end = span.end;
            kept.push(span);
        }
    }
    kept
}

pub struct LocalPii {
    entity_actions: HashMap<&'static str, PiiAction>,
    deny_regex: Option<Regex>,
    deny_action: PiiAction,
    allow_set: HashSet<String>,
    mask_token: String,
}

impl LocalPii {
    pub fn new(config: LocalPiiConfig) -> Self {
        let entity_actions = if config.pii_entities_config.is_empty() {
            BUILTINS
                .iter()
                .map(|r| (r.entity_type, PiiAction::Mask))
                .collect()
        } else {
            BUILTINS
                .iter()
                .filter_map(|r| {
                    config
                        .pii_entities_config
                        .get(r.entity_type)
                        .map(|action| (r.entity_type, *action))
                })
                .collect()
        };

        Self {
            entity_actions,
            deny_regex: build_deny_regex(&config.deny_list),
            deny_action: config.deny_list_action,
            allow_set: config.allow_list.iter().map(|s| s.to_lowercase()).collect(),
            mask_token: config
                .mask_token
                .unwrap_or_else(|| DEFAULT_MASK_TOKEN.to_owned()),
        }
    }

    fn is_allowed(&self, matched: &str) -> bool {
        !self.allow_set.is_empty() && self.allow_set.contains(&matched.to_lowercase())
    }

    fn detect(&self, text: &str) -> Vec<Span> {
        let mut spans: Vec<Span> = Vec::new();

        for recognizer in BUILTINS.iter() {
            let Some(&action) = self.entity_actions.get(recognizer.entity_type) else {
                continue;
            };
            for m in recognizer.regex.find_iter(text) {
                let passes_validator = recognizer.validator.is_none_or(|v| v(m.as_str()));
                if passes_validator && !self.is_allowed(m.as_str()) {
                    spans.push(Span {
                        start: m.start(),
                        end: m.end(),
                        entity: recognizer.entity_type.to_owned(),
                        action,
                    });
                }
            }
        }

        if let Some(deny) = &self.deny_regex {
            for m in deny.find_iter(text) {
                if !self.is_allowed(m.as_str()) {
                    spans.push(Span {
                        start: m.start(),
                        end: m.end(),
                        entity: CUSTOM_ENTITY.to_owned(),
                        action: self.deny_action,
                    });
                }
            }
        }

        resolve_overlaps(spans)
    }

    fn render_mask(&self, entity: &str) -> String {
        self.mask_token.replace("{entity}", entity)
    }

    fn mask_text(&self, text: &str, spans: &[Span], counts: &mut HashMap<String, u32>) -> String {
        let mut out = String::with_capacity(text.len());
        let mut cursor = 0usize;
        for span in spans {
            out.push_str(&text[cursor..span.start]);
            out.push_str(&self.render_mask(&span.entity));
            *counts.entry(span.entity.clone()).or_insert(0) += 1;
            cursor = span.end;
        }
        out.push_str(&text[cursor..]);
        out
    }
}

#[async_trait::async_trait]
impl Guardrail for LocalPii {
    fn provider_name(&self) -> &'static str {
        "local_pii"
    }

    async fn apply(
        &self,
        input: &GuardrailInput,
        _input_type: InputType,
        _ctx: &RequestContext,
    ) -> Result<GuardrailOutcome, ProviderError> {
        let start = Instant::now();
        let mut masked_texts: Vec<String> = Vec::with_capacity(input.texts.len());
        let mut masked_entity_count: HashMap<String, u32> = HashMap::new();
        let mut any_masked = false;

        for text in &input.texts {
            let spans = self.detect(text);

            if let Some(blocked) = spans.iter().find(|s| s.action == PiiAction::Block) {
                return Ok(GuardrailOutcome {
                    verdict: Verdict::Block {
                        violation_message: format!(
                            "Blocked entity detected: {}. This entity is not allowed to be used in this request.",
                            blocked.entity
                        ),
                        detections: vec![Detection {
                            category: blocked.entity.clone(),
                            label: None,
                            score: None,
                            action: Some("BLOCKED".to_owned()),
                        }],
                    },
                    provider_response: serde_json::json!({"engine": "local_pii"}),
                    duration_ms: start.elapsed().as_millis() as u64,
                });
            }

            if spans.is_empty() {
                masked_texts.push(text.clone());
                continue;
            }

            any_masked = true;
            masked_texts.push(self.mask_text(text, &spans, &mut masked_entity_count));
        }

        let duration_ms = start.elapsed().as_millis() as u64;
        let provider_response = serde_json::json!({
            "engine": "local_pii",
            "masked_entity_count": &masked_entity_count,
        });

        let verdict = if any_masked {
            Verdict::Mask {
                texts: masked_texts,
                masked_entity_count,
                detections: vec![],
            }
        } else {
            Verdict::Pass
        };

        Ok(GuardrailOutcome {
            verdict,
            provider_response,
            duration_ms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply(config: LocalPiiConfig, text: &str) -> Verdict {
        let provider = LocalPii::new(config);
        let input = GuardrailInput {
            texts: vec![text.to_owned()],
            ..Default::default()
        };
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(provider.apply(&input, InputType::Request, &RequestContext::default()))
            .unwrap()
            .verdict
    }

    #[test]
    fn luhn_accepts_valid_card_and_rejects_typo() {
        assert!(luhn_valid("4111111111111111"));
        assert!(!luhn_valid("4111111111111112"));
        assert!(!luhn_valid("123456789"));
    }

    #[test]
    fn masks_email_with_default_token() {
        let verdict = apply(
            LocalPiiConfig::default(),
            "reach me at jane.doe@example.com please",
        );
        match verdict {
            Verdict::Mask {
                texts,
                masked_entity_count,
                ..
            } => {
                assert_eq!(texts[0], "reach me at <EMAIL_ADDRESS> please");
                assert_eq!(masked_entity_count.get("EMAIL_ADDRESS"), Some(&1));
            }
            other => panic!("expected mask, got {other:?}"),
        }
    }

    #[test]
    fn allow_list_exempts_matching_value() {
        let config = LocalPiiConfig {
            allow_list: vec!["support@example.com".to_owned()],
            ..Default::default()
        };
        assert_eq!(apply(config, "email support@example.com"), Verdict::Pass);
    }

    #[test]
    fn blocks_entity_configured_as_block() {
        let config = LocalPiiConfig {
            pii_entities_config: HashMap::from([("US_SSN".to_owned(), PiiAction::Block)]),
            ..Default::default()
        };
        match apply(config, "ssn 123-45-6789") {
            Verdict::Block { detections, .. } => {
                assert_eq!(detections[0].category, "US_SSN");
            }
            other => panic!("expected block, got {other:?}"),
        }
    }

    #[test]
    fn only_configured_entities_are_detected() {
        let config = LocalPiiConfig {
            pii_entities_config: HashMap::from([("EMAIL_ADDRESS".to_owned(), PiiAction::Mask)]),
            ..Default::default()
        };
        // SSN present but not configured, so it passes through untouched.
        assert_eq!(apply(config, "ssn 123-45-6789"), Verdict::Pass);
    }

    #[test]
    fn deny_list_masks_custom_term_with_custom_token() {
        let config = LocalPiiConfig {
            deny_list: vec!["Project Aurora".to_owned()],
            mask_token: Some("[REDACTED]".to_owned()),
            ..Default::default()
        };
        match apply(config, "ship Project Aurora now") {
            Verdict::Mask {
                texts,
                masked_entity_count,
                ..
            } => {
                assert_eq!(texts[0], "ship [REDACTED] now");
                assert_eq!(masked_entity_count.get("CUSTOM"), Some(&1));
            }
            other => panic!("expected mask, got {other:?}"),
        }
    }

    #[test]
    fn credit_card_prefers_longer_span_over_phone() {
        match apply(LocalPiiConfig::default(), "card 4111 1111 1111 1111 end") {
            Verdict::Mask {
                texts,
                masked_entity_count,
                ..
            } => {
                assert_eq!(texts[0], "card <CREDIT_CARD> end");
                assert_eq!(masked_entity_count.get("CREDIT_CARD"), Some(&1));
                assert!(masked_entity_count.get("PHONE_NUMBER").is_none());
            }
            other => panic!("expected mask, got {other:?}"),
        }
    }

    #[test]
    fn clean_text_passes() {
        assert_eq!(
            apply(LocalPiiConfig::default(), "nothing to see here"),
            Verdict::Pass
        );
    }
}
