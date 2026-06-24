//! In-process PII detection and masking.
//!
//! Compiled once from a [`LocalPiiConfig`] and reused across requests, so no
//! regex is recompiled on the hot path. Detection runs a combined [`RegexSet`]
//! pre-filter first: clean text (the common case) costs a single pass instead
//! of one per recognizer, and only recognizers the set says can match are
//! re-run to recover their spans.

use std::collections::{HashMap, HashSet};

use regex::{Regex, RegexSet};

use super::config::{LocalPiiConfig, PiiAction};
use super::verdict::{Detection, Verdict};

const CUSTOM_ENTITY: &str = "CUSTOM";
const DEFAULT_MASK_TOKEN: &str = "<{entity}>";

struct Builtin {
    entity_type: &'static str,
    pattern: &'static str,
    validator: Option<fn(&str) -> bool>,
}

const BUILTINS: &[Builtin] = &[
    Builtin {
        entity_type: "EMAIL_ADDRESS",
        pattern: r"(?i)\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b",
        validator: None,
    },
    Builtin {
        entity_type: "US_SSN",
        pattern: r"\b\d{3}-\d{2}-\d{4}\b",
        validator: None,
    },
    Builtin {
        entity_type: "CREDIT_CARD",
        pattern: r"\b\d(?:[ -]?\d){12,18}\b",
        validator: Some(luhn_valid),
    },
    Builtin {
        entity_type: "PHONE_NUMBER",
        pattern: r"(?:\+?\d{1,3}[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}",
        validator: None,
    },
    Builtin {
        entity_type: "IP_ADDRESS",
        pattern: r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b",
        validator: None,
    },
];

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
    sum.is_multiple_of(10)
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

struct Recognizer {
    entity_type: &'static str,
    regex: Regex,
    validator: Option<fn(&str) -> bool>,
    action: PiiAction,
}

struct Span {
    start: usize,
    end: usize,
    entity: &'static str,
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

pub struct LocalPiiEngine {
    recognizers: Vec<Recognizer>,
    /// Combined set over `recognizers`, same order. One pass tells us which
    /// recognizers can match before we re-run them to recover spans.
    recognizer_set: RegexSet,
    deny_regex: Option<Regex>,
    deny_action: PiiAction,
    allow_set: HashSet<String>,
    mask_token: String,
}

impl LocalPiiEngine {
    pub fn new(config: LocalPiiConfig) -> Self {
        let recognizers: Vec<Recognizer> = BUILTINS
            .iter()
            .filter_map(|b| {
                let action = if config.pii_entities_config.is_empty() {
                    PiiAction::Mask
                } else {
                    *config.pii_entities_config.get(b.entity_type)?
                };
                Some(Recognizer {
                    entity_type: b.entity_type,
                    regex: Regex::new(b.pattern).expect("builtin recognizer pattern is valid"),
                    validator: b.validator,
                    action,
                })
            })
            .collect();

        let recognizer_set = RegexSet::new(recognizers.iter().map(|r| r.regex.as_str()))
            .unwrap_or_else(|_| RegexSet::empty());

        Self {
            recognizers,
            recognizer_set,
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

        for idx in self.recognizer_set.matches(text).into_iter() {
            let recognizer = &self.recognizers[idx];
            for m in recognizer.regex.find_iter(text) {
                let passes_validator = recognizer.validator.is_none_or(|v| v(m.as_str()));
                if passes_validator && !self.is_allowed(m.as_str()) {
                    spans.push(Span {
                        start: m.start(),
                        end: m.end(),
                        entity: recognizer.entity_type,
                        action: recognizer.action,
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
                        entity: CUSTOM_ENTITY,
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
            out.push_str(&self.render_mask(span.entity));
            *counts.entry(span.entity.to_owned()).or_insert(0) += 1;
            cursor = span.end;
        }
        out.push_str(&text[cursor..]);
        out
    }

    /// Screen and mask a batch of texts. Consumes `texts` so unchanged entries
    /// move through without a copy; a clean batch returns [`Verdict::Pass`]
    /// without allocating a result list at all.
    pub fn scan_texts(&self, texts: Vec<String>) -> Verdict {
        let mut masked_texts: Vec<String> = Vec::with_capacity(texts.len());
        let mut masked_entity_count: HashMap<String, u32> = HashMap::new();
        let mut any_masked = false;

        for text in texts {
            let spans = self.detect(&text);

            if let Some(blocked) = spans.iter().find(|s| s.action == PiiAction::Block) {
                return Verdict::Block {
                    violation_message: format!(
                        "Blocked entity detected: {}. This entity is not allowed to be used in this request.",
                        blocked.entity
                    ),
                    detections: vec![Detection {
                        category: blocked.entity.to_owned(),
                        label: None,
                        score: None,
                        action: Some("BLOCKED".to_owned()),
                    }],
                };
            }

            if spans.is_empty() {
                masked_texts.push(text);
                continue;
            }

            any_masked = true;
            masked_texts.push(self.mask_text(&text, &spans, &mut masked_entity_count));
        }

        if any_masked {
            Verdict::Mask {
                texts: masked_texts,
                masked_entity_count,
                detections: vec![],
            }
        } else {
            Verdict::Pass
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply(config: LocalPiiConfig, text: &str) -> Verdict {
        LocalPiiEngine::new(config).scan_texts(vec![text.to_owned()])
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
                assert!(!masked_entity_count.contains_key("PHONE_NUMBER"));
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

    #[test]
    fn unchanged_texts_are_preserved_alongside_masked_ones() {
        let verdict = LocalPiiEngine::new(LocalPiiConfig::default()).scan_texts(vec![
            "clean line".to_owned(),
            "mail me jane.doe@example.com".to_owned(),
        ]);
        match verdict {
            Verdict::Mask { texts, .. } => {
                assert_eq!(texts[0], "clean line");
                assert_eq!(texts[1], "mail me <EMAIL_ADDRESS>");
            }
            other => panic!("expected mask, got {other:?}"),
        }
    }
}
