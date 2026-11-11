use serde_json::{Map, Value};

use super::ChoiceValue;

#[macro_rules_attribute::apply(crate::wire_type)]
pub struct DecisionsResponse {
    pub model: String,
    pub answers: Vec<DecisionAnswer>,
    pub usage: DecisionsUsage,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type")]
pub enum DecisionAnswer {
    #[serde(rename = "predicate")]
    Predicate {
        #[serde(default)]
        name: Option<String>,
        probability: f64,
    },
    #[serde(rename = "choice")]
    Choice {
        #[serde(default)]
        name: Option<String>,
        choice: ChoiceValue,
        probabilities: Vec<ChoiceProbability>,
        #[serde(default)]
        confidence: Option<f64>,
    },
    #[serde(rename = "score")]
    Score {
        #[serde(default)]
        name: Option<String>,
        score: f64,
        probabilities: Vec<ScoreProbability>,
        #[serde(default)]
        confidence: Option<f64>,
    },
    #[serde(rename = "box")]
    Box {
        #[serde(default)]
        name: Option<String>,
        #[serde(rename = "box")]
        region: DecisionBox,
        #[serde(default)]
        confidence: Option<f64>,
    },
    #[serde(rename = "refusal")]
    Refusal {
        #[serde(default)]
        name: Option<String>,
    },
}

#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ChoiceProbability {
    pub value: ChoiceValue,
    pub probability: f64,
}

#[macro_rules_attribute::apply(crate::wire_type)]
pub struct ScoreProbability {
    pub value: i64,
    pub label: String,
    pub probability: f64,
}

#[macro_rules_attribute::apply(crate::wire_type)]
pub struct DecisionBox {
    pub top: f64,
    pub bottom: f64,
    pub left: f64,
    pub right: f64,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Copy, Default, Eq)]
pub struct DecisionsUsage {
    pub input_tokens: u64,
    #[serde(default)]
    pub input_tokens_details: DecisionsInputTokensDetails,
    pub output_tokens: u64,
    #[serde(default)]
    pub output_tokens_details: DecisionsOutputTokensDetails,
    pub total_tokens: u64,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Copy, Default, Eq)]
pub struct DecisionsInputTokensDetails {
    #[serde(default)]
    pub cached_tokens: u64,
    #[serde(default)]
    pub cache_write_tokens: u64,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Copy, Default, Eq)]
pub struct DecisionsOutputTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: u64,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    fn a_response_round_trips_every_answer_type() {
        let body = json!({
            "model": "gpt-6-luna",
            "answers": [
                {"type": "predicate", "name": "refund", "probability": 0.9},
                {"type": "choice", "name": null, "choice": true,
                 "probabilities": [{"value": true, "probability": 0.7}], "confidence": 0.7},
                {"type": "score", "name": "mood", "score": 1.5,
                 "probabilities": [{"value": 1, "label": "angry", "probability": 0.5}], "confidence": null},
                {"type": "box", "name": "mug", "box": {"top": 0.1, "bottom": 0.2, "left": 0.3, "right": 0.4, "unit": "fraction"}, "confidence": 0.8},
                {"type": "refusal", "name": "other"},
            ],
            "usage": {"input_tokens": 10, "input_tokens_details": {"cached_tokens": 4, "cache_write_tokens": 0},
                      "output_tokens": 2, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 12},
        });

        let response: DecisionsResponse = serde_json::from_value(body.clone()).unwrap();

        assert_eq!(serde_json::to_value(&response).unwrap(), body);
    }

    #[rstest]
    fn usage_details_default_to_zero() {
        let usage: DecisionsUsage = serde_json::from_value(
            json!({"input_tokens": 3, "output_tokens": 1, "total_tokens": 4}),
        )
        .unwrap();

        assert_eq!(
            usage.input_tokens_details,
            DecisionsInputTokensDetails::default()
        );
        assert_eq!(usage.output_tokens_details.reasoning_tokens, 0);
    }
}
