use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum Verdict {
    Pass,
    Mask {
        texts: Vec<String>,
        masked_entity_count: HashMap<String, u32>,
        #[serde(default)]
        detections: Vec<Detection>,
    },
    Block {
        violation_message: String,
        #[serde(default)]
        detections: Vec<Detection>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Detection {
    pub category: String,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub score: Option<f64>,
    #[serde(default)]
    pub action: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardrailOutcome {
    pub verdict: Verdict,
    #[serde(default)]
    pub provider_response: serde_json::Value,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuardrailStatus {
    Success,
    GuardrailIntervened,
    GuardrailFailedToRespond,
}

impl GuardrailOutcome {
    pub fn status(&self) -> GuardrailStatus {
        match &self.verdict {
            Verdict::Pass | Verdict::Mask { .. } => GuardrailStatus::Success,
            Verdict::Block { .. } => GuardrailStatus::GuardrailIntervened,
        }
    }
}
