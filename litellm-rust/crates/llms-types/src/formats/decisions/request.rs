use serde_json::Value;

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(untagged)]
pub enum DecisionsInput {
    Text(String),
    Messages(Vec<DecisionsInputMessage>),
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Copy, Default, Eq)]
pub enum DecisionsRole {
    #[default]
    #[serde(rename = "user")]
    User,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[derive(Copy, Default, Eq)]
pub enum DecisionsMessageType {
    #[default]
    #[serde(rename = "message")]
    Message,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(deny_unknown_fields)]
pub struct DecisionsInputMessage {
    #[serde(default)]
    pub role: DecisionsRole,
    #[serde(rename = "type", default)]
    pub kind: DecisionsMessageType,
    pub content: DecisionsMessageContent,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(untagged)]
pub enum DecisionsMessageContent {
    Text(String),
    Parts(Vec<DecisionsContentPart>),
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", deny_unknown_fields)]
pub enum DecisionsContentPart {
    #[serde(rename = "input_text")]
    InputText { text: String },
    #[serde(rename = "input_image")]
    InputImage {
        image_url: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(untagged)]
pub enum ChoiceValue {
    Bool(bool),
    Text(String),
}

impl ChoiceValue {
    pub fn key(&self) -> &str {
        match self {
            Self::Bool(true) => "true",
            Self::Bool(false) => "false",
            Self::Text(text) => text,
        }
    }
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(deny_unknown_fields)]
pub struct DecisionChoice {
    pub value: ChoiceValue,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(deny_unknown_fields)]
pub struct DecisionLevel {
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(tag = "type", deny_unknown_fields)]
pub enum DecisionQuestion {
    #[serde(rename = "predicate")]
    Predicate {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        instructions: String,
    },
    #[serde(rename = "choice")]
    Choice {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        instructions: String,
        choices: Vec<DecisionChoice>,
    },
    #[serde(rename = "score")]
    Score {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        instructions: String,
        levels: Vec<DecisionLevel>,
    },
    #[serde(rename = "box")]
    Box {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        instructions: String,
    },
}

impl DecisionQuestion {
    pub fn name(&self) -> Option<&str> {
        match self {
            Self::Predicate { name, .. }
            | Self::Choice { name, .. }
            | Self::Score { name, .. }
            | Self::Box { name, .. } => name.as_deref(),
        }
    }
}

#[macro_rules_attribute::apply(crate::wire_type)]
pub struct DecisionsRequest {
    pub input: DecisionsInput,
    pub questions: Vec<DecisionQuestion>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub images: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub videos: Option<Value>,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    fn a_request_reads_text_and_message_input() {
        let request: DecisionsRequest = serde_json::from_value(json!({
            "model": "luna",
            "input": [{"content": [
                {"type": "input_text", "text": "hi"},
                {"type": "input_image", "image_url": "https://x/a.png"},
            ]}],
            "questions": [
                {"type": "predicate", "instructions": "Refund?"},
                {"type": "choice", "name": "urgency", "instructions": "How urgent?",
                 "choices": [{"value": "low"}, {"value": true, "description": "now"}]},
                {"type": "score", "instructions": "Mood?", "levels": [{"label": "calm"}, {"label": "angry"}]},
            ],
        }))
        .unwrap();

        assert_eq!(
            serde_json::to_value(&request).unwrap(),
            json!({
                "input": [{"role": "user", "type": "message", "content": [
                    {"type": "input_text", "text": "hi"},
                    {"type": "input_image", "image_url": "https://x/a.png"},
                ]}],
                "questions": [
                    {"type": "predicate", "instructions": "Refund?"},
                    {"type": "choice", "name": "urgency", "instructions": "How urgent?",
                     "choices": [{"value": "low"}, {"value": true, "description": "now"}]},
                    {"type": "score", "instructions": "Mood?", "levels": [{"label": "calm"}, {"label": "angry"}]},
                ],
            })
        );
    }

    #[rstest]
    #[case::unknown_question_field(json!({"input": "x", "questions": [{"type": "predicate", "instructions": "q", "extra": 1}]}))]
    #[case::unknown_question_type(json!({"input": "x", "questions": [{"type": "rank", "instructions": "q"}]}))]
    #[case::missing_instructions(json!({"input": "x", "questions": [{"type": "predicate"}]}))]
    #[case::unknown_choice_field(json!({"input": "x", "questions": [{"type": "choice", "instructions": "q", "choices": [{"value": "a", "weight": 1}]}]}))]
    #[case::assistant_message(json!({"input": [{"role": "assistant", "content": "x"}], "questions": []}))]
    #[case::unknown_message_field(json!({"input": [{"content": "x", "name": "n"}], "questions": []}))]
    #[case::unknown_part(json!({"input": [{"content": [{"type": "input_file", "file_id": "f"}]}], "questions": []}))]
    #[case::numeric_input(json!({"input": 3, "questions": []}))]
    #[case::no_questions_key(json!({"input": "x"}))]
    fn malformed_requests_are_refused(#[case] body: Value) {
        assert!(serde_json::from_value::<DecisionsRequest>(body).is_err());
    }

    #[rstest]
    #[case::text(ChoiceValue::Text("low".into()), "low")]
    #[case::true_(ChoiceValue::Bool(true), "true")]
    #[case::false_(ChoiceValue::Bool(false), "false")]
    fn a_choice_is_keyed_by_its_text(#[case] value: ChoiceValue, #[case] key: &str) {
        assert_eq!(value.key(), key);
    }
}
