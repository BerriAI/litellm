use serde_json::Value;

#[macro_rules_attribute::apply(crate::wire_type)]
pub struct AudioTranscriptionUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
}

#[macro_rules_attribute::apply(crate::wire_type)]
pub struct AudioTranscriptionResponseData {
    pub text: String,
    pub usage: Option<AudioTranscriptionUsage>,
}

impl AudioTranscriptionResponseData {
    pub fn into_json(self) -> Value {
        match self.usage {
            None => serde_json::json!({"text": self.text}),
            Some(usage) => serde_json::json!({
                "text": self.text,
                "usage": {
                    "type": "tokens",
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "total_tokens": usage.total_tokens,
                },
            }),
        }
    }
}
