use serde::{Deserialize, Serialize};

/// One or more wire-format messages produced by a realtime transform.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RealtimeTransformResult {
    pub messages: Vec<String>,
}

impl RealtimeTransformResult {
    /// Forward a single message unchanged (the OpenAI baseline).
    pub fn passthrough(message: &str) -> Self {
        Self {
            messages: vec![message.to_string()],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_produces_single_element_vec() {
        let result = RealtimeTransformResult::passthrough("hello");
        assert_eq!(result.messages, vec!["hello".to_string()]);
    }
}
