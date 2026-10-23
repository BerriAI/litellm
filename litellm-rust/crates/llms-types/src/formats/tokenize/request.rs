#[macro_rules_attribute::apply(crate::wire_type)]
pub struct TokenizeRequest {
    pub model: String,
    pub prompt: String,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::TokenizeRequest;

    #[rstest]
    #[case::text("Hello")]
    #[case::empty("")]
    fn prompt_round_trips_without_normalization(#[case] prompt: &str) {
        let body = json!({"model": "model-alias", "prompt": prompt});
        let request: TokenizeRequest = serde_json::from_value(body.clone()).unwrap();
        assert_eq!(serde_json::to_value(request).unwrap(), body);
    }

    #[rstest]
    #[case::missing_model(json!({"prompt": "Hello"}))]
    #[case::missing_prompt(json!({"model": "model-alias"}))]
    #[case::null_prompt(json!({"model": "model-alias", "prompt": null}))]
    #[case::non_text_prompt(json!({"model": "model-alias", "prompt": [1, 2]}))]
    fn malformed_requests_are_rejected(#[case] body: Value) {
        assert!(serde_json::from_value::<TokenizeRequest>(body).is_err());
    }
}
