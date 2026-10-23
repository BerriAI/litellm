use super::TokenizeOutput;

#[macro_rules_attribute::apply(crate::wire_type)]
pub struct TokenizeRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default, skip_serializing_if = "TokenizeOutput::is_available")]
    pub output: TokenizeOutput,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::{TokenizeOutput, TokenizeRequest};

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

    #[rstest]
    #[case::count(TokenizeOutput::Count, "count")]
    #[case::ids(TokenizeOutput::Ids, "ids")]
    #[case::tokens(TokenizeOutput::Tokens, "tokens")]
    #[case::both(TokenizeOutput::Both, "both")]
    #[case::available(TokenizeOutput::Available, "available")]
    fn output_selection_uses_stable_string_values(
        #[case] output: TokenizeOutput,
        #[case] serialized: &str,
    ) {
        let body = json!({"model": "m", "prompt": "", "output": serialized});
        let request: TokenizeRequest = serde_json::from_value(body.clone()).unwrap();

        assert_eq!(request.output, output);
        assert_eq!(
            serde_json::to_value(request).unwrap(),
            if output.is_available() {
                json!({"model": "m", "prompt": ""})
            } else {
                body
            }
        );
    }

    #[rstest]
    #[case::invalid("token_ids")]
    #[case::case_mismatch("COUNT")]
    fn output_selection_rejects_unknown_spellings(#[case] output: &str) {
        assert!(serde_json::from_value::<TokenizeOutput>(json!(output)).is_err());
    }
}
