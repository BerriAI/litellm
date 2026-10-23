#[macro_rules_attribute::apply(crate::wire_type)]
pub struct TokenizeResponse {
    pub model: String,
    pub count: u64,
    pub token_ids: Option<Vec<u64>>,
    pub tokens: Option<Vec<String>>,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::TokenizeResponse;

    #[rstest]
    fn unavailable_token_ids_are_null_and_strings_remain_strings() {
        let response = TokenizeResponse {
            model: "model".into(),
            count: 2,
            token_ids: None,
            tokens: Some(vec!["Hello".into(), " world".into()]),
        };
        assert_eq!(
            serde_json::to_value(response).unwrap(),
            json!({"model": "model", "count": 2, "token_ids": null, "tokens": ["Hello", " world"]})
        );
    }

    #[rstest]
    #[case::negative_count(json!({"model": "m", "count": -1}))]
    #[case::string_ids(json!({"model": "m", "count": 1, "token_ids": ["hello"]}))]
    #[case::numeric_strings(json!({"model": "m", "count": 1, "tokens": [123]}))]
    fn token_representations_have_fixed_types(#[case] body: Value) {
        assert!(serde_json::from_value::<TokenizeResponse>(body).is_err());
    }
}
