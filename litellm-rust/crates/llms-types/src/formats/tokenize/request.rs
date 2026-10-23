use serde::{Deserialize, Deserializer};

use super::TokenizeInclude;

#[macro_rules_attribute::apply(crate::wire_type)]
#[serde(deny_unknown_fields)]
pub struct TokenizeRequest {
    #[serde(deserialize_with = "deserialize_model")]
    #[cfg_attr(feature = "schema", schemars(length(min = 1)))]
    pub model: String,
    pub prompt: String,
    #[serde(default, skip_serializing_if = "TokenizeInclude::is_empty")]
    pub include: TokenizeInclude,
    #[serde(
        default,
        deserialize_with = "deserialize_special_tokens",
        skip_serializing_if = "Option::is_none"
    )]
    #[cfg_attr(feature = "schema", schemars(with = "bool"))]
    pub add_special_tokens: Option<bool>,
}

fn deserialize_model<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let model = String::deserialize(deserializer)?;
    if model.is_empty() {
        return Err(serde::de::Error::custom("model must not be empty"));
    }
    Ok(model)
}

fn deserialize_special_tokens<'de, D>(deserializer: D) -> Result<Option<bool>, D::Error>
where
    D: Deserializer<'de>,
{
    bool::deserialize(deserializer).map(Some)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::TokenizeRequest;
    use crate::formats::tokenize::TokenizeOutput;

    #[rstest]
    #[case::text("Hello")]
    #[case::empty("")]
    #[case::unicode("  café 世界\n")]
    fn prompt_round_trips_without_normalization(#[case] prompt: &str) {
        let body = json!({"model": "model-alias", "prompt": prompt});
        let request: TokenizeRequest = serde_json::from_value(body.clone()).unwrap();
        assert_eq!(request.include.output(), TokenizeOutput::Count);
        assert_eq!(request.add_special_tokens, None);
        assert_eq!(serde_json::to_value(request).unwrap(), body);
    }

    #[rstest]
    #[case::missing_model(json!({"prompt": "Hello"}))]
    #[case::empty_model(json!({"model":"", "prompt":""}))]
    #[case::null_model(json!({"model":null, "prompt":""}))]
    #[case::numeric_model(json!({"model":1, "prompt":""}))]
    #[case::missing_prompt(json!({"model": "model-alias"}))]
    #[case::null_prompt(json!({"model": "model-alias", "prompt": null}))]
    #[case::non_text_prompt(json!({"model": "model-alias", "prompt": [1, 2]}))]
    #[case::non_object(json!([]))]
    fn malformed_requests_are_rejected(#[case] body: Value) {
        assert!(serde_json::from_value::<TokenizeRequest>(body).is_err());
    }

    #[rstest]
    #[case::null_special("add_special_tokens", json!(null))]
    #[case::string_special("add_special_tokens", json!("false"))]
    #[case::numeric_special("add_special_tokens", json!(0))]
    #[case::null_include("include", json!(null))]
    #[case::output("output", json!("count"))]
    #[case::messages("messages", json!([]))]
    #[case::tools("tools", json!([]))]
    #[case::images("images", json!([]))]
    #[case::template("chat_template", json!(""))]
    #[case::generation_prompt("add_generation_prompt", json!(false))]
    fn invalid_options_are_rejected(#[case] field: &str, #[case] value: Value) {
        let mut body = json!({"model":"m", "prompt":""});
        body[field] = value;
        assert!(serde_json::from_value::<TokenizeRequest>(body).is_err());
    }

    #[rstest]
    #[case::enabled(true)]
    #[case::disabled(false)]
    fn explicit_special_policy_round_trips(#[case] add_special_tokens: bool) {
        let body = json!({"model":"m", "prompt":"", "include":["tokens"], "add_special_tokens":add_special_tokens});
        let request: TokenizeRequest = serde_json::from_value(body.clone()).unwrap();
        assert_eq!(request.add_special_tokens, Some(add_special_tokens));
        assert_eq!(serde_json::to_value(request).unwrap(), body);
    }

    #[cfg(feature = "schema")]
    #[rstest]
    fn schema_matches_strict_request() {
        let schema = schemars::schema_for!(TokenizeRequest).to_value();
        assert_eq!(schema["additionalProperties"], false);
        assert_eq!(schema["properties"]["model"]["minLength"], 1);
        assert_eq!(
            schema["properties"]["add_special_tokens"]["type"],
            "boolean"
        );
        assert!(schema["properties"].get("output").is_none());
        assert!(
            schema["required"]
                .as_array()
                .unwrap()
                .contains(&json!("model"))
        );
        assert!(
            schema["required"]
                .as_array()
                .unwrap()
                .contains(&json!("prompt"))
        );
        assert_eq!(schema["$defs"]["TokenizeInclude"]["uniqueItems"], true);
        assert_eq!(
            schema["$defs"]["TokenizeRepresentation"]["enum"],
            json!(["token_ids", "tokens"])
        );
    }
}
