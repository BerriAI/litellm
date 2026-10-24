use serde::{Deserialize, Deserializer, Serialize};
use strum::VariantArray;

use super::TokenizeOutput;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, VariantArray)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum TokenizeRepresentation {
    TokenIds,
    Tokens,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(transparent)]
pub struct TokenizeInclude(
    #[serde(deserialize_with = "deserialize_unique")]
    #[cfg_attr(
        feature = "schema",
        schemars(with = "std::collections::HashSet<TokenizeRepresentation>")
    )]
    Vec<TokenizeRepresentation>,
);

impl TokenizeInclude {
    pub fn new(token_ids: bool, tokens: bool) -> Self {
        Self(
            [
                token_ids.then_some(TokenizeRepresentation::TokenIds),
                tokens.then_some(TokenizeRepresentation::Tokens),
            ]
            .into_iter()
            .flatten()
            .collect(),
        )
    }

    pub fn output(&self) -> TokenizeOutput {
        match (
            self.0.contains(&TokenizeRepresentation::TokenIds),
            self.0.contains(&TokenizeRepresentation::Tokens),
        ) {
            (false, false) => TokenizeOutput::Count,
            (true, false) => TokenizeOutput::Ids,
            (false, true) => TokenizeOutput::Tokens,
            (true, true) => TokenizeOutput::Both,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

fn deserialize_unique<'de, D>(deserializer: D) -> Result<Vec<TokenizeRepresentation>, D::Error>
where
    D: Deserializer<'de>,
{
    let values = Vec::<TokenizeRepresentation>::deserialize(deserializer)?;
    if values
        .iter()
        .enumerate()
        .any(|(index, value)| values[..index].contains(value))
    {
        return Err(serde::de::Error::custom(
            "include must not contain duplicate entries",
        ));
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::{TokenizeInclude, TokenizeOutput};

    #[rstest]
    #[case::count(json!([]), TokenizeOutput::Count)]
    #[case::ids(json!(["token_ids"]), TokenizeOutput::Ids)]
    #[case::tokens(json!(["tokens"]), TokenizeOutput::Tokens)]
    #[case::both(json!(["token_ids", "tokens"]), TokenizeOutput::Both)]
    #[case::reversed(json!(["tokens", "token_ids"]), TokenizeOutput::Both)]
    fn include_round_trips(#[case] body: Value, #[case] output: TokenizeOutput) {
        let include: TokenizeInclude = serde_json::from_value(body.clone()).unwrap();
        assert_eq!(include.output(), output);
        assert_eq!(serde_json::to_value(include).unwrap(), body);
    }

    #[rstest]
    #[case::duplicate_ids(json!(["token_ids", "token_ids"]))]
    #[case::duplicate_tokens(json!(["tokens", "tokens"]))]
    #[case::unknown(json!(["ids"]))]
    #[case::wrong_case(json!(["Tokens"]))]
    #[case::number(json!([1]))]
    #[case::null_item(json!([null]))]
    #[case::null(json!(null))]
    #[case::string(json!("tokens"))]
    #[case::object(json!({"tokens":true}))]
    fn malformed_include_is_rejected(#[case] body: Value) {
        assert!(serde_json::from_value::<TokenizeInclude>(body).is_err());
    }
}
