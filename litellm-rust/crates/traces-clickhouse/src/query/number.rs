use serde::{Deserialize, Deserializer, de::DeserializeOwned};

pub(super) fn deserialize<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: DeserializeOwned,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Number {
        Quoted(String),
        Unquoted(serde_json::Number),
    }
    match Number::deserialize(deserializer)? {
        Number::Quoted(value) => serde_json::from_str(&value),
        Number::Unquoted(value) => serde_json::from_value(serde_json::Value::Number(value)),
    }
    .map_err(serde::de::Error::custom)
}

#[cfg(test)]
mod tests {
    use crate::query::named::SpanErrorRow;
    use rstest::rstest;

    #[rstest]
    #[case::quoted_max(serde_json::json!(u64::MAX.to_string()), Some(u64::MAX))]
    #[case::unquoted_max(serde_json::json!(u64::MAX), Some(u64::MAX))]
    #[case::overflow(serde_json::json!("18446744073709551616"), None)]
    #[case::negative(serde_json::json!(-1), None)]
    #[case::fraction(serde_json::json!(1.5), None)]
    fn numeric_rows_enforce_integer_range(
        #[case] value: serde_json::Value,
        #[case] expected: Option<u64>,
    ) {
        let row = serde_json::from_value::<SpanErrorRow>(serde_json::json!({
            "span_id": "span", "message": "error", "total_chars": value, "version": "hash"
        }));
        match expected {
            Some(value) => assert_eq!(row.unwrap().0.total_chars, value),
            None => assert!(row.is_err()),
        }
    }
}
