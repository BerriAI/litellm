use serde::{Deserialize, Deserializer};

/// A typed value, or the text a config spells in its place: an `os.environ/NAME` reference
/// the secrets layer resolves later, or a word Python's coercion reads the value from.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum Spelled<T> {
    Value(T),
    Text(String),
}

/// `x if isinstance(x, list) else [x]`, for `#[serde(deserialize_with)]`.
pub fn one_or_many<'de, D: Deserializer<'de>, T: Deserialize<'de>>(
    deserializer: D,
) -> Result<Option<Box<[T]>>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum OneOrMany<T> {
        Many(Box<[T]>),
        One(T),
    }

    Ok(
        Option::<OneOrMany<T>>::deserialize(deserializer)?.map(|value| match value {
            OneOrMany::Many(values) => values,
            OneOrMany::One(value) => Box::from([value]),
        }),
    )
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde::Deserialize;
    use serde_json::{Value, json};

    use super::{Spelled, one_or_many};

    #[derive(Deserialize)]
    struct Entries {
        #[serde(default, deserialize_with = "one_or_many")]
        entries: Option<Box<[Value]>>,
    }

    fn entries(document: Value) -> Option<Box<[Value]>> {
        serde_json::from_value::<Entries>(document).unwrap().entries
    }

    #[rstest]
    #[case::one(json!({"entries": "langfuse"}), Some(vec![json!("langfuse")]))]
    #[case::one_mapping(json!({"entries": {"name": "otel"}}), Some(vec![json!({"name": "otel"})]))]
    #[case::many(json!({"entries": ["langfuse", "otel"]}), Some(vec![json!("langfuse"), json!("otel")]))]
    #[case::many_keeps_nested_lists(json!({"entries": [["a"], "b"]}), Some(vec![json!(["a"]), json!("b")]))]
    #[case::empty(json!({"entries": []}), Some(vec![]))]
    #[case::null(json!({"entries": null}), None)]
    #[case::absent(json!({}), None)]
    fn one_or_many_wraps_a_single_entry_and_keeps_a_list(
        #[case] document: Value,
        #[case] expected: Option<Vec<Value>>,
    ) {
        assert_eq!(entries(document), expected.map(Vec::into_boxed_slice));
    }

    #[rstest]
    #[case::value(json!(3.5), Spelled::Value(3.5))]
    #[case::integer(json!(3), Spelled::Value(3.0))]
    #[case::reference(json!("os.environ/TIMEOUT"), Spelled::Text("os.environ/TIMEOUT".into()))]
    #[case::numeric_text(json!("3"), Spelled::Text("3".into()))]
    fn spelled_keeps_text_that_is_not_the_value_type(
        #[case] input: Value,
        #[case] expected: Spelled<f64>,
    ) {
        assert_eq!(
            serde_json::from_value::<Spelled<f64>>(input).unwrap(),
            expected
        );
    }
}
