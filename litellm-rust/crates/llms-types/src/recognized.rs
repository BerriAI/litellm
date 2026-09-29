use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Recognized<T> {
    Known(T),
    Unrecognized(Value),
}

impl<T> Recognized<T> {
    pub fn known(&self) -> Option<&T> {
        match self {
            Self::Known(value) => Some(value),
            Self::Unrecognized(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case::known(json!(7), Recognized::Known(7))]
    #[case::wrong_type(json!("7"), Recognized::Unrecognized(json!("7")))]
    #[case::out_of_range(json!(-1), Recognized::Unrecognized(json!(-1)))]
    #[case::object(json!({"a": 1}), Recognized::Unrecognized(json!({"a": 1})))]
    fn value_is_known_only_when_it_parses_as_the_type(
        #[case] value: Value,
        #[case] expected: Recognized<u64>,
    ) {
        assert_eq!(
            serde_json::from_value::<Recognized<u64>>(value.clone()).unwrap(),
            expected
        );
        assert_eq!(serde_json::to_value(expected).unwrap(), value);
    }
}
