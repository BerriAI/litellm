use serde::Deserialize;

/// A typed value, or the text a config spells in its place: an `os.environ/NAME` reference
/// the secrets layer resolves later, or a word Python's coercion reads the value from.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum Spelled<T> {
    Value(T),
    Text(String),
}

impl Spelled<bool> {
    /// The flag, reading a `true` word the way Python's coercion does. An `os.environ/`
    /// reference left unresolved is not set.
    pub fn is_set(&self) -> bool {
        match self {
            Self::Value(value) => *value,
            Self::Text(text) => text.trim().eq_ignore_ascii_case("true"),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::value(Spelled::Value(true), true)]
    #[case::false_value(Spelled::Value(false), false)]
    #[case::word(Spelled::Text(" True ".into()), true)]
    #[case::other_word(Spelled::Text("yes".into()), false)]
    #[case::reference(Spelled::Text("os.environ/DROP".into()), false)]
    fn flags_read_values_and_words(#[case] flag: Spelled<bool>, #[case] expected: bool) {
        assert_eq!(flag.is_set(), expected);
    }
}
