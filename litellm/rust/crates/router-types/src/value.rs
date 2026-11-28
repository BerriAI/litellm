use std::fmt;

use serde::Deserialize;

/// A typed value, or the text a config spells in its place: an `os.environ/NAME` reference
/// the secrets layer resolves later, or a word Python's coercion reads the value from.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum Spelled<T> {
    Value(T),
    Text(String),
}

impl Spelled<f64> {
    /// The number a config spells, as Python reads it off one: the value, or the text as `float(text)`
    /// reads it, and `None` where that raises (an unresolved `os.environ/` reference included).
    pub fn lax(&self) -> Option<f64> {
        match self {
            Self::Value(value) => Some(*value),
            Self::Text(text) => litellm_python_compat::number::float(text),
        }
    }
}

impl<T: fmt::Display> fmt::Display for Spelled<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Value(value) => value.fmt(formatter),
            Self::Text(text) => formatter.write_str(text),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::Spelled;

    #[rstest]
    #[case::a_number(Spelled::Value(1.5), Some(1.5))]
    #[case::text_that_spells_one(Spelled::Text("10".into()), Some(10.0))]
    #[case::padded_text(Spelled::Text(" 2.5 ".into()), Some(2.5))]
    #[case::digits_grouped_as_python_allows(Spelled::Text("1_000".into()), Some(1000.0))]
    #[case::an_unresolved_reference(Spelled::Text("os.environ/TPM".into()), None)]
    #[case::a_word(Spelled::Text("ten".into()), None)]
    fn a_spelled_number_reads_as_pythons_float(
        #[case] spelled: Spelled<f64>,
        #[case] read: Option<f64>,
    ) {
        assert_eq!(spelled.lax(), read);
    }
}
