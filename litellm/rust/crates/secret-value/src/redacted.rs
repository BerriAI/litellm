use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Redacted<T>(T);

impl<T> Redacted<T> {
    pub fn new(value: T) -> Self {
        Self(value)
    }

    pub fn expose(&self) -> &T {
        &self.0
    }

    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T> From<T> for Redacted<T> {
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

impl<T> fmt::Debug for Redacted<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("[REDACTED]")
    }
}

impl<E: Error> fmt::Display for Redacted<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, formatter)
    }
}

impl<E: Error> Error for Redacted<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.0.source()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use rstest::rstest;

    use super::Redacted;

    #[rstest]
    fn debug_hides_every_entry() {
        let bag = Redacted::new(BTreeMap::from([("api_key", "credential-value")]));

        let debug = format!("{bag:?}");

        assert!(!debug.contains("credential-value"), "{debug}");
        assert!(!debug.contains("api_key"), "{debug}");
        assert_eq!(debug, "[REDACTED]");
    }

    #[rstest]
    fn an_error_cause_hides_only_its_debug() {
        let cause = Redacted::new(std::io::Error::other("credential-value"));

        assert_eq!(format!("{cause:?}"), "[REDACTED]");
        assert_eq!(cause.to_string(), "credential-value");
    }

    #[rstest]
    fn deserializes_as_the_wrapped_value() {
        let bag: Redacted<BTreeMap<String, String>> =
            serde_json::from_str(r#"{"api_key": "credential-value"}"#).unwrap();

        assert_eq!(
            bag.expose().get("api_key").map(String::as_str),
            Some("credential-value")
        );
    }
}
