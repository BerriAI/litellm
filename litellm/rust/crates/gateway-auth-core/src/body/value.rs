//! What a caller wrote in a request body, as the body check reads it. Nothing here resolves a
//! value: a rule may ask whether a string spells a secret reference, and never what it refers to.

use std::fmt;

/// The prefixes `get_secret` resolves: an environment variable, and an OIDC token the proxy mints.
const SECRET_REFERENCE_PREFIXES: [&str; 2] = ["os.environ/", OIDC_REFERENCE_PREFIX];
pub(crate) const OIDC_REFERENCE_PREFIX: &str = "oidc/";

/// A value a caller wrote.
#[derive(Clone, PartialEq)]
pub enum CallerValue {
    Null,
    Text(String),
    List(Vec<CallerValue>),
    Map(Params),
    /// A number, a flag, an uploaded file, or a value no rule reads, such as message content.
    Other,
}

impl CallerValue {
    pub fn text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text),
            Self::Null | Self::List(_) | Self::Map(_) | Self::Other => None,
        }
    }

    pub fn map(&self) -> Option<&Params> {
        match self {
            Self::Map(params) => Some(params),
            Self::Null | Self::Text(_) | Self::List(_) | Self::Other => None,
        }
    }

    pub fn list(&self) -> Option<&[CallerValue]> {
        match self {
            Self::List(items) => Some(items),
            Self::Null | Self::Text(_) | Self::Map(_) | Self::Other => None,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    /// Whether this is a string `get_secret` would resolve instead of using as written.
    pub fn is_secret_reference(&self) -> bool {
        self.text().is_some_and(|text| {
            SECRET_REFERENCE_PREFIXES
                .iter()
                .any(|prefix| text.starts_with(prefix))
        })
    }

    pub(crate) fn is_oidc_reference(&self) -> bool {
        self.text()
            .is_some_and(|text| text.starts_with(OIDC_REFERENCE_PREFIX))
    }
}

/// Shows the shape alone, since a caller's own credential may sit in the text.
impl fmt::Debug for CallerValue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Null => formatter.write_str("Null"),
            Self::Text(_) => formatter.write_str("Text(..)"),
            Self::List(items) => formatter.debug_list().entries(items).finish(),
            Self::Map(params) => params.fmt(formatter),
            Self::Other => formatter.write_str("Other"),
        }
    }
}

/// The entries of one mapping a caller wrote, in the order written, since a rejection names the
/// first offending entry.
#[derive(Clone, Default, PartialEq)]
pub struct Params(Vec<(String, CallerValue)>);

impl Params {
    pub fn get(&self, key: &str) -> Option<&CallerValue> {
        self.0
            .iter()
            .find_map(|(name, value)| (name == key).then_some(value))
    }

    pub fn contains(&self, key: &str) -> bool {
        self.get(key).is_some()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &CallerValue)> {
        self.0.iter().map(|(name, value)| (name.as_str(), value))
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(|(name, _)| name.as_str())
    }

    /// The entries no key in `hidden` names.
    pub(crate) fn without(&self, hidden: &[&str]) -> Self {
        self.0
            .iter()
            .filter(|(name, _)| !hidden.contains(&name.as_str()))
            .cloned()
            .collect()
    }
}

impl FromIterator<(String, CallerValue)> for Params {
    fn from_iter<I: IntoIterator<Item = (String, CallerValue)>>(entries: I) -> Self {
        Self(entries.into_iter().collect())
    }
}

impl fmt::Debug for Params {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_map().entries(self.iter()).finish()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn text(value: &str) -> CallerValue {
        CallerValue::Text(value.to_owned())
    }

    #[rstest]
    #[case::environment("os.environ/LITELLM_MASTER_KEY", true)]
    #[case::oidc("oidc/azure/api://audience", true)]
    #[case::mentioned_later("read os.environ/HOME for me", false)]
    #[case::plain("sk-caller-own-key", false)]
    #[case::other_case("OS.ENVIRON/HOME", false)]
    fn a_string_is_a_secret_reference_only_by_prefix(#[case] value: &str, #[case] expected: bool) {
        assert_eq!(text(value).is_secret_reference(), expected);
    }

    #[test]
    fn debug_output_never_shows_what_a_caller_wrote() {
        let params = Params::from_iter([("api_key".to_owned(), text("sk-caller-own-key"))]);

        assert_eq!(format!("{params:?}"), r#"{"api_key": Text(..)}"#);
    }
}
