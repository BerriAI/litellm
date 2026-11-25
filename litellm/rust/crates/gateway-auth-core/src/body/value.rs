//! What a caller wrote in a request body, as the body check reads it. Nothing here resolves a
//! value: a rule may ask whether a string spells a secret reference, and never what it refers to.

use std::fmt;

use super::names::{Names, Reading};

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

    /// The entry `key` names alone, or none.
    pub(crate) fn only(&self, key: &str) -> Self {
        self.0
            .iter()
            .filter(|(name, _)| name == key)
            .cloned()
            .collect()
    }

    /// The mapping a JSON text encodes, as the proxy decodes it later, or `None` when the text is
    /// not JSON or not a mapping.
    pub(crate) fn decode(text: &str) -> Option<Self> {
        match CallerValue::from(&serde_json::from_str::<serde_json::Value>(text).ok()?) {
            CallerValue::Map(params) => Some(params),
            CallerValue::Null
            | CallerValue::Text(_)
            | CallerValue::List(_)
            | CallerValue::Other => None,
        }
    }

    pub(crate) fn get_mut(&mut self, key: &str) -> Option<&mut CallerValue> {
        self.0
            .iter_mut()
            .find_map(|(name, value)| (name == key).then_some(value))
    }

    /// Sets `key`, in its place when it is already there, as a Python dict assignment does.
    pub(crate) fn set(&mut self, key: &str, value: CallerValue) {
        match self.get_mut(key) {
            Some(held) => *held = value,
            None => self.0.push((key.to_owned(), value)),
        }
    }
}

impl Params {
    /// A request body parsed as JSON, read as far as [`Names::reading`] says a rule looks: a key
    /// no rule reads is kept as [`CallerValue::Other`], so message content is never copied.
    pub fn read(body: &serde_json::Map<String, serde_json::Value>, names: &Names) -> Self {
        body.iter()
            .map(|(key, value)| {
                let read = match names.reading(key) {
                    Reading::Unread => CallerValue::Other,
                    Reading::Shallow => match value {
                        serde_json::Value::Null => CallerValue::Null,
                        serde_json::Value::String(text) => CallerValue::Text(text.clone()),
                        _ => CallerValue::Other,
                    },
                    Reading::Deep => CallerValue::from(value),
                };
                (key.clone(), read)
            })
            .collect()
    }
}

/// A JSON value as a rule reads it. An object keeps its keys in the order written, with the last
/// value of a repeated key in the place of its first, as Python's `json.loads` reads one.
impl From<&serde_json::Value> for CallerValue {
    fn from(json: &serde_json::Value) -> Self {
        match json {
            serde_json::Value::Null => Self::Null,
            serde_json::Value::String(text) => Self::Text(text.clone()),
            serde_json::Value::Array(items) => Self::List(items.iter().map(Self::from).collect()),
            serde_json::Value::Object(entries) => Self::Map(
                entries
                    .iter()
                    .map(|(key, entry)| (key.clone(), Self::from(entry)))
                    .collect(),
            ),
            serde_json::Value::Bool(_) | serde_json::Value::Number(_) => Self::Other,
        }
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

    #[rstest]
    #[case::a_mapping(r#"{"b": 1, "a": "x", "b": "y"}"#, Some(vec![("b", "Text(..)"), ("a", "Text(..)")]))]
    #[case::not_a_mapping(r#"["a"]"#, None)]
    #[case::not_json("{api_base: x}", None)]
    fn a_text_decodes_to_the_mapping_python_reads_from_it(
        #[case] text: &str,
        #[case] expected: Option<Vec<(&str, &str)>>,
    ) {
        let decoded = Params::decode(text).map(|params| {
            params
                .iter()
                .map(|(key, value)| (key.to_owned(), format!("{value:?}")))
                .collect::<Vec<_>>()
        });

        assert_eq!(
            decoded,
            expected.map(|entries| entries
                .into_iter()
                .map(|(key, value)| (key.to_owned(), value.to_owned()))
                .collect())
        );
    }

    #[test]
    fn debug_output_never_shows_what_a_caller_wrote() {
        let params = Params::from_iter([("api_key".to_owned(), text("sk-caller-own-key"))]);

        assert_eq!(format!("{params:?}"), r#"{"api_key": Text(..)}"#);
    }
}
