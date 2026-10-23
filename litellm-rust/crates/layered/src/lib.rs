mod env;
mod error;
pub mod read;
mod sources;

pub use env::Env;
pub use error::{Error, Invalid};
pub use litellm_core_utils::settings::Lookup;
pub use litellm_layered_derive::Layered;
pub use sources::Sources;

pub trait Partial: Default {
    fn or(self, lower: Self) -> Self;
}

pub fn merge<P: Partial>(highest_precedence_first: impl IntoIterator<Item = P>) -> P {
    highest_precedence_first
        .into_iter()
        .reduce(P::or)
        .unwrap_or_default()
}

pub fn read_env<T>(
    env: &(impl Lookup + ?Sized),
    names: &'static [&'static str],
    keep_blank: bool,
    read: impl Fn(&str) -> Result<Option<T>, Invalid>,
) -> Result<Option<T>, Error> {
    for name in names {
        let Some(value) = env.get(name) else {
            continue;
        };
        if !keep_blank && value.trim().is_empty() {
            continue;
        }
        if let Some(value) = read(&value).map_err(|invalid| Error { name, invalid })? {
            return Ok(Some(value));
        }
    }
    Ok(None)
}

#[doc(hidden)]
pub mod __private {
    use std::fmt;

    use serde_json::{Map, Value};

    pub use serde;
    pub use serde_json;

    pub fn from_litellm_params<T>(
        params: &Map<String, Value>,
        names: &[&str],
        keep_blank: bool,
        decode: impl Fn(&Value) -> Option<T>,
    ) -> Option<T> {
        names
            .iter()
            .filter_map(|name| params.get(*name))
            .filter(|value| keep_blank || !is_blank(value))
            .find_map(decode)
    }

    fn is_blank(value: &Value) -> bool {
        value.as_str().is_some_and(|text| text.trim().is_empty())
    }

    pub struct Redacted<'a, T>(pub &'a Option<T>);

    impl<T> fmt::Debug for Redacted<'_, T> {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self.0 {
                Some(_) => formatter.write_str("Some(\"[REDACTED]\")"),
                None => formatter.write_str("None"),
            }
        }
    }
}
