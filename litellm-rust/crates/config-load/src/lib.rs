mod error;
mod http;
pub mod include;

use std::{collections::BTreeMap, ffi::OsString, path::Path};

use litellm_config::Config;
use litellm_core_utils::settings::Lookup;

pub use self::{error::Error, http::http_settings};

pub fn load(path: impl AsRef<Path>) -> Result<Config, Error> {
    let path = path.as_ref();
    let root = path
        .canonicalize()
        .map_err(|error| Error::document(&path, error.into()))?;
    load_from(&include::Files, root)
}

pub fn load_from<S: include::Store>(store: &S, root: S::Key) -> Result<Config, Error> {
    Ok(Config::from_mapping(
        include::merge(&include::documents(store, root)?).document,
    )?)
}

#[derive(Clone, Default)]
pub struct Environment(BTreeMap<String, String>);

impl Environment {
    pub fn from_os(variables: impl IntoIterator<Item = (OsString, OsString)>) -> Self {
        variables
            .into_iter()
            .filter_map(|(name, value)| Some((name.into_string().ok()?, value.into_string().ok()?)))
            .collect()
    }

    pub fn overlaid(self, higher: impl IntoIterator<Item = (String, String)>) -> Self {
        Self(self.0.into_iter().chain(higher).collect())
    }

    pub fn var(&self, name: &str) -> Option<&str> {
        self.0.get(name).map(String::as_str)
    }

    pub fn variables(&self) -> impl Iterator<Item = (String, String)> + '_ {
        self.0
            .iter()
            .map(|(name, value)| (name.clone(), value.clone()))
    }
}

impl FromIterator<(String, String)> for Environment {
    fn from_iter<I: IntoIterator<Item = (String, String)>>(variables: I) -> Self {
        Self(variables.into_iter().collect())
    }
}

impl Lookup for Environment {
    fn get(&self, name: &str) -> Option<String> {
        self.0.get(name).cloned()
    }
}
