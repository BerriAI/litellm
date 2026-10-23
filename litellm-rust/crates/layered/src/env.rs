use std::{collections::BTreeMap, ffi::OsString, fmt};

use crate::Lookup;

#[derive(Clone, Default, PartialEq, Eq)]
pub struct Env(BTreeMap<String, String>);

impl Env {
    #[allow(clippy::disallowed_methods)]
    pub fn from_process() -> Self {
        Self::from_vars(std::env::vars_os())
    }

    pub fn from_vars(vars: impl IntoIterator<Item = (OsString, OsString)>) -> Self {
        Self(
            vars.into_iter()
                .filter_map(|(name, value)| {
                    Some((name.into_string().ok()?, value.into_string().ok()?))
                })
                .collect(),
        )
    }

    pub fn overlaid(self, values: impl IntoIterator<Item = (String, String)>) -> Self {
        Self(self.0.into_iter().chain(values).collect())
    }

    pub fn of(values: &[(&str, &str)]) -> Self {
        Self(
            values
                .iter()
                .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
                .collect(),
        )
    }
}

impl Lookup for Env {
    fn get(&self, name: &str) -> Option<String> {
        self.0.get(name).cloned()
    }
}

impl fmt::Debug for Env {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("Env")
            .field(&self.0.keys().collect::<Vec<_>>())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    fn overlaid_values_win_over_the_snapshot() {
        let env = Env::of(&[("SHARED", "process"), ("PROCESS_ONLY", "process")])
            .overlaid([("SHARED".to_owned(), "config".to_owned())]);
        assert_eq!(env.get("SHARED").as_deref(), Some("config"));
        assert_eq!(env.get("PROCESS_ONLY").as_deref(), Some("process"));
        assert_eq!(env.get("ABSENT"), None);
    }

    #[rstest]
    fn debug_lists_names_without_values() {
        let debug = format!("{:?}", Env::of(&[("API_KEY", "sk-hidden")]));
        assert!(debug.contains("API_KEY"));
        assert!(!debug.contains("sk-hidden"));
    }
}
