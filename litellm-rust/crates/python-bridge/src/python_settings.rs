use pyo3::prelude::*;

const MODULE: &str = "litellm.rust_bridge.settings";

/// Every group of `litellm.*` module globals the native routes read. Environment overrides are
/// applied on the Rust side, so each function returns only what the Python process configured.
/// A group is deleted once Rust owns loading that configuration, so this enum only shrinks.
///
/// `litellm/rust_bridge/settings.py` is the only Python module behind it, and
/// `python_settings.json` pins the fields each function returns on both sides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PythonSettings {
    Http,
}

impl PythonSettings {
    #[cfg(test)]
    pub(crate) const ALL: [Self; 1] = [Self::Http];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Http => "http_settings",
        }
    }

    pub(crate) fn read(self, py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        py.import(MODULE)?.getattr(self.name())?.call0()
    }
}

#[cfg(test)]
pub(crate) const CONTRACT: &str = include_str!("../python_settings.json");

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::{CONTRACT, PythonSettings};

    #[test]
    fn every_settings_group_is_in_the_python_contract() {
        let contract: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(CONTRACT).unwrap();
        let declared: BTreeSet<&str> = contract.keys().map(String::as_str).collect();
        let read: BTreeSet<&str> = PythonSettings::ALL.map(PythonSettings::name).into();
        assert_eq!(read, declared);
    }
}
