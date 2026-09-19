use litellm_core_utils::settings::Lookup;
use pyo3::prelude::*;

const MODULE: &str = "litellm.rust_bridge.settings";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PythonSettings {
    Http,
    UrlPolicy,
    ProviderDefaults,
}

impl PythonSettings {
    #[cfg(test)]
    pub(crate) const ALL: [Self; 3] = [Self::Http, Self::UrlPolicy, Self::ProviderDefaults];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Http => "http_settings",
            Self::UrlPolicy => "url_policy",
            Self::ProviderDefaults => "provider_defaults",
        }
    }

    pub(crate) fn read(self, py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        py.import(MODULE)?.getattr(self.name())?.call0()
    }

    pub(crate) fn warn(py: Python<'_>, message: &str) -> PyResult<()> {
        py.import(MODULE)?.getattr("warn")?.call1((message,))?;
        Ok(())
    }
}

pub(crate) struct PythonSecrets;

impl Lookup for PythonSecrets {
    fn get(&self, name: &str) -> Option<String> {
        Python::attach(|py| {
            py.import(MODULE)
                .and_then(|module| module.getattr("secret")?.call1((name,)))
                .and_then(|value| value.extract::<Option<String>>())
                .unwrap_or_else(|error| {
                    let _ =
                        PythonSettings::warn(py, &format!("reading secret {name} failed: {error}"));
                    None
                })
        })
    }
}

#[cfg(test)]
pub(crate) const CONTRACT: &str = include_str!("../python_settings.json");

#[cfg(test)]
mod tests {
    use std::{collections::BTreeSet, ffi::CString};

    use litellm_core_utils::settings::Lookup;
    use pyo3::{prelude::*, types::PyDict};

    use super::{CONTRACT, PythonSecrets, PythonSettings};

    #[test]
    fn every_settings_group_is_in_the_python_contract() {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            locals.set_item("contract", CONTRACT).unwrap();
            let source = CString::new("import json\nkeys = list(json.loads(contract))").unwrap();
            py.run(&source, Some(&locals), Some(&locals)).unwrap();
            let declared: BTreeSet<String> = locals
                .get_item("keys")
                .unwrap()
                .unwrap()
                .extract::<Vec<String>>()
                .unwrap()
                .into_iter()
                .collect();
            let read: BTreeSet<String> = PythonSettings::ALL
                .map(|group| group.name().to_owned())
                .into();
            assert_eq!(read, declared);
        });
    }

    #[test]
    fn secrets_come_from_the_python_secret_reader_and_a_failed_read_is_unset() {
        Python::initialize();
        Python::attach(|py| {
            py.run(
                c"
import sys
import types
settings = types.ModuleType('litellm.rust_bridge.settings')
settings.warnings = []
def secret(name):
    if name == 'BROKEN':
        raise RuntimeError('vault down')
    return {'MISTRAL_API_KEY': 'from-vault'}.get(name)
settings.secret = secret
settings.warn = settings.warnings.append
sys.modules.setdefault('litellm', types.ModuleType('litellm'))
sys.modules.setdefault('litellm.rust_bridge', types.ModuleType('litellm.rust_bridge'))
sys.modules['litellm.rust_bridge.settings'] = settings
",
                None,
                None,
            )
            .unwrap();
        });
        assert_eq!(
            PythonSecrets.get("MISTRAL_API_KEY").as_deref(),
            Some("from-vault")
        );
        assert_eq!(PythonSecrets.get("ABSENT"), None);
        assert_eq!(PythonSecrets.get("BROKEN"), None);
        Python::attach(|py| {
            let warnings: Vec<String> = py
                .import("litellm.rust_bridge.settings")
                .unwrap()
                .getattr("warnings")
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(warnings.len(), 1);
            assert!(warnings[0].contains("BROKEN") && warnings[0].contains("vault down"));
        });
    }
}
