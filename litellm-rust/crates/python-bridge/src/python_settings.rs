use pyo3::prelude::*;

const MODULE: &str = "litellm.rust_bridge.settings";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PythonSettings {
    Http,
    UrlPolicy,
}

impl PythonSettings {
    #[cfg(test)]
    pub(crate) const ALL: [Self; 2] = [Self::Http, Self::UrlPolicy];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Http => "http_settings",
            Self::UrlPolicy => "url_policy",
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

#[cfg(test)]
pub(crate) const CONTRACT: &str = include_str!("../python_settings.json");

#[cfg(test)]
mod tests {
    use std::{collections::BTreeSet, ffi::CString};

    use pyo3::{prelude::*, types::PyDict};

    use super::{CONTRACT, PythonSettings};

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
}
