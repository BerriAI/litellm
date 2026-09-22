use pyo3::prelude::*;

use crate::coercion::{Field, ProjectionError};

const MODULE: &str = "litellm.rust_bridge.settings";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PythonSettings {
    Http,
    UrlPolicy,
    ProviderDefaults,
    SecretManager,
}

/// The semantic adapter a setting is projected through; the identifiers match
/// `python_settings.json` and the coercion contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Adapter {
    Truthy,
    ExactTrue,
    StrictString,
    OptionalStrictString,
    FalsyOptionalString,
    TuningString,
    StringCollection,
    HostCollection,
    SslVerifyInput,
    PythonBinding,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Precedence {
    ModuleGlobal,
    Accessor,
}

/// One row of the settings contract, declared next to the projector that reads it.
/// Production projection reads fields by name; the contract metadata feeds the manifest test.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "contract metadata is consumed by the manifest test"
    )
)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct SettingSpec {
    pub(crate) group: PythonSettings,
    pub(crate) name: &'static str,
    pub(crate) adapter: Adapter,
    pub(crate) precedence: Precedence,
    pub(crate) sensitive: bool,
    pub(crate) shapes: &'static [&'static str],
    pub(crate) unsupported_live: Option<&'static str>,
}

impl SettingSpec {
    pub(crate) const fn new(group: PythonSettings, name: &'static str, adapter: Adapter) -> Self {
        Self {
            group,
            name,
            adapter,
            precedence: Precedence::ModuleGlobal,
            sensitive: false,
            shapes: &[],
            unsupported_live: None,
        }
    }

    pub(crate) const fn sensitive(self) -> Self {
        Self {
            sensitive: true,
            ..self
        }
    }

    pub(crate) const fn accessor(self) -> Self {
        Self {
            precedence: Precedence::Accessor,
            ..self
        }
    }

    pub(crate) const fn shapes(self, shapes: &'static [&'static str]) -> Self {
        Self { shapes, ..self }
    }

    pub(crate) const fn unsupported_live(self, policy: &'static str) -> Self {
        Self {
            unsupported_live: Some(policy),
            ..self
        }
    }
}

/// One accessor result, read while attached to Python and projected field by field.
pub(crate) struct Snapshot<'py> {
    group: PythonSettings,
    value: Bound<'py, PyAny>,
}

impl<'py> Snapshot<'py> {
    pub(crate) fn field(&self, spec: &SettingSpec) -> Result<Field<'py>, ProjectionError> {
        debug_assert_eq!(
            spec.group, self.group,
            "{} read from another group",
            spec.name
        );
        Field::read(&self.value, self.group.name(), spec.name)
    }
}

impl PythonSettings {
    #[cfg(test)]
    pub(crate) const ALL: [Self; 4] = [
        Self::Http,
        Self::UrlPolicy,
        Self::ProviderDefaults,
        Self::SecretManager,
    ];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Http => "http_settings",
            Self::UrlPolicy => "url_policy",
            Self::ProviderDefaults => "provider_defaults",
            Self::SecretManager => "secret_manager",
        }
    }

    #[cfg(test)]
    pub(crate) fn version(self) -> u32 {
        match self {
            Self::Http | Self::UrlPolicy | Self::ProviderDefaults => 1,
            Self::SecretManager => 3,
        }
    }

    #[cfg(test)]
    /// The specs of every field this group's projector reads.
    pub(crate) fn specs(self) -> &'static [SettingSpec] {
        match self {
            Self::Http => crate::http::HTTP_SPECS,
            Self::UrlPolicy => crate::http::URL_POLICY_SPECS,
            Self::ProviderDefaults => crate::routes::ocr::PROVIDER_DEFAULT_SPECS,
            Self::SecretManager => crate::secrets::config::SECRET_MANAGER_SPECS,
        }
    }

    pub(crate) fn read(self, py: Python<'_>) -> PyResult<Snapshot<'_>> {
        let value = py.import(MODULE)?.getattr(self.name())?.call0()?;
        Ok(Snapshot { group: self, value })
    }

    #[cfg(test)]
    pub(crate) fn snapshot(self, value: Bound<'_, PyAny>) -> Snapshot<'_> {
        Snapshot { group: self, value }
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
    use std::collections::HashSet;

    use pyo3::prelude::*;
    use serde_json::{Value, json};

    use super::{CONTRACT, Precedence, PythonSettings, SettingSpec};

    fn manifest_row(spec: &SettingSpec) -> Value {
        json!({
            "adapter": format!("{:?}", spec.adapter),
            "required": true,
            "precedence": match spec.precedence {
                Precedence::ModuleGlobal => "module_global",
                Precedence::Accessor => "accessor",
            },
            "sensitive": spec.sensitive,
            "shapes": spec.shapes,
            "unsupported_live": spec.unsupported_live,
        })
    }

    #[test]
    fn settings_manifest_matches_the_projector_specs() {
        Python::initialize();
        let manifest: Value = Python::attach(|py| {
            let value = py
                .import("json")
                .unwrap()
                .call_method1("loads", (CONTRACT,))
                .unwrap();
            litellm_host_python::from_py(&value).unwrap()
        });
        let expected: serde_json::Map<String, Value> = PythonSettings::ALL
            .into_iter()
            .map(|group| {
                let fields: serde_json::Map<String, Value> = group
                    .specs()
                    .iter()
                    .map(|spec| (spec.name.to_owned(), manifest_row(spec)))
                    .collect();
                (
                    group.name().to_owned(),
                    json!({ "version": group.version(), "fields": fields }),
                )
            })
            .collect();
        assert_eq!(manifest, Value::Object(expected));
    }

    #[test]
    fn every_spec_sits_in_its_own_group_table_with_a_unique_name() {
        for group in PythonSettings::ALL {
            let mut names = HashSet::new();
            for spec in group.specs() {
                assert_eq!(
                    spec.group,
                    group,
                    "{} is filed under {}",
                    spec.name,
                    group.name()
                );
                assert!(names.insert(spec.name), "{} declared twice", spec.name);
            }
            assert!(!names.is_empty(), "{} has no specs", group.name());
        }
    }
}
