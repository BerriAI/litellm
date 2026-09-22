use pyo3::prelude::*;

const MODULE: &str = "litellm.rust_bridge.settings";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PythonSettings {
    Http,
    UrlPolicy,
    ProviderDefaults,
    SecretManager,
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
    use super::{CONTRACT, PythonSettings};
    use pyo3::prelude::*;
    use serde_json::{Value, json};

    struct SettingSpec {
        group: &'static str,
        name: &'static str,
        adapter: &'static str,
        precedence: &'static str,
        sensitive: bool,
        shapes: &'static [&'static str],
        unsupported_live: Option<&'static str>,
    }

    const SETTINGS: &[SettingSpec] = &[
        SettingSpec {
            group: "http_settings",
            name: "ssl_verify",
            adapter: "SslVerifyInput",
            precedence: "module_global",
            sensitive: false,
            shapes: &["none", "bool", "str"],
            unsupported_live: Some("configuration_error"),
        },
        SettingSpec {
            group: "http_settings",
            name: "ssl_certificate",
            adapter: "OptionalStrictString",
            precedence: "module_global",
            sensitive: false,
            shapes: &[],
            unsupported_live: None,
        },
        SettingSpec {
            group: "http_settings",
            name: "ssl_security_level",
            adapter: "TuningString",
            precedence: "module_global",
            sensitive: false,
            shapes: &[],
            unsupported_live: None,
        },
        SettingSpec {
            group: "http_settings",
            name: "ssl_ecdh_curve",
            adapter: "TuningString",
            precedence: "module_global",
            sensitive: false,
            shapes: &[],
            unsupported_live: None,
        },
        SettingSpec {
            group: "http_settings",
            name: "force_ipv4",
            adapter: "Truthy",
            precedence: "module_global",
            sensitive: false,
            shapes: &[],
            unsupported_live: None,
        },
        SettingSpec {
            group: "http_settings",
            name: "http2",
            adapter: "ExactTrue",
            precedence: "module_global",
            sensitive: false,
            shapes: &[],
            unsupported_live: None,
        },
        SettingSpec {
            group: "http_settings",
            name: "aiohttp_trust_env",
            adapter: "Truthy",
            precedence: "module_global",
            sensitive: false,
            shapes: &[],
            unsupported_live: None,
        },
        SettingSpec {
            group: "http_settings",
            name: "disable_aiohttp_trust_env",
            adapter: "Truthy",
            precedence: "module_global",
            sensitive: false,
            shapes: &[],
            unsupported_live: None,
        },
        SettingSpec {
            group: "http_settings",
            name: "disable_aiohttp_transport",
            adapter: "ExactTrue",
            precedence: "module_global",
            sensitive: false,
            shapes: &[],
            unsupported_live: None,
        },
        SettingSpec {
            group: "http_settings",
            name: "user_agent",
            adapter: "StrictString",
            precedence: "accessor",
            sensitive: false,
            shapes: &[],
            unsupported_live: None,
        },
        SettingSpec {
            group: "url_policy",
            name: "user_url_validation",
            adapter: "Truthy",
            precedence: "module_global",
            sensitive: false,
            shapes: &[],
            unsupported_live: None,
        },
        SettingSpec {
            group: "url_policy",
            name: "user_url_allowed_hosts",
            adapter: "HostCollection",
            precedence: "module_global",
            sensitive: false,
            shapes: &[],
            unsupported_live: None,
        },
        SettingSpec {
            group: "provider_defaults",
            name: "vertex_project",
            adapter: "FalsyOptionalString",
            precedence: "module_global",
            sensitive: true,
            shapes: &[],
            unsupported_live: None,
        },
        SettingSpec {
            group: "provider_defaults",
            name: "vertex_location",
            adapter: "FalsyOptionalString",
            precedence: "module_global",
            sensitive: true,
            shapes: &[],
            unsupported_live: None,
        },
        SettingSpec {
            group: "provider_defaults",
            name: "enable_azure_ad_token_refresh",
            adapter: "ExactTrue",
            precedence: "module_global",
            sensitive: false,
            shapes: &[],
            unsupported_live: None,
        },
        SettingSpec {
            group: "secret_manager",
            name: "readable",
            adapter: "StrictBool",
            precedence: "accessor",
            sensitive: false,
            shapes: &[],
            unsupported_live: None,
        },
    ];

    #[test]
    fn settings_manifest_matches_the_semantic_contract() {
        pyo3::Python::initialize();
        let manifest: Value = pyo3::Python::attach(|py| {
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
                let fields: serde_json::Map<String, Value> = SETTINGS
                    .iter()
                    .filter(|spec| spec.group == group.name())
                    .map(|spec| {
                        (
                            spec.name.to_owned(),
                            json!({
                                "adapter": spec.adapter,
                                "required": true,
                                "precedence": spec.precedence,
                                "sensitive": spec.sensitive,
                                "shapes": spec.shapes,
                                "unsupported_live": spec.unsupported_live,
                            }),
                        )
                    })
                    .collect();
                (
                    group.name().to_owned(),
                    json!({"version": 1, "fields": fields}),
                )
            })
            .collect();
        assert_eq!(manifest, Value::Object(expected));
    }
}
