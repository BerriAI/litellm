use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    sync::{Arc, LazyLock, Mutex, PoisonError},
};

use litellm_http::{HttpClientConfig, HttpClientPool, HttpSettings, SslVerify, Unsupported};
use litellm_llms::custom_httpx::media::{PublicDnsResolver, UrlPolicy};
use pyo3::{prelude::*, types::PyDict};

use crate::{errors::RustBridgeDeclined, python_settings::PythonSettings};

static POOL: LazyLock<HttpClientPool> =
    LazyLock::new(|| HttpClientPool::new(Arc::new(PublicDnsResolver)));

static REPORTED_UNSUPPORTED: LazyLock<Mutex<HashSet<Unsupported>>> = LazyLock::new(Mutex::default);

pub(crate) fn pool() -> &'static HttpClientPool {
    &POOL
}

pub(crate) fn call_config(
    py: Python<'_>,
    kwargs: &Bound<'_, PyDict>,
    asynchronous: bool,
) -> PyResult<HttpClientConfig> {
    let configured = settings(&PythonSettings::Http.read(py)?)?
        .with_environment(&|name| std::env::var(name).ok());
    let settings = for_call(configured, call_ssl_verify(kwargs)?, asynchronous)
        .without_missing_files(&|path: &Path| path.exists());
    let resolution = HttpClientConfig::resolve(&settings);
    for unsupported in unreported(&REPORTED_UNSUPPORTED, resolution.unsupported) {
        PythonSettings::warn(py, &unsupported.to_string())?;
    }
    Ok(resolution.config)
}

fn unreported(
    reported: &Mutex<HashSet<Unsupported>>,
    unsupported: Vec<Unsupported>,
) -> Vec<Unsupported> {
    let mut reported = reported.lock().unwrap_or_else(PoisonError::into_inner);
    unsupported
        .into_iter()
        .filter(|unsupported| reported.insert(unsupported.clone()))
        .collect()
}

pub(crate) fn url_policy(py: Python<'_>) -> PyResult<UrlPolicy> {
    let policy: PythonUrlPolicy =
        PythonSettings::UrlPolicy
            .read(py)?
            .extract()
            .map_err(|error: PyErr| {
                RustBridgeDeclined::new_err(format!(
                    "litellm URL policy cannot be used by the Rust route: {error}"
                ))
            })?;
    Ok(UrlPolicy {
        validate: policy.user_url_validation,
        allowed_hosts: policy.user_url_allowed_hosts,
    })
}

fn call_ssl_verify(kwargs: &Bound<'_, PyDict>) -> PyResult<Option<SslVerify>> {
    Ok(kwargs
        .get_item("ssl_verify")?
        .and_then(|value| ssl_verify(&value)))
}

fn for_call(
    configured: HttpSettings,
    call_ssl_verify: Option<SslVerify>,
    asynchronous: bool,
) -> HttpSettings {
    HttpSettings {
        ssl_verify: call_ssl_verify.or(configured.ssl_verify),
        httpx_transport: configured.httpx_transport || !asynchronous,
        ..configured
    }
}

#[derive(FromPyObject)]
struct PythonUrlPolicy {
    user_url_validation: bool,
    user_url_allowed_hosts: Vec<String>,
}

#[derive(FromPyObject)]
struct PythonHttpSettings<'py> {
    ssl_verify: Bound<'py, PyAny>,
    ssl_certificate: Option<String>,
    ssl_security_level: Option<String>,
    ssl_ecdh_curve: Option<String>,
    force_ipv4: bool,
    http2: bool,
    aiohttp_trust_env: bool,
    disable_aiohttp_trust_env: bool,
    disable_aiohttp_transport: bool,
    user_agent: String,
}

fn settings(value: &Bound<'_, PyAny>) -> PyResult<HttpSettings> {
    let python: PythonHttpSettings = value.extract().map_err(|error: PyErr| {
        RustBridgeDeclined::new_err(format!(
            "litellm HTTP settings cannot be used by the Rust route: {error}"
        ))
    })?;
    Ok(HttpSettings {
        ssl_verify: ssl_verify(&python.ssl_verify),
        ssl_certificate: python.ssl_certificate.map(PathBuf::from),
        ssl_security_level: python.ssl_security_level,
        ssl_ecdh_curve: python.ssl_ecdh_curve,
        force_ipv4: python.force_ipv4,
        http2: python.http2,
        httpx_transport: python.disable_aiohttp_transport,
        user_agent: Some(python.user_agent),
        trust_proxy_env: python.aiohttp_trust_env,
        ignore_proxy_env: python.disable_aiohttp_trust_env,
        ..HttpSettings::default()
    })
}

fn ssl_verify(value: &Bound<'_, PyAny>) -> Option<SslVerify> {
    if let Ok(enabled) = value.extract::<bool>() {
        return Some(if enabled {
            SslVerify::Enabled
        } else {
            SslVerify::Disabled
        });
    }
    value
        .extract::<String>()
        .ok()
        .map(|path| SslVerify::parse(&path))
}

#[cfg(test)]
mod tests {
    use litellm_http::Verify;
    use rstest::rstest;

    use super::*;
    use crate::python_settings::CONTRACT;

    fn python_settings<'py>(py: Python<'py>, overrides: &str) -> Bound<'py, PyAny> {
        let source = format!(
            "
import json
import types
defaults = dict(
    ssl_verify=True,
    ssl_certificate=None,
    ssl_security_level=None,
    ssl_ecdh_curve=None,
    force_ipv4=False,
    http2=False,
    aiohttp_trust_env=False,
    disable_aiohttp_trust_env=False,
    disable_aiohttp_transport=False,
    user_agent='litellm/test',
)
defaults.update(dict({overrides}))
settings = types.SimpleNamespace(**{{name: defaults[name] for name in json.loads(contract)['http_settings']}})
"
        );
        let locals = PyDict::new(py);
        locals.set_item("contract", CONTRACT).unwrap();
        let source = std::ffi::CString::new(source).unwrap();
        py.run(&source, Some(&locals), Some(&locals)).unwrap();
        locals.get_item("settings").unwrap().unwrap()
    }

    #[test]
    fn default_python_settings_produce_default_settings_with_verification_on() {
        Python::initialize();
        Python::attach(|py| {
            let settings = settings(&python_settings(py, "")).unwrap();
            assert_eq!(
                settings,
                HttpSettings {
                    ssl_verify: Some(SslVerify::Enabled),
                    user_agent: Some("litellm/test".into()),
                    ..HttpSettings::default()
                }
            );
        });
    }

    #[test]
    fn python_settings_flow_into_settings() {
        Python::initialize();
        Python::attach(|py| {
            let settings = settings(&python_settings(
                py,
                "
ssl_verify='/etc/ssl/corp.pem',
ssl_certificate='/etc/ssl/client.pem',
ssl_security_level='2',
ssl_ecdh_curve='X25519',
force_ipv4=True,
http2=True,
aiohttp_trust_env=True,
disable_aiohttp_trust_env=True,
disable_aiohttp_transport=True,
user_agent='litellm/9.9.9',
",
            ))
            .unwrap();
            assert_eq!(
                settings,
                HttpSettings {
                    ssl_verify: Some(SslVerify::CaBundle("/etc/ssl/corp.pem".into())),
                    ssl_certificate: Some("/etc/ssl/client.pem".into()),
                    ssl_security_level: Some("2".into()),
                    ssl_ecdh_curve: Some("X25519".into()),
                    force_ipv4: true,
                    http2: true,
                    httpx_transport: true,
                    user_agent: Some("litellm/9.9.9".into()),
                    trust_proxy_env: true,
                    ignore_proxy_env: true,
                    ..HttpSettings::default()
                }
            );
        });
    }

    #[test]
    fn user_agent_environment_variable_beats_the_python_default() {
        Python::initialize();
        Python::attach(|py| {
            let settings = settings(&python_settings(py, ""))
                .unwrap()
                .with_environment(&|name| {
                    (name == "LITELLM_USER_AGENT").then(|| "operator/1".to_string())
                });
            assert_eq!(settings.user_agent.as_deref(), Some("operator/1"));
        });
    }

    #[rstest]
    #[case::disabled("ssl_verify=False", Verify::Disabled)]
    #[case::disabled_string("ssl_verify='False'", Verify::Disabled)]
    #[case::enabled_string("ssl_verify='true'", Verify::BuiltInRoots)]
    #[case::bundle("ssl_verify='/tmp/ca.pem'", Verify::CaBundle("/tmp/ca.pem".into()))]
    fn ssl_verify_global_resolves_like_get_ssl_verify(
        #[case] overrides: &str,
        #[case] expected: Verify,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let settings = settings(&python_settings(py, overrides)).unwrap();
            let config = HttpClientConfig::resolve(&settings).config;
            assert_eq!(config.verify, expected);
        });
    }

    #[test]
    fn ssl_context_global_is_ignored_so_environment_and_defaults_apply() {
        Python::initialize();
        Python::attach(|py| {
            let settings = settings(&python_settings(py, "ssl_verify=object()")).unwrap();
            assert_eq!(settings.ssl_verify, None);
        });
    }

    #[test]
    fn unsupported_settings_are_reported_once_per_process() {
        let reported = Mutex::default();
        let curve = Unsupported::EcdhCurve("secp521r1".into());
        let level = Unsupported::SecurityLevel("@SECLEVEL=1".into());
        assert_eq!(
            unreported(&reported, vec![curve.clone(), level.clone()]),
            [curve.clone(), level]
        );
        assert_eq!(unreported(&reported, vec![curve]), []);
    }

    #[test]
    fn mistyped_python_settings_decline_instead_of_raising() {
        Python::initialize();
        Python::attach(|py| {
            let error = settings(&python_settings(py, "force_ipv4='yes'")).unwrap_err();
            assert!(error.is_instance_of::<RustBridgeDeclined>(py));
        });
    }

    #[test]
    fn call_ssl_verify_beats_the_configured_and_environment_value() {
        Python::initialize();
        Python::attach(|py| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("ssl_verify", false).unwrap();
            let configured = HttpSettings {
                ssl_verify: Some(SslVerify::Enabled),
                ..HttpSettings::default()
            };
            let settings = for_call(configured, call_ssl_verify(&kwargs).unwrap(), true);
            assert_eq!(settings.ssl_verify, Some(SslVerify::Disabled));
        });
    }

    #[test]
    fn absent_call_ssl_verify_keeps_the_configured_value() {
        Python::initialize();
        Python::attach(|py| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("ssl_verify", py.None()).unwrap();
            let configured = HttpSettings {
                ssl_verify: Some(SslVerify::Disabled),
                ..HttpSettings::default()
            };
            let settings = for_call(configured.clone(), call_ssl_verify(&kwargs).unwrap(), true);
            assert_eq!(settings, configured);
        });
    }

    #[test]
    fn live_ssl_context_argument_is_ignored_so_the_configured_value_applies() {
        Python::initialize();
        Python::attach(|py| {
            let kwargs = PyDict::new(py);
            kwargs
                .set_item("ssl_verify", py.eval(c"object()", None, None).unwrap())
                .unwrap();
            let configured = HttpSettings {
                ssl_verify: Some(SslVerify::Disabled),
                ..HttpSettings::default()
            };
            let settings = for_call(configured.clone(), call_ssl_verify(&kwargs).unwrap(), true);
            assert_eq!(settings, configured);
        });
    }

    #[rstest]
    #[case::asynchronous(true, false)]
    #[case::synchronous(false, true)]
    fn synchronous_calls_honor_environment_proxies_even_when_aiohttp_opts_out(
        #[case] asynchronous: bool,
        #[case] expected: bool,
    ) {
        let opted_out = HttpSettings {
            ignore_proxy_env: true,
            ..HttpSettings::default()
        };
        let settings = for_call(opted_out, None, asynchronous);
        let config = HttpClientConfig::resolve(&settings).config;
        assert_eq!(config.trust_proxy_env, expected);
    }
}
