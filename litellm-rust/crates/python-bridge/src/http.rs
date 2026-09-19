use std::{
    path::{Path, PathBuf},
    sync::{Arc, LazyLock},
};

use litellm_http::{HttpClientConfig, HttpClientPool, HttpSettings, SslVerify};
use litellm_llms::custom_httpx::media::PublicDnsResolver;
use pyo3::{prelude::*, types::PyDict};

use crate::{errors::RustBridgeDeclined, python_settings::PythonSettings};

static POOL: LazyLock<HttpClientPool> =
    LazyLock::new(|| HttpClientPool::new(Arc::new(PublicDnsResolver)));

pub(crate) fn pool() -> &'static HttpClientPool {
    &POOL
}

pub(crate) fn call_config(
    py: Python<'_>,
    kwargs: &Bound<'_, PyDict>,
    asynchronous: bool,
) -> PyResult<HttpClientConfig> {
    decline_live_client(kwargs)?;
    decline_custom_url_policy(&PythonSettings::UrlPolicy.read(py)?)?;
    let configured = settings(&PythonSettings::Http.read(py)?)?
        .with_environment(&|name| std::env::var(name).ok());
    let settings = for_call(configured, call_ssl_verify(kwargs)?, asynchronous)
        .without_missing_files(&|path: &Path| path.exists());
    HttpClientConfig::resolve(&settings)
        .map_err(|error| RustBridgeDeclined::new_err(error.to_string()))
}

fn call_ssl_verify(kwargs: &Bound<'_, PyDict>) -> PyResult<Option<SslVerify>> {
    kwargs
        .get_item("ssl_verify")?
        .filter(|value| !value.is_none())
        .map(|value| ssl_verify(&value, "the ssl_verify argument"))
        .transpose()
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

fn decline_live_client(kwargs: &Bound<'_, PyDict>) -> PyResult<()> {
    if kwargs
        .get_item("client")?
        .is_some_and(|value| !value.is_none())
    {
        return Err(RustBridgeDeclined::new_err(
            "client is a live Python HTTP client and cannot be used by the Rust route",
        ));
    }
    Ok(())
}

#[derive(FromPyObject)]
struct PythonUrlPolicy {
    user_url_validation: bool,
    user_url_allowed_hosts: Vec<String>,
}

fn decline_custom_url_policy(value: &Bound<'_, PyAny>) -> PyResult<()> {
    match value.extract::<PythonUrlPolicy>() {
        Ok(policy) if policy.user_url_validation && policy.user_url_allowed_hosts.is_empty() => {
            Ok(())
        }
        Ok(_) | Err(_) => Err(RustBridgeDeclined::new_err(
            "litellm.user_url_validation / user_url_allowed_hosts are applied by the Python route",
        )),
    }
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
        ssl_verify: Some(ssl_verify(&python.ssl_verify, "litellm.ssl_verify")?),
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

fn ssl_verify(value: &Bound<'_, PyAny>, source: &str) -> PyResult<SslVerify> {
    if let Ok(enabled) = value.extract::<bool>() {
        return Ok(if enabled {
            SslVerify::Enabled
        } else {
            SslVerify::Disabled
        });
    }
    if let Ok(path) = value.extract::<String>() {
        return Ok(SslVerify::parse(&path));
    }
    Err(RustBridgeDeclined::new_err(format!(
        "{source} is a live Python object and cannot be used by the Rust route"
    )))
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
            let config = HttpClientConfig::resolve(&settings).unwrap();
            assert_eq!(config.verify, expected);
        });
    }

    #[test]
    fn ssl_context_global_declines_instead_of_being_dropped() {
        Python::initialize();
        Python::attach(|py| {
            let error = settings(&python_settings(py, "ssl_verify=object()")).unwrap_err();
            assert!(error.is_instance_of::<RustBridgeDeclined>(py));
            assert!(error.value(py).to_string().contains("litellm.ssl_verify"));
        });
    }

    fn url_policy<'py>(py: Python<'py>, fields: &str) -> Bound<'py, PyAny> {
        let source = std::ffi::CString::new(format!(
            "import types\npolicy = types.SimpleNamespace({fields})"
        ))
        .unwrap();
        let locals = PyDict::new(py);
        py.run(&source, Some(&locals), Some(&locals)).unwrap();
        locals.get_item("policy").unwrap().unwrap()
    }

    #[test]
    fn default_url_policy_stays_on_the_rust_route() {
        Python::initialize();
        Python::attach(|py| {
            let policy = url_policy(py, "user_url_validation=True, user_url_allowed_hosts=[]");
            decline_custom_url_policy(&policy).unwrap();
        });
    }

    #[rstest]
    #[case::validation_off("user_url_validation=False, user_url_allowed_hosts=[]")]
    #[case::allowlist("user_url_validation=True, user_url_allowed_hosts=['docs.internal']")]
    #[case::mistyped("user_url_validation=True, user_url_allowed_hosts=None")]
    fn custom_url_policy_declines_so_python_applies_it(#[case] fields: &str) {
        Python::initialize();
        Python::attach(|py| {
            let error = decline_custom_url_policy(&url_policy(py, fields)).unwrap_err();
            assert!(error.is_instance_of::<RustBridgeDeclined>(py));
        });
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
    fn live_ssl_context_argument_declines() {
        Python::initialize();
        Python::attach(|py| {
            let kwargs = PyDict::new(py);
            kwargs
                .set_item("ssl_verify", py.eval(c"object()", None, None).unwrap())
                .unwrap();
            let error = call_ssl_verify(&kwargs).unwrap_err();
            assert!(error.is_instance_of::<RustBridgeDeclined>(py));
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
        let config = HttpClientConfig::resolve(&settings).unwrap();
        assert_eq!(config.trust_proxy_env, expected);
    }

    #[test]
    fn live_python_client_declines_before_dispatch() {
        Python::initialize();
        Python::attach(|py| {
            let kwargs = PyDict::new(py);
            kwargs
                .set_item("client", py.eval(c"object()", None, None).unwrap())
                .unwrap();
            let error = decline_live_client(&kwargs).unwrap_err();
            assert!(error.is_instance_of::<RustBridgeDeclined>(py));
        });
    }

    #[rstest]
    #[case::absent_client("{}")]
    #[case::none_client("{'client': None}")]
    #[case::proxy_shared_session("{'shared_session': object()}")]
    fn calls_without_a_python_client_stay_on_the_rust_route(#[case] kwargs: &str) {
        Python::initialize();
        Python::attach(|py| {
            let source = std::ffi::CString::new(kwargs).unwrap();
            let kwargs = py.eval(&source, None, None).unwrap();
            decline_live_client(kwargs.cast::<PyDict>().unwrap()).unwrap();
        });
    }
}
