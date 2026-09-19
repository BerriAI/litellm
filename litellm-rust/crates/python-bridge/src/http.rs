use std::{
    path::PathBuf,
    sync::{Arc, LazyLock},
};

use litellm_http::{HttpClientConfig, HttpClientPool, HttpSettings, SslVerify};
use litellm_llms::custom_httpx::media::PublicDnsResolver;
use pyo3::{prelude::*, types::PyDict};

use crate::{errors::RustBridgeDeclined, python_settings::PythonSettings};

static POOL: LazyLock<HttpClientPool> =
    LazyLock::new(|| HttpClientPool::new(Arc::new(PublicDnsResolver)));

/// Keyword arguments that carry a live Python HTTP client or session. They cannot cross into
/// Rust, so a call that supplies one stays on the Python path.
const LIVE_CLIENT_ARGUMENTS: [&str; 3] = ["client", "shared_session", "aclient_session"];

pub(crate) fn pool() -> &'static HttpClientPool {
    &POOL
}

/// The client configuration for one call: the `litellm.*` HTTP settings with the environment
/// overlaid, the same way `http_handler.py` combines them.
pub(crate) fn call_config(
    py: Python<'_>,
    kwargs: &Bound<'_, PyDict>,
) -> PyResult<HttpClientConfig> {
    decline_live_clients(kwargs)?;
    let settings = settings(&PythonSettings::Http.read(py)?)?
        .with_environment(&|name| std::env::var(name).ok());
    HttpClientConfig::resolve(&settings)
        .map_err(|error| RustBridgeDeclined::new_err(error.to_string()))
}

pub(crate) fn decline_live_clients(kwargs: &Bound<'_, PyDict>) -> PyResult<()> {
    for name in LIVE_CLIENT_ARGUMENTS {
        if kwargs.get_item(name)?.is_some_and(|value| !value.is_none()) {
            return Err(RustBridgeDeclined::new_err(format!(
                "{name} is a live Python HTTP client and cannot be used by the Rust route"
            )));
        }
    }
    Ok(())
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
    user_agent: String,
}

fn settings(value: &Bound<'_, PyAny>) -> PyResult<HttpSettings> {
    let python: PythonHttpSettings = value.extract()?;
    Ok(HttpSettings {
        ssl_verify: Some(ssl_verify(&python.ssl_verify)?),
        ssl_certificate: python.ssl_certificate.map(PathBuf::from),
        ssl_security_level: python.ssl_security_level,
        ssl_ecdh_curve: python.ssl_ecdh_curve,
        force_ipv4: python.force_ipv4,
        http2: python.http2,
        user_agent: Some(python.user_agent),
        trust_proxy_env: python.aiohttp_trust_env,
        ..HttpSettings::default()
    })
}

fn ssl_verify(value: &Bound<'_, PyAny>) -> PyResult<SslVerify> {
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
    Err(RustBridgeDeclined::new_err(
        "litellm.ssl_verify is a live Python object and cannot be used by the Rust route",
    ))
}

#[cfg(test)]
mod tests {
    use litellm_http::Verify;
    use rstest::rstest;

    use super::*;
    use crate::python_settings::CONTRACT;

    /// A stand-in for `http_settings()` carrying exactly the fields the contract declares, so a
    /// field Rust reads but Python does not return fails here.
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
                    user_agent: Some("litellm/9.9.9".into()),
                    trust_proxy_env: true,
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

    #[rstest]
    #[case::client("client")]
    #[case::shared_session("shared_session")]
    #[case::aclient_session("aclient_session")]
    fn live_python_clients_decline_before_dispatch(#[case] name: &str) {
        Python::initialize();
        Python::attach(|py| {
            let kwargs = PyDict::new(py);
            kwargs
                .set_item(name, py.eval(c"object()", None, None).unwrap())
                .unwrap();
            let error = decline_live_clients(&kwargs).unwrap_err();
            assert!(error.is_instance_of::<RustBridgeDeclined>(py));
            assert!(error.value(py).to_string().contains(name));
        });
    }

    #[test]
    fn none_valued_client_arguments_are_not_live_clients() {
        Python::initialize();
        Python::attach(|py| {
            let kwargs = PyDict::new(py);
            for name in LIVE_CLIENT_ARGUMENTS {
                kwargs.set_item(name, py.None()).unwrap();
            }
            decline_live_clients(&kwargs).unwrap();
        });
    }
}
