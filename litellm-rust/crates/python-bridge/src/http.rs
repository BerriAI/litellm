use std::{path::PathBuf, sync::LazyLock};

use litellm_http::{HttpClientConfig, HttpClientPool, HttpSettings, SslVerify};
use pyo3::{prelude::*, types::PyDict};

use crate::errors::RustBridgeDeclined;

static POOL: LazyLock<HttpClientPool> = LazyLock::new(HttpClientPool::new);

/// Keyword arguments that carry a live Python HTTP client or session. They cannot cross into
/// Rust, so a call that supplies one stays on the Python path.
const LIVE_CLIENT_ARGUMENTS: [&str; 3] = ["client", "shared_session", "aclient_session"];

pub(crate) fn pool() -> &'static HttpClientPool {
    &POOL
}

/// The client configuration for one call: the process settings from the `litellm` module and
/// the environment, narrowed by the call's own `ssl_verify`.
pub(crate) fn call_config(
    py: Python<'_>,
    kwargs: &Bound<'_, PyDict>,
) -> PyResult<HttpClientConfig> {
    decline_live_clients(kwargs)?;
    let settings = settings(py.import("litellm")?.as_any())?
        .with_environment(&|name| std::env::var(name).ok());
    let per_call = kwargs
        .get_item("ssl_verify")?
        .map(|value| ssl_verify(&value, "ssl_verify"))
        .transpose()?
        .flatten();
    HttpClientConfig::resolve(&settings, per_call.as_ref())
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

/// Read the `litellm.*` globals `http_handler.py` consults. `globals` is the `litellm` module in
/// production and any attribute holder in tests.
pub(crate) fn settings(globals: &Bound<'_, PyAny>) -> PyResult<HttpSettings> {
    Ok(HttpSettings {
        ssl_verify: ssl_verify(&globals.getattr("ssl_verify")?, "litellm.ssl_verify")?,
        ssl_certificate: optional_path(globals, "ssl_certificate")?,
        ssl_security_level: globals.getattr("ssl_security_level")?.extract()?,
        ssl_ecdh_curve: globals.getattr("ssl_ecdh_curve")?.extract()?,
        force_ipv4: globals.getattr("force_ipv4")?.extract()?,
        http2: globals.getattr("http2")?.extract()?,
        trust_proxy_env: globals.getattr("aiohttp_trust_env")?.extract()?,
        ..HttpSettings::default()
    })
}

fn optional_path(globals: &Bound<'_, PyAny>, name: &str) -> PyResult<Option<PathBuf>> {
    Ok(globals
        .getattr(name)?
        .extract::<Option<String>>()?
        .map(PathBuf::from))
}

fn ssl_verify(value: &Bound<'_, PyAny>, name: &str) -> PyResult<Option<SslVerify>> {
    if value.is_none() {
        return Ok(None);
    }
    if let Ok(enabled) = value.extract::<bool>() {
        return Ok(Some(if enabled {
            SslVerify::Enabled
        } else {
            SslVerify::Disabled
        }));
    }
    if let Ok(path) = value.extract::<String>() {
        return Ok(Some(SslVerify::CaBundle(PathBuf::from(path))));
    }
    Err(RustBridgeDeclined::new_err(format!(
        "{name} is a live Python object and cannot be used by the Rust route"
    )))
}

#[cfg(test)]
mod tests {
    use litellm_http::Verify;
    use rstest::rstest;

    use super::*;

    fn eval<'py>(py: Python<'py>, source: &std::ffi::CStr) -> Bound<'py, PyDict> {
        let locals = PyDict::new(py);
        py.run(source, Some(&locals), Some(&locals)).unwrap();
        locals
    }

    fn globals<'py>(py: Python<'py>, overrides: &str) -> Bound<'py, PyAny> {
        let source = format!(
            "
import types
globals = types.SimpleNamespace(
    ssl_verify=True,
    ssl_certificate=None,
    ssl_security_level=None,
    ssl_ecdh_curve=None,
    force_ipv4=False,
    http2=False,
    aiohttp_trust_env=False,
)
{overrides}
"
        );
        let source = std::ffi::CString::new(source).unwrap();
        eval(py, &source).get_item("globals").unwrap().unwrap()
    }

    #[test]
    fn default_globals_produce_default_settings_with_verification_on() {
        Python::initialize();
        Python::attach(|py| {
            let settings = settings(&globals(py, "")).unwrap();
            assert_eq!(
                settings,
                HttpSettings {
                    ssl_verify: Some(SslVerify::Enabled),
                    ..HttpSettings::default()
                }
            );
        });
    }

    #[test]
    fn globals_flow_into_settings() {
        Python::initialize();
        Python::attach(|py| {
            let settings = settings(&globals(
                py,
                "
globals.ssl_verify = '/etc/ssl/corp.pem'
globals.ssl_certificate = '/etc/ssl/client.pem'
globals.ssl_security_level = '2'
globals.ssl_ecdh_curve = 'X25519'
globals.force_ipv4 = True
globals.http2 = True
globals.aiohttp_trust_env = True
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
                    trust_proxy_env: true,
                    ..HttpSettings::default()
                }
            );
        });
    }

    #[test]
    fn disabled_verification_global_resolves_to_disabled() {
        Python::initialize();
        Python::attach(|py| {
            let settings = settings(&globals(py, "globals.ssl_verify = False")).unwrap();
            assert_eq!(settings.ssl_verify, Some(SslVerify::Disabled));
            let config = HttpClientConfig::resolve(&settings, None).unwrap();
            assert_eq!(config.verify, Verify::Disabled);
        });
    }

    #[test]
    fn ssl_context_global_declines_instead_of_being_dropped() {
        Python::initialize();
        Python::attach(|py| {
            let error = settings(&globals(py, "globals.ssl_verify = object()")).unwrap_err();
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

    #[rstest]
    #[case::disabled(c"False", Some(SslVerify::Disabled))]
    #[case::enabled(c"True", Some(SslVerify::Enabled))]
    #[case::bundle(c"'/tmp/ca.pem'", Some(SslVerify::CaBundle("/tmp/ca.pem".into())))]
    #[case::unset(c"None", None)]
    fn per_call_ssl_verify_values_project(
        #[case] source: &std::ffi::CStr,
        #[case] expected: Option<SslVerify>,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let value = py.eval(source, None, None).unwrap();
            assert_eq!(ssl_verify(&value, "ssl_verify").unwrap(), expected);
        });
    }
}
