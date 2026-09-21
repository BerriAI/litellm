use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    sync::{Arc, LazyLock, Mutex, PoisonError},
};

use litellm_core_utils::settings::ProcessEnvironment;
use litellm_http::{
    HttpClientConfig, HttpClientPool, HttpSettings, HttpSettingsLayer, Resolution, SslVerify,
    TlsSource, Unsupported,
    media::{PublicDnsResolver, UrlPolicy},
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

use crate::{coercion::Field, python_settings::PythonSettings};

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
    let settings = HttpSettings::from_layers([
        for_call(call_ssl_verify(kwargs)?, asynchronous),
        HttpSettingsLayer::from_environment(&ProcessEnvironment),
        configured(&PythonSettings::Http.read(py)?)?,
    ])
    .without_missing_files(&|path: &Path| path.exists());
    let resolution = Resolution::from(&settings);
    for unsupported in unreported(&REPORTED_UNSUPPORTED, resolution.unsupported) {
        PythonSettings::warn(py, &unsupported.to_string())?;
    }
    Ok(resolution.config)
}

pub(crate) fn client_error(error: litellm_http::Error) -> PyErr {
    match error {
        litellm_http::Error::Read {
            tls_source: TlsSource::ClientIdentity,
            ..
        }
        | litellm_http::Error::InvalidPem {
            tls_source: TlsSource::ClientIdentity,
            ..
        } => PyValueError::new_err(
            "http_settings.ssl_certificate: expected a readable PEM certificate and private key",
        ),
        litellm_http::Error::Read {
            tls_source: TlsSource::CaBundle,
            ..
        }
        | litellm_http::Error::InvalidPem {
            tls_source: TlsSource::CaBundle,
            ..
        } => PyValueError::new_err("http_settings.ssl_verify: expected a readable PEM CA bundle"),
        _ => PyValueError::new_err("http_settings: native HTTP client configuration is invalid"),
    }
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
    project_url_policy(&PythonSettings::UrlPolicy.read(py)?)
}

fn project_url_policy(value: &Bound<'_, PyAny>) -> PyResult<UrlPolicy> {
    Ok(UrlPolicy {
        validate: Field::read(value, "url_policy.user_url_validation")?
            .truthy()?
            .0,
        allowed_hosts: Field::read(value, "url_policy.user_url_allowed_hosts")?
            .host_collection()?
            .0,
    })
}

fn call_ssl_verify(kwargs: &Bound<'_, PyDict>) -> PyResult<Option<SslVerify>> {
    match kwargs.get_item("ssl_verify")? {
        Some(value) => Ok(Field::new("request.ssl_verify", value).ssl_verify()?.0),
        None => Ok(None),
    }
}

fn for_call(call_ssl_verify: Option<SslVerify>, asynchronous: bool) -> HttpSettingsLayer {
    HttpSettingsLayer {
        ssl_verify: call_ssl_verify,
        disable_aiohttp_transport: (!asynchronous).then_some(true),
        ..HttpSettingsLayer::default()
    }
}

fn configured(value: &Bound<'_, PyAny>) -> PyResult<HttpSettingsLayer> {
    Ok(HttpSettingsLayer {
        ssl_verify: Field::read(value, "http_settings.ssl_verify")?
            .ssl_verify()?
            .0,
        ssl_certificate: Field::read(value, "http_settings.ssl_certificate")?
            .optional_strict_string()?
            .0
            .map(PathBuf::from),
        ssl_security_level: Field::read(value, "http_settings.ssl_security_level")?
            .tuning_string()?
            .0,
        ssl_ecdh_curve: Field::read(value, "http_settings.ssl_ecdh_curve")?
            .tuning_string()?
            .0,
        force_ipv4: Some(Field::read(value, "http_settings.force_ipv4")?.truthy()?.0),
        http2: Some(Field::read(value, "http_settings.http2")?.exact_true().0),
        aiohttp_trust_env: Some(
            Field::read(value, "http_settings.aiohttp_trust_env")?
                .truthy()?
                .0,
        ),
        disable_aiohttp_trust_env: Some(
            Field::read(value, "http_settings.disable_aiohttp_trust_env")?
                .truthy()?
                .0,
        ),
        disable_aiohttp_transport: Some(
            Field::read(value, "http_settings.disable_aiohttp_transport")?
                .exact_true()
                .0,
        ),
        user_agent: Some(Field::read(value, "http_settings.user_agent")?.schema_string()?),
        ..HttpSettingsLayer::default()
    })
}

#[cfg(test)]
mod tests {
    use litellm_http::Verify;
    use pyo3::exceptions::PyRuntimeError;
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
settings = types.SimpleNamespace(**{{name: defaults[name] for name in json.loads(contract)['http_settings']['fields']}})
"
        );
        let locals = PyDict::new(py);
        locals.set_item("contract", CONTRACT).unwrap();
        let source = std::ffi::CString::new(source).unwrap();
        py.run(&source, Some(&locals), Some(&locals)).unwrap();
        locals.get_item("settings").unwrap().unwrap()
    }

    #[test]
    fn default_python_settings_resolve_to_default_settings_with_verification_on() {
        Python::initialize();
        Python::attach(|py| {
            let layer = configured(&python_settings(py, "")).unwrap();
            assert_eq!(
                HttpSettings::from_layers([layer]),
                HttpSettings {
                    ssl_verify: Some(SslVerify::Enabled),
                    user_agent: Some("litellm/test".into()),
                    ..HttpSettings::default()
                }
            );
        });
    }

    #[test]
    fn client_error_uses_tls_source_when_paths_match() {
        Python::initialize();
        Python::attach(|py| {
            let path = PathBuf::from("/shared.pem");
            let ca_error = client_error(litellm_http::Error::InvalidPem {
                path: path.clone(),
                message: "invalid".into(),
                tls_source: TlsSource::CaBundle,
            });
            assert_eq!(
                ca_error.to_string(),
                "ValueError: http_settings.ssl_verify: expected a readable PEM CA bundle"
            );
            let client_error = client_error(litellm_http::Error::InvalidPem {
                path,
                message: "invalid".into(),
                tls_source: TlsSource::ClientIdentity,
            });
            assert!(client_error.is_instance_of::<PyValueError>(py));
            assert_eq!(
                client_error.to_string(),
                "ValueError: http_settings.ssl_certificate: expected a readable PEM certificate and private key"
            );
        });
    }

    #[test]
    fn python_settings_flow_into_the_configured_layer() {
        Python::initialize();
        Python::attach(|py| {
            let layer = configured(&python_settings(
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
                layer,
                HttpSettingsLayer {
                    ssl_verify: Some(SslVerify::CaBundle("/etc/ssl/corp.pem".into())),
                    ssl_certificate: Some("/etc/ssl/client.pem".into()),
                    ssl_security_level: Some("2".into()),
                    ssl_ecdh_curve: Some("X25519".into()),
                    force_ipv4: Some(true),
                    http2: Some(true),
                    aiohttp_trust_env: Some(true),
                    disable_aiohttp_trust_env: Some(true),
                    disable_aiohttp_transport: Some(true),
                    user_agent: Some("litellm/9.9.9".into()),
                    ..HttpSettingsLayer::default()
                }
            );
        });
    }

    #[test]
    fn user_agent_environment_variable_beats_the_python_default() {
        Python::initialize();
        Python::attach(|py| {
            let settings = HttpSettings::from_layers([
                HttpSettingsLayer::from_environment(&|name: &str| {
                    (name == "LITELLM_USER_AGENT").then(|| "operator/1".to_string())
                }),
                configured(&python_settings(py, "")).unwrap(),
            ]);
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
            let layer = configured(&python_settings(py, overrides)).unwrap();
            let config = Resolution::from(&HttpSettings::from_layers([layer])).config;
            assert_eq!(config.verify, expected);
        });
    }

    #[rstest]
    #[case("ssl_verify=object()")]
    #[case("ssl_verify=__import__('ssl').SSLContext(__import__('ssl').PROTOCOL_TLS_CLIENT)")]
    #[case("ssl_certificate=1")]
    fn invalid_http_configuration_is_terminal(#[case] overrides: &str) {
        Python::initialize();
        Python::attach(|py| {
            let error = configured(&python_settings(py, overrides)).unwrap_err();
            assert!(error.is_instance_of::<PyValueError>(py));
            assert!(error.to_string().contains("http_settings.ssl_"));
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
    fn mutable_globals_use_their_consumer_operations() {
        Python::initialize();
        Python::attach(|py| {
            let layer = configured(&python_settings(py,
                "force_ipv4='yes', http2=1, disable_aiohttp_transport=1, aiohttp_trust_env=[1], disable_aiohttp_trust_env=[], ssl_security_level=1, ssl_ecdh_curve=[]"
            )).unwrap();
            assert_eq!(layer.force_ipv4, Some(true));
            assert_eq!(layer.http2, Some(false));
            assert_eq!(layer.disable_aiohttp_transport, Some(false));
            assert_eq!(layer.aiohttp_trust_env, Some(true));
            assert_eq!(layer.disable_aiohttp_trust_env, Some(false));
            assert_eq!(layer.ssl_security_level, None);
            assert_eq!(layer.ssl_ecdh_curve, None);
            let error = configured(&python_settings(py, "user_agent=1")).unwrap_err();
            assert!(error.is_instance_of::<PyRuntimeError>(py));
        });
    }

    fn configured_ssl_verify(ssl_verify: SslVerify) -> HttpSettingsLayer {
        HttpSettingsLayer {
            ssl_verify: Some(ssl_verify),
            ..HttpSettingsLayer::default()
        }
    }

    #[test]
    fn call_ssl_verify_beats_the_configured_value() {
        Python::initialize();
        Python::attach(|py| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("ssl_verify", false).unwrap();
            let call = for_call(call_ssl_verify(&kwargs).unwrap(), true);
            let settings =
                HttpSettings::from_layers([call, configured_ssl_verify(SslVerify::Enabled)]);
            assert_eq!(settings.ssl_verify, Some(SslVerify::Disabled));
        });
    }

    #[test]
    fn absent_call_ssl_verify_keeps_the_configured_value() {
        Python::initialize();
        Python::attach(|py| {
            let kwargs = PyDict::new(py);
            kwargs.set_item("ssl_verify", py.None()).unwrap();
            let call = for_call(call_ssl_verify(&kwargs).unwrap(), true);
            let settings =
                HttpSettings::from_layers([call, configured_ssl_verify(SslVerify::Disabled)]);
            assert_eq!(settings.ssl_verify, Some(SslVerify::Disabled));
        });
    }

    #[test]
    fn live_ssl_context_argument_raises_instead_of_using_another_layer() {
        Python::initialize();
        Python::attach(|py| {
            let kwargs = PyDict::new(py);
            let ssl = py.import("ssl").unwrap();
            let context = ssl
                .getattr("SSLContext")
                .unwrap()
                .call1((ssl.getattr("PROTOCOL_TLS_CLIENT").unwrap(),))
                .unwrap();
            kwargs.set_item("ssl_verify", context).unwrap();
            let error = call_ssl_verify(&kwargs).unwrap_err();
            assert!(error.is_instance_of::<PyValueError>(py));
            assert!(error.to_string().contains("request.ssl_verify"));
            assert!(error.to_string().contains("SSLContext"));
        });
    }

    #[test]
    fn url_policy_uses_truthiness_and_normalized_owned_hosts() {
        Python::initialize();
        Python::attach(|py| {
            let value = py.eval(c"__import__('types').SimpleNamespace(user_url_validation=[], user_url_allowed_hosts=['B.test', 'a.test.', 'b.test'])", None, None).unwrap();
            assert_eq!(
                project_url_policy(&value).unwrap(),
                UrlPolicy {
                    validate: false,
                    allowed_hosts: vec!["a.test".into(), "b.test".into()],
                }
            );
        });
    }

    #[rstest]
    #[case::asynchronous(true, false)]
    #[case::synchronous(false, true)]
    fn synchronous_calls_honor_environment_proxies_even_when_aiohttp_opts_out(
        #[case] asynchronous: bool,
        #[case] expected: bool,
    ) {
        let opted_out = HttpSettingsLayer {
            disable_aiohttp_trust_env: Some(true),
            disable_aiohttp_transport: Some(false),
            ..HttpSettingsLayer::default()
        };
        let settings = HttpSettings::from_layers([for_call(None, asynchronous), opted_out]);
        assert_eq!(settings.trust_proxy_env, expected);
    }
}
