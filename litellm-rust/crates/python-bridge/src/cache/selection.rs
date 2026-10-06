use super::python;
use litellm_cache_response::{CachePolicy, ResponseCacheConfig, ScopedCache};
use litellm_core::caching::Cachable;
use litellm_host::{
    machine::{HostServices, MachineFault},
    protocol::Protocol,
};
use pyo3::{prelude::*, types::PyDict};

pub(crate) struct PythonCached<P>(std::marker::PhantomData<P>);

impl<P: Protocol> Protocol for PythonCached<P> {
    type Request = (P::Request, Option<PythonCacheConfig>);
    type Response = P::Response;
    type Error = P::Error;
    type HostCall = python::CacheCall;
    type Chunk = P::Chunk;
    type StreamHead = P::StreamHead;
}

pub(crate) struct PythonCacheConfig {
    policy: CachePolicy,
    surface: &'static str,
    config: ResponseCacheConfig,
}

impl PythonCacheConfig {
    pub(crate) fn attach<P: Protocol<HostCall = python::CacheCall>>(
        self,
        services: HostServices<P>,
    ) -> (ScopedCache, CachePolicy)
    where
        P::Error: From<MachineFault>,
    {
        (
            ScopedCache::shared(python::service(services, self.surface, self.config)),
            self.policy,
        )
    }
}

#[derive(Clone, Copy)]
enum CacheRule {
    Configured,
    RequestAllowsCaching,
    SupportsCallType,
    PythonPolicyAllowsCaching,
}

const CACHE_RULES: [CacheRule; 4] = [
    CacheRule::Configured,
    CacheRule::RequestAllowsCaching,
    CacheRule::SupportsCallType,
    CacheRule::PythonPolicyAllowsCaching,
];

impl CacheRule {
    fn allows(
        self,
        cache: &Bound<'_, PyAny>,
        arguments: &Bound<'_, PyDict>,
        call_type: &str,
    ) -> PyResult<bool> {
        match self {
            Self::Configured => Ok(!cache.is_none()),
            Self::RequestAllowsCaching => Ok(!arguments
                .get_item("caching")?
                .is_some_and(|value| value.is(pyo3::types::PyBool::new(cache.py(), false)))),
            Self::SupportsCallType => {
                let supported = cache.getattr("supported_call_types")?;
                Ok(!supported.is_none() && supported.contains(call_type)?)
            }
            Self::PythonPolicyAllowsCaching => cache
                .call_method("should_use_cache", (), Some(arguments))?
                .extract::<bool>(),
        }
    }
}

fn selected_cache<'py>(
    cache: Bound<'py, PyAny>,
    arguments: &Bound<'py, PyDict>,
    call_type: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let allowed = CACHE_RULES
        .into_iter()
        .try_fold(true, |allowed, rule| -> PyResult<bool> {
            Ok(allowed && rule.allows(&cache, arguments, call_type)?)
        })?;
    Ok(allowed.then_some(cache))
}

pub(crate) fn configure_python_cache<P: Cachable>(
    host: &mut python::PythonCache,
    py: Python<'_>,
    arguments: &Bound<'_, PyDict>,
    call_type: &str,
) -> PyResult<Option<PythonCacheConfig>> {
    let configured = py.import("litellm")?.getattr("cache")?;
    let Some(cache) = selected_cache(configured, arguments, call_type)? else {
        return Ok(None);
    };
    let controls = arguments
        .get_item("cache")?
        .filter(|value| !value.is_none());
    let boolean = |name: &str| -> PyResult<bool> {
        controls
            .as_ref()
            .map(|value| value.cast::<PyDict>()?.get_item(name))
            .transpose()?
            .flatten()
            .map(|value| value.extract())
            .transpose()
            .map(|value| value.unwrap_or(false))
    };
    let policy = CachePolicy {
        no_cache: boolean("no-cache")?,
        no_store: boolean("no-store")?,
        ..CachePolicy::default()
    };
    let namespace = cache
        .getattr_opt("namespace")?
        .filter(|value| !value.is_none())
        .map(|value| value.extract())
        .transpose()?
        .unwrap_or_default();
    host.bind(cache, arguments);
    Ok(Some(PythonCacheConfig {
        policy,
        surface: P::SURFACE,
        config: ResponseCacheConfig {
            namespace,
            supports_isolated_scope: false,
            ..ResponseCacheConfig::default()
        },
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::{fixture, rstest};
    use std::ffi::CString;

    #[fixture]
    fn selection_cache() -> Py<PyModule> {
        Python::initialize();
        Python::attach(|py| {
            PyModule::from_code(
                py,
                c"class SelectionCache:
    def __init__(self, supported=('completion',), default_on=True, fail_at=None):
        self.supported = supported
        self.default_on = default_on
        self.fail_at = fail_at
        self.calls = ()
        self.error = ValueError('cache selection failed')

    @property
    def supported_call_types(self):
        self.calls += ('supported',)
        if self.fail_at == 'supported':
            raise self.error
        return self.supported

    def should_use_cache(self, **kwargs):
        self.calls += ('policy',)
        if self.fail_at == 'policy':
            raise self.error
        return self.default_on or kwargs.get('cache', {}).get('use-cache') is True
",
                c"cache_selection_test.py",
                c"cache_selection_test",
            )
            .unwrap()
            .unbind()
        })
    }

    #[rstest]
    #[case::unconfigured("None", "{}", false, vec![])]
    #[case::unconfigured_with_opt_in("None", "{'caching': True}", false, vec![])]
    #[case::request_disabled(
        "SelectionCache(fail_at='supported')", "{'caching': False}", false, vec![]
    )]
    #[case::unsupported(
        "SelectionCache(supported=('embedding',), fail_at='policy')",
        "{}", false, vec!["supported"]
    )]
    #[case::empty_supported("SelectionCache(supported=())", "{}", false, vec!["supported"])]
    #[case::missing_supported("SelectionCache(supported=None)", "{}", false, vec!["supported"])]
    #[case::default_on("SelectionCache()", "{}", true, vec!["supported", "policy"])]
    #[case::default_off(
        "SelectionCache(default_on=False)", "{}", false, vec!["supported", "policy"]
    )]
    #[case::caching_flag_does_not_opt_in(
        "SelectionCache(default_on=False)", "{'caching': True}", false,
        vec!["supported", "policy"]
    )]
    #[case::explicit_opt_in(
        "SelectionCache(default_on=False)", "{'cache': {'use-cache': True}}", true,
        vec!["supported", "policy"]
    )]
    #[case::request_disabled_with_opt_in(
        "SelectionCache(fail_at='supported')",
        "{'caching': False, 'cache': {'use-cache': True}}", false, vec![]
    )]
    fn selection_short_circuits_python_checks(
        selection_cache: Py<PyModule>,
        #[case] cache_expression: &str,
        #[case] arguments_expression: &str,
        #[case] expected_selected: bool,
        #[case] expected_calls: Vec<&str>,
    ) {
        Python::attach(|py| {
            let globals = selection_cache.bind(py).dict();
            let cache = py
                .eval(
                    &CString::new(cache_expression).unwrap(),
                    Some(&globals),
                    None,
                )
                .unwrap();
            let arguments = py
                .eval(&CString::new(arguments_expression).unwrap(), None, None)
                .unwrap()
                .cast_into::<PyDict>()
                .unwrap();
            let selected = selected_cache(cache.clone(), &arguments, "completion").unwrap();
            assert_eq!(selected.is_some(), expected_selected);
            if let Some(selected) = selected {
                assert!(selected.is(&cache));
            }
            let calls: Vec<String> = if cache.is_none() {
                Vec::new()
            } else {
                cache.getattr("calls").unwrap().extract().unwrap()
            };
            assert_eq!(calls, expected_calls);
        });
    }

    #[rstest]
    #[case::supported("supported", vec!["supported"])]
    #[case::policy("policy", vec!["supported", "policy"])]
    fn selection_preserves_python_errors(
        selection_cache: Py<PyModule>,
        #[case] fail_at: &str,
        #[case] expected_calls: Vec<&str>,
    ) {
        Python::attach(|py| {
            let arguments = PyDict::new(py);
            let constructor = selection_cache.bind(py).getattr("SelectionCache").unwrap();
            let options = PyDict::new(py);
            options.set_item("fail_at", fail_at).unwrap();
            let cache = constructor.call((), Some(&options)).unwrap();
            let error = selected_cache(cache.clone(), &arguments, "completion").unwrap_err();
            assert!(error.value(py).is(cache.getattr("error").unwrap()));
            let calls: Vec<String> = cache.getattr("calls").unwrap().extract().unwrap();
            assert_eq!(calls, expected_calls);
        });
    }
}
