mod binding;
mod future;
pub(crate) mod native;
pub(crate) mod python;

use litellm_cache::Error;
use litellm_cache_response::{CacheOptions, CacheScope, ResponseCacheService, ScopedCache};
use litellm_host::{
    machine::{HostServices, MachineFault},
    protocol::Protocol,
};
use pyo3::{
    exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError},
    prelude::*,
    types::PyDict,
};
use std::sync::Arc;

pub(crate) use self::binding::ResolvedCache;

fn cache_error(error: Error) -> PyErr {
    match error {
        Error::InvalidEntry => PyValueError::new_err(error.to_string()),
        Error::UnsupportedOperation => PyNotImplementedError::new_err(error.to_string()),
        _ => PyRuntimeError::new_err(error.to_string()),
    }
}

enum Backend {
    Disabled,
    Native(Arc<dyn ResponseCacheService>),
    Python { namespace: String },
}

pub(crate) struct Selection {
    backend: Backend,
    options: CacheOptions,
}

impl Selection {
    pub(crate) fn attach<P: Protocol<HostCall = python::CacheCall>>(
        self,
        services: HostServices<P>,
    ) -> (Option<ScopedCache>, CacheOptions)
    where
        P::Error: From<MachineFault>,
    {
        let service = match self.backend {
            Backend::Disabled => None,
            Backend::Native(service) => Some(service),
            Backend::Python { namespace } => Some(python::service(services, namespace)),
        };
        (
            service.map(|service| ScopedCache::new(service, CacheScope::Shared)),
            self.options,
        )
    }
}

fn selected_cache<'py>(
    py: Python<'py>,
    kwargs: &Bound<'py, PyDict>,
    call_type: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let configured = py.import("litellm")?.getattr("cache")?;
    if configured.is_none()
        || kwargs
            .get_item("caching")?
            .is_some_and(|value| value.is(pyo3::types::PyBool::new(py, false)))
    {
        return Ok(None);
    }
    let supported = configured.getattr("supported_call_types")?;
    if supported.is_none() || !supported.contains(call_type)? {
        return Ok(None);
    }
    Ok(Some(configured))
}

pub(crate) fn admit_native(
    py: Python<'_>,
    kwargs: &Bound<'_, PyDict>,
    call_type: &str,
) -> PyResult<()> {
    if let Some(configured) = selected_cache(py, kwargs, call_type)?
        && native::v2::native_handle(&configured)?.is_none()
    {
        return Err(crate::errors::RustBridgeDeclined::new_err(
            "the configured cache requires Python inference",
        ));
    }
    Ok(())
}

pub(crate) fn configured_native(
    py: Python<'_>,
    kwargs: &Bound<'_, PyDict>,
    call_type: &str,
) -> PyResult<(
    Option<Arc<dyn ResponseCacheService>>,
    litellm_cache_response::CacheOptions,
)> {
    let Some(configured) = selected_cache(py, kwargs, call_type)? else {
        return Ok((
            None,
            litellm_cache_response::CacheOptions::new(litellm_cache_response::CacheScope::Shared),
        ));
    };
    native_configuration(&configured, kwargs)
}

fn native_configuration(
    configured: &Bound<'_, PyAny>,
    kwargs: &Bound<'_, PyDict>,
) -> PyResult<(Option<Arc<dyn ResponseCacheService>>, CacheOptions)> {
    if !configured
        .call_method("should_use_cache", (), Some(kwargs))?
        .extract::<bool>()?
    {
        return Ok((
            None,
            litellm_cache_response::CacheOptions::new(litellm_cache_response::CacheScope::Shared),
        ));
    }
    native::v2::configured(configured, kwargs)
}

pub(crate) fn configure(
    python: &mut python::PythonCache,
    py: Python<'_>,
    arguments: &Bound<'_, PyDict>,
    call_type: &str,
) -> PyResult<Selection> {
    let selected = selected_cache(py, arguments, call_type)?;
    let Some(cache) = selected else {
        return Ok(Selection {
            backend: Backend::Disabled,
            options: CacheOptions::new(CacheScope::Shared),
        });
    };
    if native::v2::native_handle(&cache)?.is_some() {
        let (native, options) = native_configuration(&cache, arguments)?;
        return Ok(Selection {
            backend: native.map_or(Backend::Disabled, Backend::Native),
            options,
        });
    }
    let enabled = cache
        .call_method("should_use_cache", (), Some(arguments))?
        .extract::<bool>()?;
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
    let options = CacheOptions {
        no_cache: boolean("no-cache")?,
        no_store: boolean("no-store")?,
        ..CacheOptions::new(CacheScope::Shared)
    };
    let namespace = cache
        .getattr_opt("namespace")?
        .filter(|value| !value.is_none())
        .map(|value| value.extract())
        .transpose()?
        .unwrap_or_default();
    python.bind(cache, arguments);
    Ok(Selection {
        backend: if enabled {
            Backend::Python { namespace }
        } else {
            Backend::Disabled
        },
        options,
    })
}
