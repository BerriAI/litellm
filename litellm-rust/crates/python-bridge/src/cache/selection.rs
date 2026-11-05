use super::{HandleError, python};
use litellm_cache_response::{CacheOptions, CachePolicy, CacheScope, ScopedCache};
use litellm_host::{
    machine::{HostServices, MachineFault},
    protocol::Protocol,
};
use pyo3::{prelude::*, types::PyDict};
const CACHE_HOST_MODULE: &str = "litellm.rust_bridge.host.cache";

pub(crate) struct Cached<P>(std::marker::PhantomData<P>);

impl<P: Protocol> Protocol for Cached<P> {
    type Request = (P::Request, Selection);
    type Response = P::Response;
    type Error = P::Error;
    type HostCall = python::CacheCall;
    type Chunk = P::Chunk;
    type StreamHead = P::StreamHead;
}

enum Backend {
    Disabled,
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
        P::Error: From<MachineFault> + std::error::Error,
    {
        let service = match self.backend {
            Backend::Disabled => None,
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
    let configured = py
        .import(CACHE_HOST_MODULE)?
        .getattr("configured_cache")?
        .call0()?;
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

pub(crate) fn ensure_cache_disabled(
    py: Python<'_>,
    kwargs: &Bound<'_, PyDict>,
    call_type: &str,
) -> PyResult<()> {
    let Some(configured) = selected_cache(py, kwargs, call_type)? else {
        return Ok(());
    };
    if configured
        .call_method("should_use_cache", (), Some(kwargs))?
        .extract::<bool>()?
    {
        return Err(HandleError::Unsupported.into());
    }
    Ok(())
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
        policy: CachePolicy {
            no_cache: boolean("no-cache")?,
            no_store: boolean("no-store")?,
            ..CachePolicy::default()
        },
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
