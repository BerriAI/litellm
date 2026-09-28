use std::{future::Future, pin::Pin, sync::Arc, time::Duration};

use litellm_cache::Error;
use litellm_cache_response::{
    CacheOptions, CacheScope, ResponseCacheConfig, ResponseCacheRequest, ResponseCacheService,
    ScopedCache, cache_key,
};
use litellm_host::{
    machine::{HostServices, MachineFault},
    protocol::{Protocol, Reply},
};
use litellm_host_python::{from_py, to_py};
use pyo3::{
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::PyDict,
};
use serde_json::Value;

pub(crate) enum CacheCall {
    Lookup {
        key: String,
        reply: Reply<Result<Option<Value>, Error>>,
    },
    Store {
        key: String,
        value: Value,
        reply: Reply<Result<(), Error>>,
    },
}

pub(crate) struct Cached<P>(std::marker::PhantomData<P>);

impl<P: Protocol> Protocol for Cached<P> {
    type Request = (P::Request, Selection);
    type Response = P::Response;
    type Error = P::Error;
    type HostCall = CacheCall;
    type Chunk = P::Chunk;
    type StreamHead = P::StreamHead;
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
    pub fn attach<P: Protocol<HostCall = CacheCall>>(
        self,
        services: HostServices<P>,
    ) -> (Option<ScopedCache>, CacheOptions)
    where
        P::Error: From<MachineFault>,
    {
        let service = match self.backend {
            Backend::Disabled => None,
            Backend::Native(service) => Some(service),
            Backend::Python { namespace } => Some(Arc::new(HostedCache {
                services,
                config: ResponseCacheConfig {
                    namespace,
                    ..Default::default()
                },
            }) as Arc<dyn ResponseCacheService>),
        };
        (
            service.map(|service| ScopedCache::new(service, CacheScope::Shared)),
            self.options,
        )
    }
}

struct HostedCache<P: Protocol> {
    services: HostServices<P>,
    config: ResponseCacheConfig,
}

impl<P: Protocol<HostCall = CacheCall>> ResponseCacheService for HostedCache<P>
where
    P::Error: From<MachineFault>,
{
    fn config(&self) -> &ResponseCacheConfig {
        &self.config
    }

    fn lookup<'a>(
        &'a self,
        request: &'a ResponseCacheRequest,
        _: Duration,
    ) -> Pin<Box<dyn Future<Output = Result<Option<Value>, Error>> + Send + 'a>> {
        Box::pin(async move {
            self.services
                .call(|reply| CacheCall::Lookup {
                    key: cache_key(&request.key),
                    reply,
                })
                .await
                .map_err(|_| Error::Unavailable)?
        })
    }

    fn store<'a>(
        &'a self,
        request: &'a ResponseCacheRequest,
        value: Value,
        _: Duration,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + 'a>> {
        Box::pin(async move {
            self.services
                .call(|reply| CacheCall::Store {
                    key: cache_key(&request.key),
                    value,
                    reply,
                })
                .await
                .map_err(|_| Error::Unavailable)?
        })
    }
}

enum Pending {
    Lookup(Reply<Result<Option<Value>, Error>>),
    Store(Reply<Result<(), Error>>),
}

pub(crate) struct PythonCache {
    cache: Option<Py<PyAny>>,
    arguments: Option<Py<PyDict>>,
    pending: Option<Pending>,
    asynchronous: bool,
}

impl PythonCache {
    pub fn new(asynchronous: bool) -> Self {
        Self {
            cache: None,
            arguments: None,
            pending: None,
            asynchronous,
        }
    }

    pub fn configure(
        &mut self,
        py: Python<'_>,
        arguments: &Bound<'_, PyDict>,
        call_type: &str,
    ) -> PyResult<Selection> {
        let selected = super::v2::selected_cache(py, arguments, call_type)?;
        let Some(cache) = selected else {
            return Ok(Selection {
                backend: Backend::Disabled,
                options: CacheOptions::new(CacheScope::Shared),
            });
        };
        if super::v2::native_handle(&cache)?.is_some() {
            let (native, options) = super::v2::configured(py, arguments, call_type)?;
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
        self.cache = Some(cache.unbind());
        self.arguments = Some(arguments.clone().unbind());
        Ok(Selection {
            backend: if enabled {
                Backend::Python { namespace }
            } else {
                Backend::Disabled
            },
            options,
        })
    }

    pub fn begin(&mut self, py: Python<'_>, call: CacheCall) -> PyResult<Option<Py<PyAny>>> {
        let Some(cache) = self.cache.as_ref() else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "cache operation without configured cache",
            ));
        };
        let arguments = self
            .arguments
            .as_ref()
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("cache arguments unavailable")
            })?
            .bind(py)
            .copy()?;
        let (method, result) = match call {
            CacheCall::Lookup { key, reply } => {
                self.pending = Some(Pending::Lookup(reply));
                arguments.set_item("cache_key", key)?;
                (
                    if self.asynchronous {
                        "async_get_cache"
                    } else {
                        "get_cache"
                    },
                    None,
                )
            }
            CacheCall::Store { key, value, reply } => {
                self.pending = Some(Pending::Store(reply));
                arguments.set_item("cache_key", key)?;
                (
                    if self.asynchronous {
                        "async_add_cache"
                    } else {
                        "add_cache"
                    },
                    Some(to_py(py, &value)?),
                )
            }
        };
        let result = match result {
            Some(value) => cache
                .bind(py)
                .call_method(method, (value,), Some(&arguments)),
            None => cache.bind(py).call_method(method, (), Some(&arguments)),
        }
        .map(Bound::unbind);
        if self.asynchronous && result.is_ok() {
            return result.map(Some);
        }
        self.resume(py, result)
    }

    pub fn resume(
        &mut self,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> PyResult<Option<Py<PyAny>>> {
        if let Err(error) = &result
            && !error.is_instance_of::<pyo3::exceptions::PyException>(py)
        {
            self.pending = None;
            return Err(result.err().unwrap());
        }
        match self.pending.take() {
            Some(Pending::Lookup(reply)) => {
                let value = result.map_err(|_| Error::Unavailable).and_then(|value| {
                    if value.bind(py).is_none() {
                        Ok(None)
                    } else {
                        from_py(value.bind(py))
                            .map(Some)
                            .map_err(|_| Error::InvalidEntry)
                    }
                });
                reply.send(value);
            }
            Some(Pending::Store(reply)) => {
                reply.send(result.map(|_| ()).map_err(|_| Error::Unavailable));
            }
            None => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "cache reply without pending operation",
                ));
            }
        }
        Ok(None)
    }

    pub fn close(&mut self) {
        self.pending = None;
        self.cache = None;
        self.arguments = None;
    }

    pub fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.cache)?;
        visit.call(&self.arguments)
    }
}
