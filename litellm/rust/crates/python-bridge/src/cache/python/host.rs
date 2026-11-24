use litellm_cache::Error;
use litellm_cache_response::ResponseEnvelope;
use litellm_host::protocol::Reply;
use litellm_host_python::{json_from_py, to_py};
use litellm_inference::caching::CachedOutput;
use litellm_inference_messages::CacheCall;
use pyo3::{
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::PyDict,
};
use serde_json::Value;

use crate::cache::{CACHE_HOST_MODULE, HandleError};

const STREAM_EVENTS_KEY: &str = "litellm_cached_anthropic_sse_events";

fn from_python(value: Value) -> Result<Value, Error> {
    let output = match value.get(STREAM_EVENTS_KEY) {
        Some(events) => {
            let events: Vec<String> =
                serde_json::from_value(events.clone()).map_err(Error::invalid_entry)?;
            CachedOutput::Stream(events.concat())
        }
        None => CachedOutput::Response(value),
    };
    serde_json::to_value(ResponseEnvelope::new("messages", output)).map_err(Error::invalid_entry)
}

fn to_python(value: Value) -> Result<Value, Error> {
    let envelope: ResponseEnvelope<CachedOutput<Value>> =
        serde_json::from_value(value).map_err(Error::invalid_entry)?;
    match envelope
        .decode("messages")
        .ok_or(Error::InvalidEntry(None))?
    {
        CachedOutput::Response(response) => Ok(response),
        CachedOutput::Stream(text) => Ok(serde_json::json!({
            STREAM_EVENTS_KEY: text.split_inclusive("\n\n").collect::<Vec<_>>()
        })),
    }
}

enum Pending {
    Lookup(Reply<Result<Option<Value>, Error>>),
    Store(Reply<Result<(), Error>>),
}

pub(crate) struct PythonCache {
    cache: Option<Py<PyAny>>,
    function: Option<Py<PyAny>>,
    arguments: Option<Py<PyDict>>,
    read_handler: Option<Py<PyAny>>,
    pending: Option<Pending>,
    asynchronous: bool,
    deferred: bool,
}

impl PythonCache {
    pub fn new(asynchronous: bool) -> Self {
        Self {
            cache: None,
            function: None,
            arguments: None,
            read_handler: None,
            pending: None,
            asynchronous,
            deferred: false,
        }
    }

    pub(in crate::cache) fn asynchronous(&self) -> bool {
        self.asynchronous
    }

    pub(in crate::cache) fn bind(
        &mut self,
        cache: Bound<'_, PyAny>,
        function: &Bound<'_, PyAny>,
        arguments: &Bound<'_, PyDict>,
    ) {
        self.cache = Some(cache.unbind());
        self.function = Some(function.clone().unbind());
        self.arguments = Some(arguments.clone().unbind());
    }

    pub fn begin(&mut self, py: Python<'_>, call: CacheCall) -> PyResult<Option<Py<PyAny>>> {
        let Some(cache) = self.cache.as_ref() else {
            return Err(HandleError::Unconfigured.into());
        };
        let arguments = self
            .arguments
            .as_ref()
            .ok_or(HandleError::MissingArguments)?
            .bind(py)
            .copy()?;
        let (method, result) = match call {
            CacheCall::Lookup { reply } => {
                self.pending = Some(Pending::Lookup(reply));
                (
                    if self.asynchronous {
                        "lookup"
                    } else {
                        "get_cache"
                    },
                    None,
                )
            }
            CacheCall::Store { value, reply } => {
                let value = match to_python(value) {
                    Ok(value) => value,
                    Err(error) => {
                        reply.send(Err(error));
                        return Ok(None);
                    }
                };
                if self.asynchronous && value.get(STREAM_EVENTS_KEY).is_none() {
                    self.deferred = true;
                    reply.send(Ok(()));
                    return Ok(None);
                }
                self.pending = Some(Pending::Store(reply));
                (
                    if self.asynchronous {
                        "store_stream"
                    } else {
                        "add_cache"
                    },
                    Some(to_py(py, &value)?),
                )
            }
        };
        let result = match result {
            Some(value) if self.asynchronous => py
                .import(CACHE_HOST_MODULE)?
                .getattr(method)?
                .call((cache, value), Some(&arguments)),
            Some(value) => cache
                .bind(py)
                .call_method(method, (value,), Some(&arguments)),
            None if self.asynchronous => py.import(CACHE_HOST_MODULE)?.getattr(method)?.call(
                (
                    cache,
                    self.function.as_ref().ok_or(HandleError::Unconfigured)?,
                ),
                Some(&arguments),
            ),
            None => cache.bind(py).call_method(method, (), Some(&arguments)),
        }
        .map(Bound::unbind);
        if self.asynchronous && result.is_ok() {
            return result.map(Some);
        }
        self.resume(py, result)
    }

    pub fn store_response(
        &mut self,
        py: Python<'_>,
        response: &Bound<'_, PyAny>,
        verbatim: bool,
    ) -> PyResult<()> {
        if !std::mem::take(&mut self.deferred) {
            return Ok(());
        }
        let (Some(cache), Some(arguments)) = (self.cache.as_ref(), self.arguments.as_ref()) else {
            return Err(HandleError::Unconfigured.into());
        };
        py.import(CACHE_HOST_MODULE)?
            .getattr(if verbatim {
                "store_verbatim"
            } else {
                "store_response"
            })?
            .call((cache, response), Some(&arguments.bind(py).copy()?))?;
        Ok(())
    }

    pub fn wrap_stream(&self, py: Python<'_>, stream: Py<PyAny>) -> PyResult<Py<PyAny>> {
        let (Some(cache), Some(function), Some(arguments)) =
            (&self.cache, &self.function, &self.arguments)
        else {
            return Ok(stream);
        };
        py.import(CACHE_HOST_MODULE)?
            .getattr("wrap_stream")?
            .call(
                (cache, function, stream, &self.read_handler),
                Some(&arguments.bind(py).copy()?),
            )
            .map(Bound::unbind)
    }

    pub fn resume(
        &mut self,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> PyResult<Option<Py<PyAny>>> {
        let result = match result {
            Err(error) if !error.is_instance_of::<pyo3::exceptions::PyException>(py) => {
                self.pending = None;
                return Err(error);
            }
            result => result,
        };
        match self.pending.take() {
            Some(Pending::Lookup(reply)) => {
                if self.asynchronous {
                    self.read_handler = self.caching_handler(py).ok().flatten();
                }
                let value = result.map_err(Error::unavailable).and_then(|value| {
                    if value.bind(py).is_none() {
                        Ok(None)
                    } else {
                        json_from_py(value.bind(py))
                            .map_err(Error::invalid_entry)
                            .and_then(from_python)
                            .map(Some)
                    }
                });
                reply.send(value);
            }
            Some(Pending::Store(reply)) => {
                reply.send(result.map(|_| ()).map_err(Error::unavailable));
            }
            None => return Err(HandleError::NothingPending.into()),
        }
        Ok(None)
    }

    fn caching_handler(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let Some(arguments) = self.arguments.as_ref() else {
            return Ok(None);
        };
        let handler = py
            .import(CACHE_HOST_MODULE)?
            .getattr("caching_handler")?
            .call((), Some(arguments.bind(py)))?;
        Ok((!handler.is_none()).then(|| handler.unbind()))
    }

    pub fn close(&mut self) {
        self.pending = None;
        self.read_handler = None;
        self.deferred = false;
        self.cache = None;
        self.function = None;
        self.arguments = None;
    }

    pub fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.cache)?;
        visit.call(&self.function)?;
        visit.call(&self.arguments)?;
        visit.call(&self.read_handler)
    }
}
