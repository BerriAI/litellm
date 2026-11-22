use bytes::Bytes;
use futures_util::future::BoxFuture;
use litellm_host::{machine::HostServices, protocol::Reply};
use litellm_host_python::{from_py, to_py};
use litellm_inference_messages::{Error, MessagesBody, Safeguards};
use pyo3::{
    exceptions::PyRuntimeError,
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::{PyBytes, PyDict},
};
use serde_json::Value;

use super::host::{MessagesHostCall, MessagesHosted, ROUTE_HOST_MODULE};

pub(super) enum SafeguardsCall {
    Results {
        response: MessagesBody,
        reply: Reply<Option<Value>>,
    },
    OpenStream {
        reply: Reply<bool>,
    },
    Frames {
        chunk: Bytes,
        reply: Reply<Vec<Bytes>>,
    },
    Flush {
        reply: Reply<Vec<Bytes>>,
    },
}

pub(super) struct PythonSafeguards {
    services: HostServices<MessagesHosted>,
}

impl PythonSafeguards {
    pub(super) fn new(services: HostServices<MessagesHosted>) -> Self {
        Self { services }
    }

    async fn ask<A: Send>(
        &self,
        call: impl FnOnce(Reply<A>) -> SafeguardsCall + Send,
    ) -> Result<A, Error> {
        self.services
            .call(|reply| MessagesHostCall::Safeguards(call(reply)))
            .await
    }
}

impl Safeguards for PythonSafeguards {
    fn results<'a>(
        &'a self,
        response: &'a MessagesBody,
    ) -> BoxFuture<'a, Result<Option<Value>, Error>> {
        let response = response.clone();
        Box::pin(self.ask(|reply| SafeguardsCall::Results { response, reply }))
    }

    fn open_stream(&self) -> BoxFuture<'_, Result<bool, Error>> {
        Box::pin(self.ask(|reply| SafeguardsCall::OpenStream { reply }))
    }

    fn frames(&self, chunk: Bytes) -> BoxFuture<'_, Result<Vec<Bytes>, Error>> {
        Box::pin(self.ask(|reply| SafeguardsCall::Frames { chunk, reply }))
    }

    fn flush(&self) -> BoxFuture<'_, Result<Vec<Bytes>, Error>> {
        Box::pin(self.ask(|reply| SafeguardsCall::Flush { reply }))
    }
}

enum Pending {
    Results(Reply<Option<Value>>),
    OpenStream(Reply<bool>),
    Frames(Reply<Vec<Bytes>>),
}

#[derive(Default)]
pub(super) struct SafeguardsHost {
    classifier: Option<Py<PyAny>>,
    pending: Option<Pending>,
}

impl SafeguardsHost {
    pub(super) fn configure(
        &mut self,
        py: Python<'_>,
        request: &Bound<'_, PyDict>,
    ) -> PyResult<bool> {
        let classifier = py
            .import(ROUTE_HOST_MODULE)?
            .getattr("safeguards")?
            .call1((request,))?;
        self.classifier = (!classifier.is_none()).then(|| classifier.unbind());
        Ok(self.classifier.is_some())
    }

    pub(super) fn rebind(&mut self, py: Python<'_>, request: &Bound<'_, PyDict>) -> PyResult<()> {
        if self.classifier.is_none() {
            return Ok(());
        }
        let classifier = py
            .import(ROUTE_HOST_MODULE)?
            .getattr("safeguards")?
            .call1((request,))?;
        if !classifier.is_none() {
            self.classifier = Some(classifier.unbind());
        }
        Ok(())
    }

    pub(super) fn begin(&mut self, py: Python<'_>, call: SafeguardsCall) -> PyResult<Py<PyAny>> {
        let classifier = self
            .classifier
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("safeguards call without a classifier"))?
            .bind(py);
        let (awaitable, pending) = match call {
            SafeguardsCall::Results { response, reply } => (
                classifier.call_method1("results", (to_py(py, &response)?,))?,
                Pending::Results(reply),
            ),
            SafeguardsCall::OpenStream { reply } => (
                classifier.call_method0("open_stream")?,
                Pending::OpenStream(reply),
            ),
            SafeguardsCall::Frames { chunk, reply } => (
                classifier.call_method1("frames", (PyBytes::new(py, &chunk),))?,
                Pending::Frames(reply),
            ),
            SafeguardsCall::Flush { reply } => {
                (classifier.call_method0("flush")?, Pending::Frames(reply))
            }
        };
        self.pending = Some(pending);
        Ok(awaitable.unbind())
    }

    pub(super) fn resume(&mut self, py: Python<'_>, result: PyResult<Py<PyAny>>) -> PyResult<()> {
        let pending = self
            .pending
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("safeguards reply without a pending call"))?;
        let value = result?.into_bound(py);
        match pending {
            Pending::Results(reply) => reply.send(if value.is_none() {
                None
            } else {
                Some(from_py(&value)?)
            }),
            Pending::OpenStream(reply) => reply.send(value.extract()?),
            Pending::Frames(reply) => reply.send(
                value
                    .extract::<Vec<Bound<'_, PyBytes>>>()?
                    .iter()
                    .map(|frame| Bytes::copy_from_slice(frame.as_bytes()))
                    .collect(),
            ),
        }
        Ok(())
    }

    pub(super) fn published(&self, py: Python<'_>, public: Py<PyAny>) -> PyResult<Py<PyAny>> {
        match &self.classifier {
            Some(classifier) => classifier
                .bind(py)
                .call_method1("published", (public,))
                .map(Bound::unbind),
            None => Ok(public),
        }
    }

    pub(super) fn close(&mut self) {
        self.classifier = None;
        self.pending = None;
    }

    pub(super) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.classifier)
    }
}
