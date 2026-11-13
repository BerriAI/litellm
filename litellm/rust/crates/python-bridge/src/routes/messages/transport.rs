use std::fmt;

use bytes::Bytes;
use futures_util::{StreamExt, future::BoxFuture};
use litellm_host::{
    machine::HostServices,
    protocol::{Protocol, Reply},
};
use litellm_host_python::to_py;
use litellm_inference_messages::{
    Delivered, Error, Transport, TransportRequest, TransportResponse,
};
use pyo3::{
    exceptions::{PyBaseException, PyException, PyStopAsyncIteration},
    gc::{PyTraverseError, PyVisit},
    prelude::*,
    types::{PyBytes, PyDict},
};

use super::host::ROUTE_HOST_MODULE;

pub(crate) enum TransportCall {
    Send {
        request: TransportRequest,
        reply: Reply<Result<Sent, Error>>,
    },
    Read {
        reply: Reply<Result<Option<Bytes>, Error>>,
    },
}

pub(crate) struct Sent {
    status: u16,
    headers: Vec<(String, String)>,
    body: SentBody,
}

enum SentBody {
    Text(String),
    Stream,
    Rejected { text: String, error: Error },
}

pub(super) struct PythonTransport<P: Protocol> {
    services: HostServices<P>,
}

impl<P: Protocol> PythonTransport<P> {
    pub(super) fn new(services: HostServices<P>) -> Self {
        Self { services }
    }
}

impl<P> Transport for PythonTransport<P>
where
    P: Protocol<Error = Error>,
    P::HostCall: From<TransportCall>,
{
    fn send(
        &self,
        request: TransportRequest,
    ) -> BoxFuture<'static, Result<TransportResponse, Error>> {
        let services = self.services.clone();
        Box::pin(async move {
            let sent = services
                .call(|reply| TransportCall::Send { request, reply }.into())
                .await??;
            let delivered = match sent.body {
                SentBody::Text(text) => Delivered::Complete(text),
                SentBody::Rejected { text, error } => Delivered::Rejected { text, error },
                SentBody::Stream => Delivered::Stream(
                    futures_util::stream::try_unfold(services, |services| async move {
                        let chunk = services
                            .call(|reply| TransportCall::Read { reply }.into())
                            .await??;
                        Ok(chunk.map(|chunk| (chunk, services)))
                    })
                    .boxed(),
                ),
            };
            Ok(TransportResponse {
                status: sent.status,
                headers: sent.headers,
                delivered,
            })
        })
    }
}

#[derive(thiserror::Error)]
#[error("the Python HTTP client failed")]
struct HostTransportFailure {
    error: Py<PyBaseException>,
    raw: bool,
}

impl fmt::Debug for HostTransportFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("HostTransportFailure")
    }
}

fn failed(py: Python<'_>, error: PyErr, raw: bool) -> Error {
    Error::transport(HostTransportFailure {
        error: error.into_value(py),
        raw,
    })
}

pub(super) fn python_failure(
    py: Python<'_>,
    error: &(dyn std::error::Error + 'static),
) -> Option<(PyErr, bool)> {
    litellm_error::find::<HostTransportFailure>(error).map(|failure| {
        (
            PyErr::from_value(failure.error.clone_ref(py).into_bound(py).into_any()),
            failure.raw,
        )
    })
}

enum Pending {
    Send(Reply<Result<Sent, Error>>),
    Read(Reply<Result<Option<Bytes>, Error>>),
}

pub(super) struct Selected {
    pub(super) hosted: bool,
    pub(super) read: Vec<(String, Option<String>)>,
}

#[derive(Default)]
pub(super) struct HostedTransport {
    handler: Option<Py<PyAny>>,
    read: Vec<(String, Option<String>)>,
    chunks: Option<Py<PyAny>>,
    pending: Option<Pending>,
}

impl HostedTransport {
    /// Whether the call sends through Python's client, and the secrets the host read to decide it,
    /// which the native route reuses rather than reading again.
    pub(super) fn select(
        &mut self,
        py: Python<'_>,
        request: &Bound<'_, PyDict>,
    ) -> PyResult<Selected> {
        let (handler, read): (Bound<'_, PyAny>, Vec<(String, Option<String>)>) = py
            .import(ROUTE_HOST_MODULE)?
            .getattr("transport_handler")?
            .call1((request,))?
            .extract()?;
        self.handler = (!handler.is_none()).then(|| handler.unbind());
        self.read.clone_from(&read);
        Ok(Selected {
            hosted: self.handler.is_some(),
            read,
        })
    }

    /// The secrets the host read to choose the transport, such as the `AZURE_API_BASE` the call
    /// was sent to, which Python's failure mapping reads instead of looking them up again.
    pub(super) fn read(&self) -> &[(String, Option<String>)] {
        &self.read
    }

    pub(super) fn pending(&self) -> bool {
        self.pending.is_some()
    }

    pub(super) fn begin(
        &mut self,
        py: Python<'_>,
        call: TransportCall,
        request: &Bound<'_, PyDict>,
    ) -> PyResult<Option<Py<PyAny>>> {
        match call {
            TransportCall::Send {
                request: sent,
                reply,
            } => {
                let handler = self
                    .handler
                    .as_ref()
                    .ok_or(litellm_host_python::Error::MissingState)?;
                let awaitable = py
                    .import(ROUTE_HOST_MODULE)?
                    .getattr("transport_send")?
                    .call1((
                        handler,
                        request,
                        &sent.url,
                        to_py(py, &sent.headers)?,
                        PyBytes::new(py, &sent.body),
                        sent.stream,
                    ))?;
                self.pending = Some(Pending::Send(reply));
                Ok(Some(awaitable.unbind()))
            }
            TransportCall::Read { reply } => {
                let Some(chunks) = self.chunks.as_ref() else {
                    reply.send(Ok(None));
                    return Ok(None);
                };
                let awaitable = chunks.bind(py).call_method0("__anext__")?;
                self.pending = Some(Pending::Read(reply));
                Ok(Some(awaitable.unbind()))
            }
        }
    }

    pub(super) fn resume(
        &mut self,
        py: Python<'_>,
        result: PyResult<Py<PyAny>>,
    ) -> PyResult<Option<Py<PyAny>>> {
        let result = match result {
            Err(error) if !error.is_instance_of::<PyException>(py) => {
                self.pending = None;
                return Err(error);
            }
            result => result,
        };
        match self.pending.take() {
            Some(Pending::Send(reply)) => {
                reply.send(match result {
                    Ok(sent) => self.sent(py, sent.bind(py))?,
                    Err(error) => Err(self.send_failure(py, error)?),
                });
            }
            Some(Pending::Read(reply)) => reply.send(match result {
                Ok(chunk) => Ok(Some(Bytes::copy_from_slice(
                    chunk.bind(py).cast::<PyBytes>()?.as_bytes(),
                ))),
                Err(error) if error.is_instance_of::<PyStopAsyncIteration>(py) => {
                    self.chunks = None;
                    Ok(None)
                }
                Err(error) => {
                    self.chunks = None;
                    Err(failed(py, error, true))
                }
            }),
            None => return Err(litellm_host_python::Error::MissingState.into()),
        }
        Ok(None)
    }

    fn send_failure(&self, py: Python<'_>, error: PyErr) -> PyResult<Error> {
        let raw = py
            .import(ROUTE_HOST_MODULE)?
            .getattr("RawTransportFailure")?;
        if error.value(py).is_instance(&raw)? {
            let raised = error.value(py).getattr("error")?;
            return Ok(failed(py, PyErr::from_value(raised), true));
        }
        Ok(failed(py, error, false))
    }

    fn sent(&mut self, py: Python<'_>, sent: &Bound<'_, PyAny>) -> PyResult<Result<Sent, Error>> {
        let status: u16 = sent.getattr("status")?.extract()?;
        let headers: Vec<(String, String)> = sent.getattr("headers")?.extract()?;
        let text: Option<String> = sent.getattr("text")?.extract()?;
        let rejection = sent.getattr("rejection")?;
        let chunks = sent.getattr("chunks")?;
        let body = if !rejection.is_none() {
            SentBody::Rejected {
                text: text.unwrap_or_default(),
                error: failed(py, PyErr::from_value(rejection), false),
            }
        } else if !chunks.is_none() {
            self.chunks = Some(chunks.unbind());
            SentBody::Stream
        } else {
            SentBody::Text(text.unwrap_or_default())
        };
        Ok(Ok(Sent {
            status,
            headers,
            body,
        }))
    }

    pub(super) fn close(&mut self) {
        self.pending = None;
        self.chunks = None;
        self.handler = None;
    }

    pub(super) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.handler)?;
        visit.call(&self.chunks)
    }
}
