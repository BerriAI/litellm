use futures_util::future::BoxFuture;
use litellm_host::{machine::HostServices, protocol::Reply};
use litellm_host_python::{ExecutionBody, ExecutionStep};
use litellm_inference_messages::{Error, McpGateway, MessagesBody, route::Messages};
use pyo3::{
    gc::{PyTraverseError, PyVisit},
    prelude::*,
};
use serde_json::Map;

use crate::cache::{CacheCall, Cached};

pub(super) enum MessagesHostCall {
    Cache(CacheCall),
    McpGateway(Reply<()>),
}

impl From<CacheCall> for MessagesHostCall {
    fn from(call: CacheCall) -> Self {
        Self::Cache(call)
    }
}

pub(super) type HostedMessages = Cached<Messages, MessagesHostCall>;

pub(super) struct PythonMcpGateway {
    services: HostServices<HostedMessages>,
}

impl PythonMcpGateway {
    pub(super) fn new(services: HostServices<HostedMessages>) -> Self {
        Self { services }
    }
}

impl McpGateway for PythonMcpGateway {
    fn serve(&self) -> BoxFuture<'_, Result<MessagesBody, Error>> {
        Box::pin(async move {
            self.services.call(MessagesHostCall::McpGateway).await?;
            Ok(MessagesBody::Unchecked(Map::new()))
        })
    }
}

pub(super) struct Ready(Option<Py<PyAny>>);

impl Ready {
    pub(super) fn new(value: Py<PyAny>) -> Self {
        Self(Some(value))
    }
}

impl ExecutionBody for Ready {
    fn resume(&mut self, _: Option<PyResult<Py<PyAny>>>) -> PyResult<ExecutionStep> {
        self.0
            .take()
            .map(ExecutionStep::Return)
            .ok_or_else(|| litellm_host_python::Error::MissingState.into())
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.0)
    }
}
