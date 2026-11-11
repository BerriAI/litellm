use futures_util::future::BoxFuture;
use litellm_host::{machine::HostServices, protocol::Reply};
use litellm_inference_messages::{Error, MessagesCall, PreRequestHooks, route::Messages};
use pyo3::prelude::*;

use super::host::ROUTE_HOST_MODULE;
use crate::cache::{CacheCall, Cached};

pub(super) type HostedMessages = Cached<Messages, MessagesHostCall>;

pub(super) enum MessagesHostCall {
    Cache(CacheCall),
    PreRequest(Reply<Result<MessagesCall, Error>>),
}

impl From<CacheCall> for MessagesHostCall {
    fn from(call: CacheCall) -> Self {
        Self::Cache(call)
    }
}

pub(super) fn runs_pre_request_hooks(py: Python<'_>) -> PyResult<bool> {
    py.import(ROUTE_HOST_MODULE)?
        .getattr("runs_pre_request_hooks")?
        .call0()?
        .extract()
}

pub(super) struct HostPreRequestHooks {
    services: HostServices<HostedMessages>,
}

impl HostPreRequestHooks {
    pub(super) fn new(services: HostServices<HostedMessages>) -> Self {
        Self { services }
    }
}

impl PreRequestHooks for HostPreRequestHooks {
    fn run(&self, call: MessagesCall) -> BoxFuture<'_, Result<MessagesCall, Error>> {
        drop(call);
        Box::pin(async move {
            self.services
                .call(MessagesHostCall::PreRequest)
                .await
                .and_then(|hooked| hooked)
        })
    }
}
