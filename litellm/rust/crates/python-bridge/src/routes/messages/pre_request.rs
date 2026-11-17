use futures_util::future::BoxFuture;
use litellm_host::machine::HostServices;
use litellm_inference_messages::{Error, MessagesCall, PreRequestHooks};
use pyo3::prelude::*;

use super::host::{MessagesHostCall, MessagesHosted, ROUTE_HOST_MODULE};

pub(super) fn runs_pre_request_hooks(py: Python<'_>) -> PyResult<bool> {
    py.import(ROUTE_HOST_MODULE)?
        .getattr("runs_pre_request_hooks")?
        .call0()?
        .extract()
}

pub(super) struct HostPreRequestHooks {
    services: HostServices<MessagesHosted>,
}

impl HostPreRequestHooks {
    pub(super) fn new(services: HostServices<MessagesHosted>) -> Self {
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
