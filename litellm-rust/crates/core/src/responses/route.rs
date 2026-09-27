use std::{convert::Infallible, sync::Arc};

use bytes::Bytes;
use litellm_host::{
    call::{HostedMachine, hosted_call},
    protocol::Protocol,
};
use litellm_http::{ClientVariant, HttpClientConfig};
use litellm_secrets::source::SecretSource;
use litellm_types::responses::main::ResponsesApiResponse;

use super::{
    Error,
    types::{ResponsesCall, ResponsesStreamHead},
};

pub struct Responses;

impl Protocol for Responses {
    type Response = ResponsesApiResponse;
    type Error = Error;
    type Request = ResponsesCall;
    type HostCall = Infallible;
    type Chunk = Bytes;
    type StreamHead = ResponsesStreamHead;
}

pub fn responses_machine(
    resources: &crate::resources::CoreResources,
    config: &HttpClientConfig,
    secrets: Arc<dyn SecretSource>,
) -> Result<
    impl FnOnce(ResponsesCall) -> HostedMachine<Responses> + Send + Sync + use<>,
    litellm_http::Error,
> {
    let http = resources.pool.client(config, ClientVariant::Provider)?;
    let auth = resources.auth.clone();
    Ok(move |request| {
        hosted_call(request, move |call, _, hooks| async move {
            super::execute(&http, &auth, secrets.as_ref(), call, &hooks).await
        })
    })
}
