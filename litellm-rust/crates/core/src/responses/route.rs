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
    type Projection = ResponsesCall;
    type Op = Infallible;
    type Chunk = Bytes;
    type StreamHead = ResponsesStreamHead;
}

pub fn responses_machine(
    resources: &crate::resources::CoreResources,
    config: &HttpClientConfig,
    secrets: Arc<dyn SecretSource>,
) -> Result<HostedMachine<Responses>, litellm_http::Error> {
    let http = resources.pool.client(config, ClientVariant::Provider)?;
    let auth = resources.auth.clone();
    Ok(hosted_call(move |call, host| async move {
        super::execute(&http, &auth, secrets.as_ref(), call, &host).await
    }))
}
