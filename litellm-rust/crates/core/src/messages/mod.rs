//! The Anthropic Messages call, the Rust equivalent of Python's `litellm.messages()`.
//!
//! [`messages`] prepares the provider request and sends it in process. [`route`] runs the
//! same two steps as a machine for a host that answers the call's operations itself.

mod common_utils;
mod handler;
mod prepare;
pub mod route;
mod types;

use litellm_http::{ClientVariant, HttpClientConfig};
use litellm_secrets::source::SecretSource;

pub use crate::error::RouteError as Error;
pub use types::{MessagesCall, MessagesResponse, MessagesShaping, messages_body};

pub async fn messages(
    resources: &crate::resources::CoreResources,
    config: &HttpClientConfig,
    secrets: &dyn SecretSource,
    call: MessagesCall,
) -> Result<MessagesResponse, Error> {
    messages_with_hooks(resources, config, secrets, call, &()).await
}

pub async fn messages_with_hooks(
    resources: &crate::resources::CoreResources,
    config: &HttpClientConfig,
    secrets: &dyn SecretSource,
    call: MessagesCall,
    hooks: &impl litellm_host::hooks::RouteHooks<Error>,
) -> Result<MessagesResponse, Error> {
    litellm_host::call::observe_call(hooks.observer(), async {
        let http = resources.pool.client(config, ClientVariant::Provider)?;
        execute(&http, &resources.auth, secrets, call, hooks).await
    })
    .await
}

async fn execute(
    http: &litellm_http::Client,
    auth: &litellm_auth::AuthServices,
    secrets: &dyn SecretSource,
    call: MessagesCall,
    hooks: &impl litellm_host::hooks::RouteHooks<Error>,
) -> Result<MessagesResponse, Error> {
    let request = prepare::prepare(call, secrets).await?;
    handler::execute(http, auth, request, hooks).await
}
