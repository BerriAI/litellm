//! The Anthropic Messages call, the Rust equivalent of Python's `litellm.messages()`.
//!
//! [`CoreClient::messages`](crate::CoreClient::messages) prepares the provider request and sends it in process. [`route`] runs the
//! same two steps as a machine for a host that answers the call's operations itself.

mod common_utils;
mod handler;
mod prepare;
pub mod route;
mod types;

use litellm_secrets::source::SecretSource;

pub use crate::error::RouteError as Error;
pub use types::{MessagesCall, MessagesResponse, MessagesShaping, messages_body};

impl crate::CoreClient {
    pub async fn messages(&self, call: MessagesCall) -> Result<MessagesResponse, Error> {
        self.messages_with_hooks(call, &()).await
    }

    pub async fn messages_with_hooks(
        &self,
        call: MessagesCall,
        hooks: &impl litellm_host::hooks::RouteHooks<Error>,
    ) -> Result<MessagesResponse, Error> {
        litellm_host::lifecycle::observe_call(hooks.observer(), async {
            let http = self.provider_http()?;
            execute(
                &http,
                &self.resources().auth,
                self.secret_source().as_ref(),
                call,
                hooks,
            )
            .await
        })
        .await
    }
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
