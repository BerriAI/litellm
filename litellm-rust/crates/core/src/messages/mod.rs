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
    pub fn messages(&self, call: MessagesCall) -> crate::CallBuilder<'_, MessagesCall> {
        crate::CallBuilder::new(self, call)
    }
}

impl<'a, H> std::future::IntoFuture for crate::CallBuilder<'a, MessagesCall, H>
where
    H: litellm_host::hooks::RouteHooks<Error>,
{
    type Output = Result<MessagesResponse, Error>;
    type IntoFuture = futures_util::future::BoxFuture<'a, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            litellm_host::lifecycle::observe_call(self.hooks.observer(), async {
                let http = self.client.provider_http();
                execute(
                    http,
                    &self.client.resources().auth,
                    self.client.secret_source().as_ref(),
                    self.request,
                    self.hooks,
                )
                .await
            })
            .await
        })
    }
}

#[tracing::instrument(name = "litellm.route", skip_all, fields(
    route = "messages",
    model = %call.body.model,
    provider,
    resolved_model,
    stream = call.body.params.stream == Some(true),
    outcome
))]
async fn execute(
    http: Result<litellm_http::Client, litellm_http::Error>,
    auth: &litellm_auth::AuthServices,
    secrets: &dyn SecretSource,
    call: MessagesCall,
    hooks: &impl litellm_host::hooks::RouteHooks<Error>,
) -> Result<MessagesResponse, Error> {
    crate::diagnostic::call(async {
        let http = http?;
        let request = prepare::prepare(call, secrets).await?;
        crate::diagnostic::provider(&request.body.model, request.provider.as_str());
        let execute: futures_util::future::BoxFuture<'_, Result<MessagesResponse, Error>> =
            Box::pin(handler::execute(&http, auth, request, hooks));
        execute.await
    })
    .await
}
