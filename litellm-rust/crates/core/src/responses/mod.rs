pub use crate::error::RouteError as Error;
pub mod websocket;

mod handler;
mod prepare;
pub mod route;
pub mod types;

use litellm_host::hooks::RouteHooks;
use litellm_secrets::source::SecretSource;
use types::{ResponsesCall, ResponsesOutput};

impl crate::CoreClient {
    pub fn responses(&self, call: ResponsesCall) -> crate::CallBuilder<'_, ResponsesCall> {
        crate::CallBuilder::new(self, call)
    }
}

impl<'a, H: RouteHooks<Error>> std::future::IntoFuture
    for crate::CallBuilder<'a, ResponsesCall, H>
{
    type Output = Result<ResponsesOutput, Error>;
    type IntoFuture = futures_util::future::BoxFuture<'a, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            litellm_host::lifecycle::observe_call(self.hooks.observer(), async {
                let http = self.client.provider_http()?;
                execute(
                    &http,
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

async fn execute(
    http: &litellm_http::Client,
    auth: &litellm_auth::AuthServices,
    secrets: &dyn SecretSource,
    call: ResponsesCall,
    hooks: &impl RouteHooks<Error>,
) -> Result<ResponsesOutput, Error> {
    let request = prepare::prepare(call, secrets).await?;
    handler::execute(http, auth, request, hooks).await
}
