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
    route = "responses",
    model = %call.model,
    provider,
    resolved_model,
    stream = call.optional_params.get("stream").and_then(serde_json::Value::as_bool).unwrap_or(false),
    outcome
))]
async fn execute(
    http: Result<litellm_http::Client, litellm_http::Error>,
    auth: &litellm_auth::AuthServices,
    secrets: &dyn SecretSource,
    call: ResponsesCall,
    hooks: &impl RouteHooks<Error>,
) -> Result<ResponsesOutput, Error> {
    crate::diagnostic::call(async {
        let http = http?;
        let request = prepare::prepare(call, secrets).await?;
        crate::diagnostic::provider(&request.context.model, &request.context.custom_llm_provider);
        let execute: futures_util::future::BoxFuture<'_, Result<ResponsesOutput, Error>> =
            Box::pin(handler::execute(&http, auth, request, hooks));
        execute.await
    })
    .await
}
