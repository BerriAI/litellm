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
    pub async fn responses(&self, call: ResponsesCall) -> Result<ResponsesOutput, Error> {
        self.responses_with_hooks(call, &()).await
    }

    pub async fn responses_with_hooks(
        &self,
        call: ResponsesCall,
        hooks: &impl RouteHooks<Error>,
    ) -> Result<ResponsesOutput, Error> {
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
    call: ResponsesCall,
    hooks: &impl RouteHooks<Error>,
) -> Result<ResponsesOutput, Error> {
    let request = prepare::prepare(call, secrets).await?;
    handler::execute(http, auth, request, hooks).await
}
