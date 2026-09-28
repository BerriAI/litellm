pub use crate::error::RouteError as Error;
use litellm_host::observation::ObservationSender;
pub mod websocket;

mod handler;
mod prepare;
pub mod route;
pub mod types;

use std::sync::Arc;

use litellm_auth::AuthServices;
use litellm_host::interceptors::Interceptors;
use litellm_secrets::source::SecretSource;
use types::{ResponsesCall, ResponsesOutput};

#[derive(Clone)]
pub struct ResponsesRoute {
    http: litellm_http::Client,
    auth: Arc<AuthServices>,
    secrets: Arc<dyn SecretSource>,
}

impl ResponsesRoute {
    pub fn new(
        http: litellm_http::Client,
        auth: Arc<AuthServices>,
        secrets: Arc<dyn SecretSource>,
    ) -> Self {
        Self {
            http,
            auth,
            secrets,
        }
    }

    pub async fn execute(
        &self,
        call: ResponsesCall,
        interceptors: &impl Interceptors<Error>,
        observers: Option<ObservationSender>,
    ) -> Result<ResponsesOutput, Error> {
        litellm_host::lifecycle::observe_call(
            observers.clone(),
            self.run(call, interceptors, observers.as_ref()),
        )
        .await
    }

    #[tracing::instrument(name = "litellm.route", skip_all, fields(
        route = "responses",
        model = %call.model,
        provider,
        resolved_model,
        stream = call.optional_params.get("stream").and_then(serde_json::Value::as_bool).unwrap_or(false),
        outcome
    ))]
    async fn run(
        &self,
        call: ResponsesCall,
        interceptors: &impl Interceptors<Error>,
        observers: Option<&ObservationSender>,
    ) -> Result<ResponsesOutput, Error> {
        crate::diagnostic::call(async {
            let request = prepare::prepare(call, self.secrets.as_ref()).await?;
            crate::diagnostic::provider(
                &request.context.model,
                &request.context.custom_llm_provider,
            );
            let execute: futures_util::future::BoxFuture<'_, Result<ResponsesOutput, Error>> =
                Box::pin(handler::execute(
                    &self.http,
                    &self.auth,
                    request,
                    interceptors,
                    observers,
                ));
            execute.await
        })
        .await
    }
}
