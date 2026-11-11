mod handler;

use std::{convert::Infallible, sync::Arc, time::Duration};

use litellm_auth::AuthServices;
use litellm_host::{
    call::{CallOutput, HostedMachine, hosted_call},
    interceptors::Interceptors,
    observation::ObservationSender,
    protocol::Protocol,
};
use litellm_inference::{CallOptions, context::CallContext};
use litellm_llms::base_llm::model_info::ModelInfo;
use litellm_llms_types::formats::decisions::{DecisionsRequest, DecisionsResponse};
use litellm_secrets::source::SecretSource;
use serde_json::{Map, Value};

pub use handler::serves;
pub use litellm_inference::RouteError as Error;
pub use litellm_llms::base_llm::decisions::parse_request;

#[derive(Clone)]
pub struct DecisionsRoute {
    http: litellm_http::Client,
    auth: Arc<AuthServices>,
    secrets: Arc<dyn SecretSource>,
}

pub struct DecisionsCall {
    pub request: Arc<DecisionsRequest>,
    pub model: String,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub custom_llm_provider: Option<String>,
    pub litellm_params: Map<String, Value>,
    pub extra_headers: Option<Map<String, Value>>,
    pub timeout: Option<Duration>,
    pub drop_params: bool,
    pub supports_box_questions: Option<bool>,
    pub model_info: Option<Arc<dyn ModelInfo>>,
}

pub struct Decisions;

impl Protocol for Decisions {
    type Response = DecisionsResponse;
    type Error = Error;
    type Request = DecisionsCall;
    type HostCall = Infallible;
    type Chunk = Infallible;
    type StreamHead = Infallible;
}

impl DecisionsRoute {
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

    pub fn machine(
        self,
        call: DecisionsCall,
        observers: Option<ObservationSender>,
    ) -> HostedMachine<Decisions> {
        hosted_call(
            call,
            observers,
            move |call, _, interceptors, observers| async move {
                let context = CallContext::new(
                    &interceptors,
                    CallOptions {
                        cache: None,
                        observers,
                    },
                );
                self.run(call, context).await.map(CallOutput::Complete)
            },
        )
    }

    pub async fn execute(
        &self,
        call: DecisionsCall,
        interceptors: &impl Interceptors<Error>,
        observers: Option<ObservationSender>,
    ) -> Result<DecisionsResponse, Error> {
        let context = CallContext::new(
            interceptors,
            CallOptions {
                cache: None,
                observers,
            },
        );
        litellm_host::lifecycle::observe_unary(context.observers.clone(), self.run(call, context))
            .await
    }

    #[tracing::instrument(name = "litellm.route", skip_all, fields(
        route = "decisions",
        model = %call.model,
        provider,
        resolved_model,
        stream = false,
        outcome
    ))]
    async fn run(
        &self,
        call: DecisionsCall,
        context: CallContext<'_, impl Interceptors<Error>>,
    ) -> Result<DecisionsResponse, Error> {
        litellm_inference::diagnostic::unary(Box::pin(handler::execute(self, call, context))).await
    }
}
