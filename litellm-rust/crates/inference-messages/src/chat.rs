//! A Messages request for a deployment with no Messages API of its own, served through its
//! Chat Completions API as Python's `LiteLLMMessagesToCompletionTransformationHandler` serves it.

use std::sync::Arc;

use futures_util::{StreamExt, TryStreamExt, future::ready};
use litellm_host::{
    call::{CallOutput, hosted_call},
    interceptors::Interceptors,
};
use litellm_inference::{CallOptions, context::CallContext, provider::resolve_llm_provider};
use litellm_inference_chat::{ChatCompletionsRoute, openai::OpenAiChatCall};
use litellm_llms::{
    anthropic::adapters::{
        streaming_iterator::anthropic_events,
        transformation::{
            ChatRequest, UpstreamReport, translate_anthropic_to_openai,
            translate_openai_response_to_anthropic,
        },
    },
    base_llm::{messages::streaming, model_info::ModelInfo},
};
use litellm_router_types::LitellmParams;
use serde_json::{Map, Value};

use crate::{
    Error, MessagesCall, MessagesCallResponse,
    common_utils::messages_provider,
    route::{MessagesMachine, MessagesStreamHead},
};

/// Serves a Messages call through the Chat Completions route. Kept apart from
/// [`crate::MessagesRoute`] so a host that never routes Messages to chat providers does not
/// link the chat providers' code.
#[derive(Clone)]
pub struct ChatCompletions {
    pub route: ChatCompletionsRoute,
    /// `litellm_settings.drop_params`, which the chat param rules read beside the deployment's.
    pub drop_params: bool,
    /// The model cost map, which the chat param rules read.
    pub model_info: Option<Arc<dyn ModelInfo>>,
}

impl ChatCompletions {
    /// Whether `call` names a deployment with no Messages config that Chat Completions serves.
    pub fn serves(call: &MessagesCall) -> bool {
        let provider = call.custom_llm_provider.as_deref();
        resolve_llm_provider(&call.body.model, provider, "messages")
            .is_ok_and(|resolved| messages_provider(resolved.provider, resolved.model).is_none())
            && ChatCompletionsRoute::serves_openai(&call.body.model, provider)
    }

    /// The call as a hosted machine, as [`crate::MessagesRoute::machine`] runs a native one. The
    /// usage and model the provider reports go to `report`, apart from the placeholder counts
    /// the client sees before the provider sends any.
    pub fn machine(self, call: MessagesCall, report: UpstreamReport) -> MessagesMachine {
        hosted_call(
            call,
            None,
            move |call, _, interceptors, observers| async move {
                let context = CallContext::new(
                    &interceptors,
                    CallOptions {
                        cache: None,
                        observers,
                    },
                );
                self.run(call, context, report).await
            },
        )
    }

    async fn run(
        &self,
        call: MessagesCall,
        context: CallContext<'_, impl Interceptors<Error>>,
        report: UpstreamReport,
    ) -> Result<MessagesCallResponse, Error> {
        let MessagesCall {
            body,
            api_key,
            api_base,
            custom_llm_provider,
            litellm_params,
            extra_headers,
            timeout,
            shaping,
            ..
        } = call;
        let model = body.model.clone();
        let ChatRequest { body, tool_names } = translate_anthropic_to_openai(body)?;
        let drop_params = self.drop_params
            || shaping.settings.drop_params
            || litellm_params
                .drop_params
                .as_ref()
                .is_some_and(|flag| flag.is_set());
        let call = OpenAiChatCall {
            model: model.clone(),
            body,
            api_key,
            api_base: api_base.or_else(|| litellm_params.api_base.clone()),
            custom_llm_provider,
            extra_headers,
            timeout: timeout.or_else(|| shaping.settings.request_timeout()),
            litellm_params: provider_params(&litellm_params),
            drop_params,
            model_info: self.model_info.clone(),
            ..OpenAiChatCall::default()
        };
        Ok(
            match Box::pin(self.route.run_openai(call, context)).await? {
                CallOutput::Complete(body) => CallOutput::Complete(Box::new(
                    translate_openai_response_to_anthropic(&body, &model, &tool_names, &report)?,
                )),
                CallOutput::Stream { chunks, .. } => CallOutput::Stream {
                    head: MessagesStreamHead {
                        headers: Vec::new(),
                    },
                    chunks: anthropic_events(
                        chunks.map_err(std::io::Error::other).boxed(),
                        model,
                        tool_names,
                        report,
                    )
                    .and_then(|event| ready(streaming::encode(&event)))
                    .map_err(Error::from)
                    .boxed(),
                },
            },
        )
    }
}

/// The deployment's `litellm_params` a chat provider reads beyond its connection, such as
/// Azure's `api_version` and `base_model`.
fn provider_params(params: &LitellmParams) -> Map<String, Value> {
    params
        .extra
        .iter()
        .map(|(name, value)| (name.clone(), value.clone()))
        .chain(
            params
                .api_version
                .clone()
                .map(|version| ("api_version".to_owned(), Value::String(version))),
        )
        .collect()
}
