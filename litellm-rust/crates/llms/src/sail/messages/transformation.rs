use litellm_llms_types::formats::messages::{MessagesOptionalParams, MessagesRequest};

use crate::{
    Error,
    base_llm::{
        auth::{Headers, ValidatedEnvironment},
        messages::{context::MessagesTransformContext, transformation::BaseMessagesConfig},
    },
    openai_like::{common_utils::JsonProvider, messages::transformation::OpenAILikeMessagesConfig},
};

const SAIL_PROVIDER: JsonProvider = JsonProvider::new(
    "https://api.sailresearch.com/v1",
    "SAIL_API_KEY",
    Some("SAIL_API_BASE"),
    false,
);
const COMPATIBLE_CONFIG: OpenAILikeMessagesConfig =
    OpenAILikeMessagesConfig::Registry(&SAIL_PROVIDER);

pub struct SailMessagesConfig;

pub const SAIL_MESSAGES_CONFIG: SailMessagesConfig = SailMessagesConfig;

impl BaseMessagesConfig for SailMessagesConfig {
    fn shape_request(
        &self,
        request: MessagesRequest,
        reasoning_auto_summary: bool,
    ) -> Result<MessagesRequest, Error> {
        COMPATIBLE_CONFIG.shape_request(request, reasoning_auto_summary)
    }

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        COMPATIBLE_CONFIG.get_complete_url(api_base, model, env_lookup)
    }

    fn transform_anthropic_messages_request(
        &self,
        request: MessagesRequest,
        context: &MessagesTransformContext,
    ) -> Result<MessagesRequest, Error> {
        COMPATIBLE_CONFIG.transform_anthropic_messages_request(
            MessagesRequest {
                params: MessagesOptionalParams {
                    service_tier: None,
                    ..request.params
                },
                ..request
            },
            context,
        )
    }

    fn secret_names(&self) -> &'static [&'static str] {
        COMPATIBLE_CONFIG.secret_names()
    }

    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        COMPATIBLE_CONFIG.validate_environment(headers, api_key, model, env_lookup)
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        COMPATIBLE_CONFIG.default_headers()
    }

    fn request_headers(&self, headers: Headers, request: &MessagesRequest) -> Headers {
        COMPATIBLE_CONFIG.request_headers(headers, request)
    }
}

#[cfg(test)]
mod tests {
    use litellm_llms_types::formats::messages::{Message, MessageContent, MessageRole};
    use rstest::{fixture, rstest};

    use super::*;

    #[fixture]
    fn messages_request() -> MessagesRequest {
        MessagesRequest {
            model: "test-model".into(),
            messages: vec![Message {
                role: MessageRole::User,
                content: MessageContent::Text("hi".into()),
                extra: Default::default(),
            }],
            params: MessagesOptionalParams {
                max_tokens: Some(16),
                ..Default::default()
            },
        }
    }

    #[rstest]
    #[case::none(None)]
    #[case::auto(Some("auto"))]
    #[case::priority(Some("priority"))]
    #[case::flex(Some("flex"))]
    #[case::balanced(Some("balanced"))]
    #[case::scale(Some("scale"))]
    fn test_sail_messages_send_no_window_whatever_the_tier(
        messages_request: MessagesRequest,
        #[case] service_tier: Option<&str>,
    ) {
        let expected_messages = messages_request.messages.clone();
        let request = MessagesRequest {
            params: MessagesOptionalParams {
                service_tier: service_tier.map(str::to_string),
                ..messages_request.params
            },
            ..messages_request
        };
        let result = SAIL_MESSAGES_CONFIG
            .transform_anthropic_messages_request(request, &MessagesTransformContext::default())
            .unwrap();
        assert_eq!(result.messages, expected_messages);
        let body = SAIL_MESSAGES_CONFIG.request_body(&result).unwrap();
        assert!(body.get("service_tier").is_none());
        assert!(
            body.get("metadata")
                .and_then(|metadata| metadata.get("completion_window"))
                .is_none()
        );
    }

    #[rstest]
    #[case::default(None, None, "https://api.sailresearch.com/v1/messages")]
    #[case::environment(
        None,
        Some("https://env.example/v1"),
        "https://env.example/v1/messages"
    )]
    #[case::explicit(
        Some("https://caller.example"),
        Some("https://env.example/v1"),
        "https://caller.example/v1/messages"
    )]
    fn sail_messages_endpoint_uses_registry_settings(
        #[case] api_base: Option<&str>,
        #[case] env_base: Option<&str>,
        #[case] expected: &str,
    ) {
        let lookup = |name: &str| {
            (name == "SAIL_API_BASE")
                .then(|| env_base.map(str::to_string))
                .flatten()
        };
        assert_eq!(
            SAIL_MESSAGES_CONFIG
                .get_complete_url(api_base, "m", &lookup)
                .unwrap(),
            expected
        );
    }
}
