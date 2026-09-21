use litellm_core_utils::call_arguments::CallArguments;
use litellm_types::responses::streaming::{
    ResponseInputItem, ResponsesEvent, ResponsesInput, ResponsesRequest, ResponsesResponse,
};
use litellm_types::responses::streaming_websocket::{ResponsesWsEvent, ResponsesWsTransformResult};
use serde_json::Value;

use crate::base_llm::chat::transformation::Error;
use crate::base_llm::translation::{
    ProviderIdentity, ProviderResponse, StreamLimits, StreamTranslation, TranslationError,
};

pub const OPENAI_RESPONSES_DEFAULT_API_BASE: &str = "https://api.openai.com/v1";
pub const OPENAI_RESPONSES_PATH: &str = "/responses";

pub trait ResponsesWebSocketProviderConfig: Sync {
    fn supports_native_websocket(&self) -> bool {
        false
    }

    fn model_in_websocket_url(&self) -> bool {
        true
    }

    fn complete_websocket_url(&self, api_base: Option<&str>, model: &str) -> String {
        complete_websocket_url(api_base, model, self.model_in_websocket_url())
    }

    fn transform_ws_request(
        &self,
        event: &ResponsesWsEvent,
        model: &str,
    ) -> Result<ResponsesWsTransformResult, Error>;

    fn transform_ws_response(
        &self,
        event: &ResponsesWsEvent,
        model: &str,
    ) -> Result<ResponsesWsTransformResult, Error>;
}

pub struct ResponseReference {
    pub response_id: String,
}

pub enum InputItemsOrder {
    Asc,
    Desc,
}

pub struct ListInputItemsRequest {
    pub response_id: String,
    pub after: Option<String>,
    pub before: Option<String>,
    pub include: Option<Box<[String]>>,
    pub limit: u64,
    pub order: InputItemsOrder,
}

pub struct DeleteResponseResult {
    pub id: String,
    pub object: String,
    pub deleted: bool,
}

pub struct ListInputItemsResponse {
    pub object: String,
    pub data: Box<[ResponseInputItem]>,
    pub first_id: Option<String>,
    pub last_id: Option<String>,
    pub has_more: bool,
}

pub trait BaseResponsesApiConfig: Send + Sync {
    type Params;
    type ParsedChunk;
    type Credentials;
    type Environment;
    type ProviderRequest: serde::Serialize;
    type Decoder: crate::base_llm::translation::SemanticDecoder<
            Event = litellm_types::responses::streaming::ResponsesEvent,
            Completion = litellm_types::responses::streaming::ResponsesResponse,
        >;

    fn custom_llm_provider(&self) -> &'static str;

    fn get_config(&self) -> std::collections::BTreeMap<String, Value>;

    fn supports_native_file_search(&self) -> bool;

    fn supports_encrypted_agent_messages(&self) -> bool;

    fn sign_request(
        &self,
        _request: reqwest::Request,
        _environment: &Self::Environment,
    ) -> Result<crate::base_llm::translation::SignedRequest, TranslationError>;

    fn get_supported_openai_params(&self, _provider: &ProviderIdentity) -> Box<[String]>;

    fn map_openai_params(
        &self,
        _arguments: &CallArguments,
        _provider: &ProviderIdentity,
    ) -> Result<Self::Params, TranslationError>;

    fn validate_environment(
        &self,
        _credentials: &Self::Credentials,
        _arguments: &litellm_core_utils::call_arguments::CallArguments,
        _headers: &[(String, String)],
        _api_base: Option<&str>,
        _params: &Self::Params,
        _provider: &ProviderIdentity,
    ) -> Result<Self::Environment, TranslationError>;

    fn get_complete_url(
        &self,
        _api_base: Option<&str>,
        _params: &Self::Params,
        _provider: &ProviderIdentity,
        _environment: &Self::Environment,
    ) -> Result<String, TranslationError>;

    fn transform_responses_api_request(
        &self,
        _request: &ResponsesRequest,
        _params: &Self::Params,
        _provider: &ProviderIdentity,
        _environment: &Self::Environment,
    ) -> Result<Self::ProviderRequest, TranslationError>;

    fn transform_response_api_response(
        &self,
        _response: ProviderResponse<'_>,
        _provider: &ProviderIdentity,
    ) -> Result<ResponsesResponse, TranslationError>;

    fn transform_streaming_response(
        &self,
        _provider: &ProviderIdentity,
        _parsed_chunk: Self::ParsedChunk,
    ) -> Result<ResponsesEvent, TranslationError>;

    fn transform_delete_response_api_request(
        &self,
        _request: &ResponseReference,
        _api_base: &str,
        _provider: &ProviderIdentity,
        _environment: &Self::Environment,
    ) -> Result<reqwest::Request, TranslationError>;

    fn transform_delete_response_api_response(
        &self,
        _response: ProviderResponse<'_>,
    ) -> Result<DeleteResponseResult, TranslationError>;

    fn transform_get_response_api_request(
        &self,
        _request: &ResponseReference,
        _api_base: &str,
        _provider: &ProviderIdentity,
        _environment: &Self::Environment,
    ) -> Result<reqwest::Request, TranslationError>;

    fn transform_get_response_api_response(
        &self,
        _response: ProviderResponse<'_>,
    ) -> Result<ResponsesResponse, TranslationError>;

    fn transform_list_input_items_request(
        &self,
        _request: &ListInputItemsRequest,
        _api_base: &str,
        _provider: &ProviderIdentity,
        _environment: &Self::Environment,
    ) -> Result<reqwest::Request, TranslationError>;

    fn transform_list_input_items_response(
        &self,
        _response: ProviderResponse<'_>,
    ) -> Result<ListInputItemsResponse, TranslationError>;

    fn get_error_class(
        &self,
        _error_message: &str,
        _response: ProviderResponse<'_>,
    ) -> TranslationError;

    fn should_fake_stream(&self, _params: &Self::Params) -> bool;

    fn supports_native_websocket(&self) -> bool;

    fn get_websocket_url(
        &self,
        _api_base: Option<&str>,
        _params: &Self::Params,
        _provider: &ProviderIdentity,
    ) -> Result<String, TranslationError>;

    fn model_in_websocket_url(&self) -> bool;

    fn transform_cancel_response_api_request(
        &self,
        _request: &ResponseReference,
        _api_base: &str,
        _provider: &ProviderIdentity,
        _environment: &Self::Environment,
    ) -> Result<reqwest::Request, TranslationError>;

    fn transform_cancel_response_api_response(
        &self,
        _response: ProviderResponse<'_>,
    ) -> Result<ResponsesResponse, TranslationError>;

    fn transform_compact_response_api_request(
        &self,
        _request: &ResponsesRequest,
        _api_base: &str,
        _provider: &ProviderIdentity,
        _environment: &Self::Environment,
    ) -> Result<reqwest::Request, TranslationError>;

    fn transform_compact_response_api_response(
        &self,
        _response: ProviderResponse<'_>,
    ) -> Result<ResponsesResponse, TranslationError>;

    fn strip_custom_tool_call_namespace_from_responses_input(
        _input: ResponsesInput,
    ) -> Result<ResponsesInput, TranslationError>;

    fn normalize_responses_api_request_dict(
        _request: Self::ProviderRequest,
    ) -> Result<Self::ProviderRequest, TranslationError>;

    fn stream_decoder(
        &self,
        _provider: &ProviderIdentity,
        _limits: StreamLimits,
    ) -> Result<StreamTranslation<Self::Decoder>, TranslationError>;
}

macro_rules! scaffold_responses_translation_types {
    () => {
        pub enum Parameters {}
        pub enum Credentials {}
        pub enum Environment {}

        #[derive(serde::Serialize)]
        pub enum ProviderRequest {}

        pub enum ProviderResponse {}
        pub enum ProviderFrame {}

        #[derive(Default)]
        pub struct Decoder;

        impl crate::base_llm::translation::SemanticDecoder for Decoder {
            type Frame = ProviderFrame;
            type Event = litellm_types::responses::streaming::ResponsesEvent;
            type Completion = litellm_types::responses::streaming::ResponsesResponse;

            fn decode(
                &mut self,
                _frame: Self::Frame,
            ) -> Result<
                crate::base_llm::translation::StreamProgress<Self::Event, Self::Completion>,
                crate::base_llm::translation::TranslationError,
            > {
                todo!()
            }

            fn end_of_input(
                self,
            ) -> Result<
                crate::base_llm::base_model_iterator::StreamOutcome<Self::Completion>,
                crate::base_llm::translation::TranslationError,
            > {
                todo!()
            }
        }
    };
}

macro_rules! scaffold_responses_translation_config {
    ($config:ident) => {
        #[derive(Default)]
        pub struct $config;

        $crate::base_llm::responses::transformation::scaffold_responses_translation_config!(@existing $config);
    };
    (@existing $config:ident) => {
        impl crate::base_llm::responses::transformation::BaseResponsesApiConfig for $config {
            type Params = Parameters;
            type ParsedChunk = ProviderResponse;
            type Credentials = Credentials;
            type Environment = Environment;
            type ProviderRequest = ProviderRequest;
            type Decoder = Decoder;

    fn custom_llm_provider(&self) -> &'static str {
        todo!()
    }

    fn get_config(&self) -> std::collections::BTreeMap<String, serde_json::Value> {
        todo!()
    }

    fn supports_native_file_search(&self) -> bool {
        todo!()
    }

    fn supports_encrypted_agent_messages(&self) -> bool {
        todo!()
    }

    fn sign_request(
        &self,
        _request: reqwest::Request,
        _environment: &Self::Environment,
    ) -> Result<crate::base_llm::translation::SignedRequest, crate::base_llm::translation::TranslationError> {
        todo!()
    }

    fn get_supported_openai_params(&self, _provider: &crate::base_llm::translation::ProviderIdentity) -> Box<[String]> {
        todo!()
    }

    fn map_openai_params(
        &self,
        _arguments: &litellm_core_utils::call_arguments::CallArguments,
        _provider: &crate::base_llm::translation::ProviderIdentity,
    ) -> Result<Self::Params, crate::base_llm::translation::TranslationError> {
        todo!()
    }

    fn validate_environment(
        &self,
        _credentials: &Self::Credentials,
        _arguments: &litellm_core_utils::call_arguments::CallArguments,
        _headers: &[(String, String)],
        _api_base: Option<&str>,
        _params: &Self::Params,
        _provider: &crate::base_llm::translation::ProviderIdentity,
    ) -> Result<Self::Environment, crate::base_llm::translation::TranslationError> {
        todo!()
    }

    fn get_complete_url(
        &self,
        _api_base: Option<&str>,
        _params: &Self::Params,
        _provider: &crate::base_llm::translation::ProviderIdentity,
        _environment: &Self::Environment,
    ) -> Result<String, crate::base_llm::translation::TranslationError> {
        todo!()
    }

    fn transform_responses_api_request(
        &self,
        _request: &litellm_types::responses::streaming::ResponsesRequest,
        _params: &Self::Params,
        _provider: &crate::base_llm::translation::ProviderIdentity,
        _environment: &Self::Environment,
    ) -> Result<Self::ProviderRequest, crate::base_llm::translation::TranslationError> {
        todo!()
    }

    fn transform_response_api_response(
        &self,
        _response: crate::base_llm::translation::ProviderResponse<'_>,
        _provider: &crate::base_llm::translation::ProviderIdentity,
    ) -> Result<litellm_types::responses::streaming::ResponsesResponse, crate::base_llm::translation::TranslationError> {
        todo!()
    }

    fn transform_streaming_response(&self, _provider: &crate::base_llm::translation::ProviderIdentity, _parsed_chunk: Self::ParsedChunk) -> Result<litellm_types::responses::streaming::ResponsesEvent, crate::base_llm::translation::TranslationError> {
        todo!()
    }

    fn transform_delete_response_api_request(
        &self,
        _request: &crate::base_llm::responses::transformation::ResponseReference,
        _api_base: &str,
        _provider: &crate::base_llm::translation::ProviderIdentity,
        _environment: &Self::Environment,
    ) -> Result<reqwest::Request, crate::base_llm::translation::TranslationError> {
        todo!()
    }

    fn transform_delete_response_api_response(
        &self,
        _response: crate::base_llm::translation::ProviderResponse<'_>,
    ) -> Result<crate::base_llm::responses::transformation::DeleteResponseResult, crate::base_llm::translation::TranslationError> {
        todo!()
    }

    fn transform_get_response_api_request(
        &self,
        _request: &crate::base_llm::responses::transformation::ResponseReference,
        _api_base: &str,
        _provider: &crate::base_llm::translation::ProviderIdentity,
        _environment: &Self::Environment,
    ) -> Result<reqwest::Request, crate::base_llm::translation::TranslationError> {
        todo!()
    }

    fn transform_get_response_api_response(
        &self,
        _response: crate::base_llm::translation::ProviderResponse<'_>,
    ) -> Result<litellm_types::responses::streaming::ResponsesResponse, crate::base_llm::translation::TranslationError> {
        todo!()
    }

    fn transform_list_input_items_request(
        &self,
        _request: &crate::base_llm::responses::transformation::ListInputItemsRequest,
        _api_base: &str,
        _provider: &crate::base_llm::translation::ProviderIdentity,
        _environment: &Self::Environment,
    ) -> Result<reqwest::Request, crate::base_llm::translation::TranslationError> {
        todo!()
    }

    fn transform_list_input_items_response(
        &self,
        _response: crate::base_llm::translation::ProviderResponse<'_>,
    ) -> Result<crate::base_llm::responses::transformation::ListInputItemsResponse, crate::base_llm::translation::TranslationError> {
        todo!()
    }

    fn get_error_class(&self, _error_message: &str, _response: crate::base_llm::translation::ProviderResponse<'_>) -> crate::base_llm::translation::TranslationError {
        todo!()
    }

    fn should_fake_stream(&self, _params: &Self::Params) -> bool {
        todo!()
    }

    fn supports_native_websocket(&self) -> bool {
        todo!()
    }

    fn get_websocket_url(
        &self,
        _api_base: Option<&str>,
        _params: &Self::Params,
        _provider: &crate::base_llm::translation::ProviderIdentity,
    ) -> Result<String, crate::base_llm::translation::TranslationError> {
        todo!()
    }

    fn model_in_websocket_url(&self) -> bool {
        todo!()
    }

    fn transform_cancel_response_api_request(
        &self,
        _request: &crate::base_llm::responses::transformation::ResponseReference,
        _api_base: &str,
        _provider: &crate::base_llm::translation::ProviderIdentity,
        _environment: &Self::Environment,
    ) -> Result<reqwest::Request, crate::base_llm::translation::TranslationError> {
        todo!()
    }

    fn transform_cancel_response_api_response(
        &self,
        _response: crate::base_llm::translation::ProviderResponse<'_>,
    ) -> Result<litellm_types::responses::streaming::ResponsesResponse, crate::base_llm::translation::TranslationError> {
        todo!()
    }

    fn transform_compact_response_api_request(
        &self,
        _request: &litellm_types::responses::streaming::ResponsesRequest,
        _api_base: &str,
        _provider: &crate::base_llm::translation::ProviderIdentity,
        _environment: &Self::Environment,
    ) -> Result<reqwest::Request, crate::base_llm::translation::TranslationError> {
        todo!()
    }

    fn transform_compact_response_api_response(
        &self,
        _response: crate::base_llm::translation::ProviderResponse<'_>,
    ) -> Result<litellm_types::responses::streaming::ResponsesResponse, crate::base_llm::translation::TranslationError> {
        todo!()
    }

    fn strip_custom_tool_call_namespace_from_responses_input(
        _input: litellm_types::responses::streaming::ResponsesInput,
    ) -> Result<litellm_types::responses::streaming::ResponsesInput, crate::base_llm::translation::TranslationError> {
        todo!()
    }

    fn normalize_responses_api_request_dict(_request: Self::ProviderRequest) -> Result<Self::ProviderRequest, crate::base_llm::translation::TranslationError> {
        todo!()
    }

    fn stream_decoder(
        &self,
        _provider: &crate::base_llm::translation::ProviderIdentity,
        _limits: crate::base_llm::translation::StreamLimits,
    ) -> Result<crate::base_llm::translation::StreamTranslation<Self::Decoder>, crate::base_llm::translation::TranslationError> {
        todo!()
    }
        }
    };
}

pub(crate) use scaffold_responses_translation_config;
pub(crate) use scaffold_responses_translation_types;

pub fn complete_websocket_url(
    api_base: Option<&str>,
    model: &str,
    model_in_websocket_url: bool,
) -> String {
    let base = api_base
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(OPENAI_RESPONSES_DEFAULT_API_BASE);
    let (base_without_query, query) = base
        .split_once('?')
        .map_or((base, None), |(value, query)| (value, Some(query)));
    let response_url = format!(
        "{}{}",
        base_without_query.trim_end_matches('/'),
        OPENAI_RESPONSES_PATH
    );
    let scheme_flipped = if let Some(rest) = response_url.strip_prefix("https://") {
        format!("wss://{rest}")
    } else if let Some(rest) = response_url.strip_prefix("http://") {
        format!("ws://{rest}")
    } else {
        response_url
    };
    let url = query.map_or(scheme_flipped.clone(), |value| {
        format!("{scheme_flipped}?{value}")
    });
    if !model_in_websocket_url
        || query.is_some_and(|value| {
            value
                .split('&')
                .any(|part| part.split('=').next() == Some("model"))
        })
    {
        return url;
    }
    format!(
        "{url}{}model={}",
        if query.is_some() { "&" } else { "?" },
        percent_encode(model)
    )
}

fn percent_encode(value: &str) -> String {
    value
        .bytes()
        .map(|byte| {
            if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~') {
                format!("{}", byte as char)
            } else {
                format!("%{byte:02X}")
            }
        })
        .collect()
}

pub fn enforce_model(event: &ResponsesWsEvent, model: &str) -> ResponsesWsEvent {
    if !event.is_response_create() {
        return event.clone();
    }
    let mut enforced = event.clone();
    let has_flat_model = enforced.data.contains_key("model");
    if let Some(response) = enforced
        .data
        .get_mut("response")
        .and_then(serde_json::Value::as_object_mut)
    {
        response.insert(
            "model".to_string(),
            serde_json::Value::String(model.to_string()),
        );
        if has_flat_model {
            enforced.data.insert(
                "model".to_string(),
                serde_json::Value::String(model.to_string()),
            );
        }
    } else {
        enforced.data.insert(
            "model".to_string(),
            serde_json::Value::String(model.to_string()),
        );
    }
    enforced
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(value: serde_json::Value) -> ResponsesWsEvent {
        serde_json::from_value(value).expect("valid event")
    }

    #[test]
    fn url_construction_matches_python_defaults_and_query_behavior() {
        assert_eq!(
            complete_websocket_url(None, "gpt-5", true),
            "wss://api.openai.com/v1/responses?model=gpt-5"
        );
        assert_eq!(
            complete_websocket_url(Some("http://localhost:8080/"), "gpt 5", true),
            "ws://localhost:8080/responses?model=gpt%205"
        );
        assert_eq!(
            complete_websocket_url(Some("https://example.test/v1?foo=bar"), "gpt-5", true),
            "wss://example.test/v1/responses?foo=bar&model=gpt-5"
        );
        assert_eq!(
            complete_websocket_url(Some("https://example.test?model=existing"), "gpt-5", true),
            "wss://example.test/responses?model=existing"
        );
    }

    #[test]
    fn enforce_model_overrides_flat_and_nested_values() {
        let flat = enforce_model(
            &event(serde_json::json!({"type":"response.create","model":"wrong"})),
            "gpt-5",
        );
        assert_eq!(flat.model(), Some("gpt-5"));
        let nested = enforce_model(
            &event(serde_json::json!({
                "type":"response.create",
                "model":"wrong",
                "response":{"model":"also-wrong"}
            })),
            "gpt-5",
        );
        assert_eq!(nested.model(), Some("gpt-5"));
        assert_eq!(
            nested
                .data
                .get("response")
                .and_then(|value| value.get("model")),
            Some(&serde_json::json!("gpt-5"))
        );
        let nested_without_flat = enforce_model(
            &event(serde_json::json!({
                "type":"response.create",
                "response":{"model":"also-wrong"}
            })),
            "gpt-5",
        );
        assert!(!nested_without_flat.data.contains_key("model"));
    }
}
