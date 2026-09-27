use crate::catalog::ModelInfoCatalog;

use serde_json::Value;

use crate::call_type::CallTypes;
use crate::error::CostError;
use crate::model_selection::{
    ModelSelectionRequest, get_response_model, select_model_name_for_cost_calc,
};
use crate::provider::LlmProviders;
use crate::responses_usage::ChatUsage;
use crate::usage_dispatch::get_usage_object;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResponseKind {
    Completion,
    Embedding,
    Transcription,
    Speech,
    Rerank,
    ImageGeneration,
    TextCompletion,
    SendMessage,
}

impl ResponseKind {
    fn call_type(self) -> CallTypes {
        match self {
            Self::Completion => CallTypes::completion,
            Self::Embedding => CallTypes::embedding,
            Self::Transcription => CallTypes::transcription,
            Self::Speech => CallTypes::speech,
            Self::Rerank => CallTypes::rerank,
            Self::ImageGeneration => CallTypes::image_generation,
            Self::TextCompletion => CallTypes::text_completion,
            Self::SendMessage => CallTypes::send_message,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CompletionInputRequest<'a> {
    pub model_selection: ModelSelectionRequest<'a>,
    pub call_type: Option<&'a str>,
    pub response_kind: Option<ResponseKind>,
    pub service_tier: Option<&'a Value>,
    pub optional_params: Option<&'a Value>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedCompletionInput {
    pub call_type: String,
    pub model_candidates: [Option<String>; 3],
    pub service_tier: Option<String>,
    pub usage: Option<ChatUsage>,
}

pub fn infer_call_type(explicit: Option<&str>, kind: Option<ResponseKind>) -> Option<&str> {
    explicit.or_else(|| kind.map(|kind| kind.call_type().as_str()))
}

pub fn map_traffic_type_to_service_tier(traffic_type: Option<&str>) -> Option<&'static str> {
    match traffic_type?.to_ascii_uppercase().as_str() {
        "ON_DEMAND_PRIORITY" => Some("priority"),
        "FLEX" | "BATCH" | "ON_DEMAND_FLEX" => Some("flex"),
        _ => None,
    }
}

pub fn normalize_service_tier(value: Option<&Value>) -> Option<&str> {
    value?
        .as_str()
        .filter(|tier| !tier.eq_ignore_ascii_case("auto"))
}

pub fn extract_service_tier(source: Option<&Value>) -> Option<&Value> {
    source?.get("service_tier")
}

pub fn select_service_tier(
    explicit: Option<&Value>,
    optional_params: Option<&Value>,
    response: Option<&Value>,
    hidden_params: Option<&Value>,
) -> Option<String> {
    normalize_service_tier(
        explicit.or_else(|| optional_params.and_then(|value| value.get("service_tier"))),
    )
    .or_else(|| normalize_service_tier(extract_service_tier(response)))
    .or_else(|| {
        normalize_service_tier(extract_service_tier(
            response.and_then(|value| value.get("usage")),
        ))
    })
    .map(str::to_owned)
    .or_else(|| {
        response?;
        hidden_params
            .and_then(|value| value.get("provider_specific_fields"))
            .and_then(|value| value.get("traffic_type"))
            .and_then(Value::as_str)
            .and_then(|value| map_traffic_type_to_service_tier(Some(value)))
            .map(str::to_owned)
    })
}

pub fn prepare_completion_input(
    request: CompletionInputRequest<'_>,
    catalog: &ModelInfoCatalog,
) -> Result<PreparedCompletionInput, CostError> {
    let response = request.model_selection.response;
    let call_type =
        infer_call_type(request.call_type, request.response_kind).unwrap_or("completion");
    let model = if matches!(
        call_type.parse::<CallTypes>(),
        Ok(CallTypes::image_generation | CallTypes::aimage_generation)
    ) && request.model_selection.model == Some("")
        && LlmProviders::AZURE.matches(request.model_selection.provider)
    {
        Some("dall-e-2")
    } else {
        request.model_selection.model
    };
    let selected_model = select_model_name_for_cost_calc(
        ModelSelectionRequest {
            model,
            ..request.model_selection
        },
        catalog,
    );
    let response_model = get_response_model(response).map(str::to_owned);
    let requested_model = model.map(str::to_owned);
    let usage = response.map(get_usage_object).transpose()?.flatten();
    let service_tier = select_service_tier(
        request.service_tier,
        request.optional_params,
        response,
        request.model_selection.hidden_params,
    );
    Ok(PreparedCompletionInput {
        call_type: call_type.to_owned(),
        model_candidates: [selected_model, response_model, requested_model],
        service_tier,
        usage,
    })
}
