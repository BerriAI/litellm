use jiff::Timestamp;
use serde_json::Value;

use crate::catalog::{CatalogCallError, CostCall, ModelCostRequest, ModelInfoCatalog};
use crate::completion_cost::{CompletionCost, completion_cost};
use crate::completion_input::{CompletionInputRequest, PreparedCompletionInput};
use crate::responses_usage::{ChatUsage, UsageError};

#[derive(Clone, Copy, Debug)]
pub struct CompletionResponseCostRequest<'a> {
    pub input: CompletionInputRequest<'a>,
    pub fallback_usage: Option<&'a ChatUsage>,
    pub provider: Option<&'a str>,
    pub region: Option<&'a str>,
    pub data_residency: Option<&'a str>,
    pub vertex_location: Option<&'a str>,
    pub at: Timestamp,
    pub response_time_ms: Option<f64>,
    pub prompt_characters: Option<f64>,
    pub completion_characters: Option<f64>,
    pub transcription_duration_seconds: Option<f64>,
    pub request_model: Option<&'a str>,
    pub deployment_info: Option<&'a Value>,
    pub built_in_tool_cost: f64,
    pub additional_costs: &'a [f64],
    pub discount_config: &'a Value,
    pub margin_config: &'a Value,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PricedCompletionResponse {
    pub model: String,
    pub prepared: PreparedCompletionInput,
    pub cost: CompletionCost,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompletionResponseCostError {
    Usage(UsageError),
    MissingUsage,
    MissingModel,
    UnsupportedCallType,
    Cost(CatalogCallError),
}

fn duration(request: CompletionResponseCostRequest<'_>) -> f64 {
    request.transcription_duration_seconds.unwrap_or_else(|| {
        request
            .input
            .model_selection
            .hidden_params
            .and_then(|hidden| hidden.get("audio_transcription_duration"))
            .or_else(|| {
                request
                    .input
                    .model_selection
                    .response
                    .and_then(|response| response.get("duration"))
            })
            .and_then(Value::as_f64)
            .unwrap_or(0.0)
    })
}

fn number_of_queries(optional_params: Option<&Value>) -> u64 {
    optional_params
        .and_then(|params| params.get("query"))
        .and_then(Value::as_array)
        .map_or(1, |queries| queries.len() as u64)
}

fn cost_call<'a>(
    call_type: &'a str,
    request: CompletionResponseCostRequest<'a>,
    empty_params: &'a Value,
) -> Result<CostCall<'a>, CompletionResponseCostError> {
    match call_type {
        "speech" | "aspeech" => Ok(CostCall::Speech {
            prompt_characters: request.prompt_characters,
        }),
        "transcription" | "atranscription" => Ok(CostCall::Transcription {
            duration_seconds: duration(request),
        }),
        "rerank" | "arerank" => Ok(CostCall::Rerank {
            billed_units: request
                .input
                .model_selection
                .response
                .and_then(|response| response.pointer("/meta/billed_units")),
        }),
        "vector_store_search" | "avector_store_search" => Ok(CostCall::VectorStoreSearch {
            api_type: request
                .input
                .optional_params
                .and_then(|params| params.get("api_type"))
                .and_then(Value::as_str),
        }),
        "search" | "asearch" => Ok(CostCall::Search {
            number_of_queries: Some(number_of_queries(request.input.optional_params)),
            optional_params: request.input.optional_params.unwrap_or(empty_params),
        }),
        "ocr" | "aocr" => Ok(CostCall::Ocr {
            response: request
                .input
                .model_selection
                .response
                .ok_or(CompletionResponseCostError::MissingUsage)?,
            deployment_info: request.deployment_info,
        }),
        "retrieve_batch" | "aretrieve_batch" => Ok(CostCall::Batch {
            deployment_info: request.deployment_info,
        }),
        "completion" | "acompletion" | "embedding" | "aembedding" | "text_completion"
        | "atext_completion" | "responses" | "aresponses" | "moderation" | "amoderation"
        | "generate_content" | "agenerate_content" => Ok(CostCall::Token {
            call_type,
            prompt_characters: request.prompt_characters,
            completion_characters: request.completion_characters,
            request_model: request.request_model,
        }),
        _ => Err(CompletionResponseCostError::UnsupportedCallType),
    }
}

pub fn completion_cost_from_response(
    catalog: &ModelInfoCatalog,
    request: CompletionResponseCostRequest<'_>,
) -> Result<PricedCompletionResponse, CompletionResponseCostError> {
    let prepared = catalog
        .prepare_completion_input(request.input)
        .map_err(CompletionResponseCostError::Usage)?;
    if matches!(
        prepared.call_type.as_str(),
        "completion"
            | "acompletion"
            | "embedding"
            | "aembedding"
            | "text_completion"
            | "atext_completion"
            | "responses"
            | "aresponses"
            | "moderation"
            | "amoderation"
            | "generate_content"
            | "agenerate_content"
            | "retrieve_batch"
            | "aretrieve_batch"
    ) && prepared.usage.is_none()
        && request.fallback_usage.is_none()
    {
        return Err(CompletionResponseCostError::MissingUsage);
    }
    let empty_usage = ChatUsage::default();
    let usage = prepared
        .usage
        .as_ref()
        .or(request.fallback_usage)
        .unwrap_or(&empty_usage);
    let empty_params = Value::Null;
    let call = cost_call(&prepared.call_type, request, &empty_params)?;
    let hidden_params = request.input.model_selection.hidden_params;
    let provider = hidden_params
        .and_then(|hidden| hidden.get("custom_llm_provider"))
        .and_then(Value::as_str)
        .or(request.provider);
    let explicit_pricing = request.input.model_selection.custom_pricing
        || request.input.model_selection.base_model.is_some();
    let region = if explicit_pricing {
        None
    } else {
        hidden_params
            .and_then(|hidden| hidden.get("region_name"))
            .and_then(Value::as_str)
            .or(request.region)
    };
    let attempts: Vec<_> = prepared.model_candidates.iter().flatten().collect();
    let prices: Vec<_> = attempts
        .iter()
        .map(|model| {
            let cost_request = ModelCostRequest {
                model,
                provider,
                region,
                usage,
                service_tier: prepared.service_tier.as_deref(),
                data_residency: request.data_residency,
                vertex_location: request.vertex_location,
                at: request.at,
                response_time_ms: request.response_time_ms,
            };
            (model, catalog.cost_per_token_for_call(cost_request, call))
        })
        .collect();
    let (model, priced) = prices
        .iter()
        .find(|(_, result)| result.is_ok())
        .or_else(|| prices.last())
        .ok_or(CompletionResponseCostError::MissingModel)?;
    let (prompt, output) = *priced
        .as_ref()
        .map_err(|error| CompletionResponseCostError::Cost(*error))?;
    let cost = completion_cost(
        prompt,
        output,
        request.built_in_tool_cost,
        request.additional_costs,
        provider,
        request.discount_config,
        request.margin_config,
    );
    Ok(PricedCompletionResponse {
        model: (*model).to_string(),
        prepared,
        cost,
    })
}
