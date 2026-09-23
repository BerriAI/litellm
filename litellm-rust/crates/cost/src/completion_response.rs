use jiff::Timestamp;
use serde_json::Value;

use crate::catalog::{
    CatalogCallError, CatalogImageError, CostCall, ModelCostRequest, ModelInfoCatalog,
};
use crate::completion_cost::{CompletionCost, completion_cost};
use crate::completion_input::{CompletionInputRequest, PreparedCompletionInput, ResponseKind};
use crate::image_cost_router::{
    ImageCostRouteError, ImageCostRouteRequest, call_type_has_image_response,
    route_image_generation_cost_calculator,
};
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
    pub image_quality: Option<&'a str>,
    pub image_size: Option<&'a str>,
    pub image_count: Option<u64>,
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
    Image(ImageCostRouteError),
    Video(CatalogImageError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CandidatePriceError<E> {
    MissingModel,
    Price(E),
}

fn price_candidates<T: Copy, E: Copy>(
    candidates: &[Option<String>; 3],
    price: impl Fn(&str) -> Result<T, E>,
) -> Result<(String, T), CandidatePriceError<E>> {
    let prices: Vec<_> = candidates
        .iter()
        .flatten()
        .map(|model| (model, price(model)))
        .collect();
    let (model, priced) = prices
        .iter()
        .find(|(_, result)| result.is_ok())
        .or_else(|| prices.last())
        .ok_or(CandidatePriceError::MissingModel)?;
    match priced {
        Ok(cost) => Ok(((*model).clone(), *cost)),
        Err(error) => Err(CandidatePriceError::Price(*error)),
    }
}

fn flat_priced(
    prepared: PreparedCompletionInput,
    model: String,
    total: f64,
) -> PricedCompletionResponse {
    PricedCompletionResponse {
        model,
        prepared,
        cost: completion_cost(total, 0.0, 0.0, &[], None, &Value::Null, &Value::Null),
    }
}

fn number(value: Option<&Value>) -> Option<f64> {
    match value? {
        Value::Number(value) => value.as_f64(),
        Value::String(value) => value.parse().ok(),
        _ => None,
    }
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
            deployment_info: request
                .input
                .model_selection
                .custom_pricing
                .then_some(request.deployment_info)
                .flatten(),
        }),
        "retrieve_batch" | "aretrieve_batch" => Ok(CostCall::Batch {
            deployment_info: request
                .input
                .model_selection
                .custom_pricing
                .then_some(request.deployment_info)
                .flatten(),
        }),
        "completion" | "acompletion" | "embedding" | "aembedding" | "text_completion"
        | "atext_completion" | "responses" | "aresponses" | "moderation" | "amoderation"
        | "generate_content" | "agenerate_content" | "video_retrieve" | "avideo_retrieve" => {
            Ok(CostCall::Token {
                call_type,
                prompt_characters: request.prompt_characters,
                completion_characters: request.completion_characters,
                request_model: request.request_model,
            })
        }
        _ => Err(CompletionResponseCostError::UnsupportedCallType),
    }
}

fn price_image_response(
    catalog: &ModelInfoCatalog,
    request: CompletionResponseCostRequest<'_>,
    prepared: PreparedCompletionInput,
    provider: Option<&str>,
    deployment_info: Option<&Value>,
) -> Result<PricedCompletionResponse, CompletionResponseCostError> {
    let response = request
        .input
        .model_selection
        .response
        .ok_or(CompletionResponseCostError::MissingUsage)?;
    let empty_params = Value::Null;
    let (model, total) = price_candidates(&prepared.model_candidates, |model| {
        route_image_generation_cost_calculator(
            catalog,
            ImageCostRouteRequest {
                model,
                provider,
                image_response: response,
                call_type: Some(&prepared.call_type),
                quality: request.image_quality,
                size: request.image_size,
                n: request.image_count,
                optional_params: request.input.optional_params.unwrap_or(&empty_params),
                supplied_model_info: deployment_info,
                at: request.at,
            },
        )
    })
    .map_err(|error| match error {
        CandidatePriceError::MissingModel => CompletionResponseCostError::MissingModel,
        CandidatePriceError::Price(error) => CompletionResponseCostError::Image(error),
    })?;
    Ok(flat_priced(prepared, model, total))
}

fn is_video_call(call_type: &str) -> bool {
    matches!(
        call_type,
        "create_video"
            | "acreate_video"
            | "video_edit"
            | "avideo_edit"
            | "video_remix"
            | "avideo_remix"
    )
}

fn price_video_response(
    catalog: &ModelInfoCatalog,
    request: CompletionResponseCostRequest<'_>,
    prepared: PreparedCompletionInput,
    provider: Option<&str>,
    deployment_info: Option<&Value>,
) -> Result<PricedCompletionResponse, CompletionResponseCostError> {
    let usage = request
        .input
        .model_selection
        .response
        .and_then(|response| response.get("usage"));
    let reported = number(usage.and_then(|usage| usage.get("provider_reported_cost_usd")));
    if deployment_info.is_none()
        && let Some(total) = reported
    {
        let model = prepared
            .model_candidates
            .iter()
            .flatten()
            .next()
            .ok_or(CompletionResponseCostError::MissingModel)?
            .clone();
        return Ok(flat_priced(prepared, model, total));
    }
    let duration = number(usage.and_then(|usage| usage.get("duration_seconds"))).unwrap_or(0.0);
    let count = usage
        .and_then(|usage| usage.get("video_count"))
        .and_then(Value::as_u64)
        .filter(|count| *count > 1)
        .unwrap_or(1);
    let resolution = usage
        .and_then(|usage| usage.get("video_resolution"))
        .and_then(Value::as_str)
        .map(|value| value.trim().to_ascii_lowercase());
    let (model, total) = price_candidates(&prepared.model_candidates, |model| {
        catalog
            .video_generation_cost(
                model,
                provider,
                deployment_info,
                duration,
                resolution.as_deref(),
            )
            .map(|cost| cost * count as f64)
    })
    .map_err(|error| match error {
        CandidatePriceError::MissingModel => CompletionResponseCostError::MissingModel,
        CandidatePriceError::Price(error) => CompletionResponseCostError::Video(error),
    })?;
    Ok(flat_priced(prepared, model, total))
}

pub fn completion_cost_from_response(
    catalog: &ModelInfoCatalog,
    request: CompletionResponseCostRequest<'_>,
) -> Result<PricedCompletionResponse, CompletionResponseCostError> {
    let prepared = catalog
        .prepare_completion_input(request.input)
        .map_err(CompletionResponseCostError::Usage)?;
    let hidden_params = request.input.model_selection.hidden_params;
    let provider = hidden_params
        .and_then(|hidden| hidden.get("custom_llm_provider"))
        .and_then(Value::as_str)
        .or(request.provider);
    let deployment_info = request
        .input
        .model_selection
        .custom_pricing
        .then_some(request.deployment_info)
        .flatten();
    if call_type_has_image_response(&prepared.call_type)
        && request.input.response_kind == Some(ResponseKind::ImageGeneration)
    {
        return price_image_response(catalog, request, prepared, provider, deployment_info);
    }
    if is_video_call(&prepared.call_type) {
        return price_video_response(catalog, request, prepared, provider, deployment_info);
    }
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
    let (model, (prompt, output)) = price_candidates(&prepared.model_candidates, |model| {
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
        catalog.cost_per_token_for_call(cost_request, call)
    })
    .map_err(|error| match error {
        CandidatePriceError::MissingModel => CompletionResponseCostError::MissingModel,
        CandidatePriceError::Price(error) => CompletionResponseCostError::Cost(error),
    })?;
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
        model,
        prepared,
        cost,
    })
}
