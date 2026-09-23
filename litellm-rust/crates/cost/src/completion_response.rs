use std::collections::BTreeMap;

use jiff::Timestamp;
use litellm_token_counter::{CountableRequest, TokenCounter};
use serde_json::Value;

use crate::a2a_cost::{A2ACostError, calculate_a2a_cost};
use crate::azure_ai_cost::is_azure_model_router;
use crate::billed_token_rates::TokenTypeCostBreakdown;
use crate::catalog::{
    CatalogCallError, CatalogError, CatalogImageError, CostCall, ModelCostRequest, ModelInfoCatalog,
};
use crate::completion_cost::{
    CompletionCost, ResponseCostError, completion_cost, get_response_cost_from_hidden_params,
};
use crate::completion_input::{CompletionInputRequest, PreparedCompletionInput, ResponseKind};
use crate::custom_pricing::{CustomPricing, CustomPricingError, cost_from_chat_usage};
use crate::image_cost_router::{
    ImageCostRouteError, ImageCostRouteRequest, call_type_has_image_response,
    route_image_generation_cost_calculator,
};
use crate::mcp_cost::calculate_mcp_tool_call_cost;
use crate::per_second::{DEFAULT_REPLICATE_GPU_PRICE_PER_SECOND, get_replicate_completion_pricing};
use crate::realtime_cost::{
    collect_and_combine_usage_from_realtime_stream_results, combine_usage_objects, event_usage,
    partition_results_by_service_tier,
};
use crate::responses_usage::{ChatUsage, UsageError};
use crate::speech_cost::count_characters;
use crate::tool_call_cost_tracking::{DefaultToolRates, ResponseKind as ToolResponseKind};
use crate::tool_cost_dispatch::BuiltInToolCostRequest;

#[derive(Clone, Copy, Debug)]
pub struct BuiltInToolCostConfig<'a> {
    pub response_kind: ToolResponseKind,
    pub params: &'a Value,
    pub defaults: DefaultToolRates,
}

#[derive(Clone, Copy)]
pub struct CompletionTextInput<'a> {
    pub prompt: &'a str,
    pub messages: Option<&'a Value>,
    pub completion: &'a str,
    pub counter: &'a TokenCounter,
}

#[derive(Clone, Copy)]
pub struct CompletionResponseCostRequest<'a> {
    pub input: CompletionInputRequest<'a>,
    pub fallback_usage: Option<&'a ChatUsage>,
    pub text_input: Option<CompletionTextInput<'a>>,
    pub custom_cost: CustomPricing,
    pub replicate_rate_per_second: Option<f64>,
    pub provider: Option<&'a str>,
    pub region: Option<&'a str>,
    pub data_residency: Option<&'a str>,
    pub vertex_location: Option<&'a str>,
    pub at: Timestamp,
    pub response_time_ms: Option<f64>,
    pub prompt_characters: Option<f64>,
    pub speech_prompt: Option<&'a str>,
    pub completion_characters: Option<f64>,
    pub transcription_duration_seconds: Option<f64>,
    pub request_model: Option<&'a str>,
    pub deployment_info: Option<&'a Value>,
    pub image_quality: Option<&'a str>,
    pub image_size: Option<&'a str>,
    pub image_count: Option<u64>,
    pub built_in_tool_cost: f64,
    pub built_in_tool_config: Option<BuiltInToolCostConfig<'a>>,
    pub additional_costs: &'a [f64],
    pub discount_config: &'a Value,
    pub margin_config: &'a Value,
    pub logging_details: Option<&'a Value>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PricedCompletionResponse {
    pub model: String,
    pub prepared: PreparedCompletionInput,
    pub cost: CompletionCost,
    pub token_breakdown: Option<TokenTypeCostBreakdown>,
    pub named_additional_costs: BTreeMap<String, f64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompletionResponseCostError {
    Usage(UsageError),
    MissingUsage,
    MissingModel,
    MissingProvider,
    UnsupportedCallType,
    Cost(CatalogCallError),
    Image(ImageCostRouteError),
    Video(CatalogImageError),
    Realtime(CatalogError),
    A2A(A2ACostError),
    TokenCount,
    CustomPricing(CustomPricingError),
    ProviderCost(ResponseCostError),
}

impl From<UsageError> for CompletionResponseCostError {
    fn from(value: UsageError) -> Self {
        Self::Usage(value)
    }
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

fn price_with_custom(
    candidates: &[Option<String>; 3],
    usage: &ChatUsage,
    pricing: CustomPricing,
    response_time_ms: Option<f64>,
    catalog_price: impl Fn(&str) -> Result<(f64, f64), CompletionResponseCostError>,
) -> Result<(String, (f64, f64)), CompletionResponseCostError> {
    match cost_from_chat_usage(usage, pricing, response_time_ms)
        .map_err(CompletionResponseCostError::CustomPricing)?
    {
        Some(cost) => candidates
            .iter()
            .flatten()
            .next()
            .cloned()
            .map(|model| (model, (cost.input, cost.output)))
            .ok_or(CompletionResponseCostError::MissingModel),
        None => price_candidates(candidates, catalog_price).map_err(|error| match error {
            CandidatePriceError::MissingModel => CompletionResponseCostError::MissingModel,
            CandidatePriceError::Price(error) => error,
        }),
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
        token_breakdown: None,
        named_additional_costs: BTreeMap::new(),
    }
}

fn number(value: Option<&Value>) -> Option<f64> {
    match value? {
        Value::Number(value) => value.as_f64(),
        Value::String(value) => value.parse().ok(),
        _ => None,
    }
}

fn count_text_usage(
    input: CompletionTextInput<'_>,
) -> Result<ChatUsage, CompletionResponseCostError> {
    let prompt = match input.messages {
        Some(Value::Array(messages)) if !messages.is_empty() => {
            let body = serde_json::to_vec(&serde_json::json!({"messages": messages}))
                .map_err(|_| CompletionResponseCostError::TokenCount)?;
            let request = CountableRequest::parse(&body)
                .map_err(|_| CompletionResponseCostError::TokenCount)?;
            input
                .counter
                .count_request(&request)
                .map_err(|_| CompletionResponseCostError::TokenCount)?
                .input_tokens
        }
        Some(Value::Array(_)) | None => input
            .counter
            .count_text(input.prompt)
            .map_err(|_| CompletionResponseCostError::TokenCount)?,
        Some(_) => return Err(CompletionResponseCostError::TokenCount),
    };
    let completion = input
        .counter
        .count_text(input.completion)
        .map_err(|_| CompletionResponseCostError::TokenCount)?;
    let prompt_tokens = u64::try_from(prompt)
        .map_err(|_| CompletionResponseCostError::Usage(UsageError::TokenCountOverflow))?;
    let completion_tokens = u64::try_from(completion)
        .map_err(|_| CompletionResponseCostError::Usage(UsageError::TokenCountOverflow))?;
    let total_tokens =
        prompt_tokens
            .checked_add(completion_tokens)
            .ok_or(CompletionResponseCostError::Usage(
                UsageError::TokenCountOverflow,
            ))?;
    Ok(ChatUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens,
        ..ChatUsage::default()
    })
}

pub fn response_time_ms_for_cost(response: Option<&Value>, fallback: Option<f64>) -> Option<f64> {
    response
        .and_then(|response| response.get("_response_ms"))
        .and_then(Value::as_f64)
        .or(fallback)
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
    request_model: Option<&'a str>,
    empty_params: &'a Value,
) -> Result<CostCall<'a>, CompletionResponseCostError> {
    match call_type {
        "speech" | "aspeech" => Ok(CostCall::Speech {
            prompt_characters: request.prompt_characters.or_else(|| {
                request
                    .speech_prompt
                    .map(|prompt| count_characters(prompt) as f64)
            }),
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
                request_model,
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

fn unregistered_replicate_cost(
    catalog: &ModelInfoCatalog,
    request: CompletionResponseCostRequest<'_>,
    prepared: &PreparedCompletionInput,
    provider: Option<&str>,
) -> Option<(String, f64)> {
    let model = prepared.model_candidates.iter().flatten().next()?;
    if !(provider == Some("replicate") || model.contains("replicate"))
        || catalog.contains_exact_model(model)
    {
        return None;
    }
    let total_time_ms = response_time_ms_for_cost(
        request.input.model_selection.response,
        request.response_time_ms,
    )
    .unwrap_or(0.0);
    let response = request.input.model_selection.response;
    let now_seconds = request.at.as_nanosecond() as f64 / 1_000_000_000.0;
    let total = get_replicate_completion_pricing(
        total_time_ms,
        number(response.and_then(|response| response.get("created"))),
        number(response.and_then(|response| response.get("ended"))),
        now_seconds,
        request
            .replicate_rate_per_second
            .unwrap_or(DEFAULT_REPLICATE_GPU_PRICE_PER_SECOND),
    );
    Some((model.clone(), total))
}

fn realtime_results(
    request: CompletionResponseCostRequest<'_>,
) -> Result<&[Value], CompletionResponseCostError> {
    request
        .input
        .model_selection
        .response
        .and_then(|response| response.get("results"))
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or(CompletionResponseCostError::MissingUsage)
}

fn price_realtime_response(
    catalog: &ModelInfoCatalog,
    request: CompletionResponseCostRequest<'_>,
    prepared: PreparedCompletionInput,
    provider: Option<&str>,
) -> Result<PricedCompletionResponse, CompletionResponseCostError> {
    let provider = provider.ok_or(CompletionResponseCostError::MissingProvider)?;
    let results = realtime_results(request)?;
    let combined = match prepared.usage.as_ref().or(request.fallback_usage) {
        Some(usage) => usage.clone(),
        None => collect_and_combine_usage_from_realtime_stream_results(results)
            .map_err(CompletionResponseCostError::Usage)?,
    };
    let model = prepared
        .model_candidates
        .iter()
        .flatten()
        .next()
        .ok_or(CompletionResponseCostError::MissingModel)?
        .clone();
    let total = catalog.handle_realtime_stream_cost_calculation(
        results,
        &combined,
        provider,
        &model,
        request.data_residency,
        request.at,
    );
    Ok(flat_priced(prepared, model, total))
}

fn add_completion_costs(first: CompletionCost, second: CompletionCost) -> CompletionCost {
    CompletionCost {
        input: first.input + second.input,
        output: first.output + second.output,
        built_in_tools: first.built_in_tools + second.built_in_tools,
        additional: first.additional + second.additional,
        original: first.original + second.original,
        discounted: first.discounted + second.discounted,
        total: first.total + second.total,
        discount_percent: second.discount_percent,
        discount_amount: first.discount_amount + second.discount_amount,
        margin_percent: second.margin_percent,
        margin_fixed_amount: first.margin_fixed_amount + second.margin_fixed_amount,
        margin_total_amount: first.margin_total_amount + second.margin_total_amount,
    }
}

fn built_in_tool_cost(
    catalog: &ModelInfoCatalog,
    request: CompletionResponseCostRequest<'_>,
    model: &str,
    provider: Option<&str>,
    region: Option<&str>,
    usage: Option<&ChatUsage>,
) -> f64 {
    let Some(config) = request.built_in_tool_config else {
        return request.built_in_tool_cost;
    };
    let Some(response) = request.input.model_selection.response else {
        return request.built_in_tool_cost;
    };
    request.built_in_tool_cost
        + catalog.built_in_tool_cost(
            model,
            provider,
            region,
            BuiltInToolCostRequest {
                response,
                response_kind: config.response_kind,
                usage,
                provider,
                params: config.params,
                defaults: config.defaults,
            },
        )
}

fn price_responses_websocket(
    catalog: &ModelInfoCatalog,
    request: CompletionResponseCostRequest<'_>,
    prepared: PreparedCompletionInput,
    provider: Option<&str>,
    region: Option<&str>,
) -> Result<PricedCompletionResponse, CompletionResponseCostError> {
    let results = realtime_results(request)?;
    let partition = partition_results_by_service_tier(results);
    let groups = if partition.is_empty() {
        vec![(prepared.service_tier.as_deref(), Vec::new())]
    } else {
        partition
    };
    let priced_groups: Vec<_> = groups
        .into_iter()
        .enumerate()
        .map(|(index, (tier, events))| {
            let usage = combine_usage_objects(
                events
                    .into_iter()
                    .map(event_usage)
                    .collect::<Result<Vec<_>, _>>()?,
            )?;
            let (model, (prompt, output)) = price_with_custom(
                &prepared.model_candidates,
                &usage,
                request.custom_cost,
                response_time_ms_for_cost(
                    request.input.model_selection.response,
                    request.response_time_ms,
                ),
                |model| {
                    catalog
                        .cost_per_token(ModelCostRequest {
                            model,
                            provider,
                            region,
                            usage: &usage,
                            service_tier: tier,
                            data_residency: request.data_residency,
                            vertex_location: request.vertex_location,
                            at: request.at,
                            response_time_ms: None,
                        })
                        .map_err(CompletionResponseCostError::Realtime)
                },
            )?;
            let built_in = if index == 0 {
                built_in_tool_cost(catalog, request, &model, provider, region, Some(&usage))
            } else {
                0.0
            };
            let additional = if index == 0 {
                request.additional_costs
            } else {
                &[]
            };
            Ok::<_, CompletionResponseCostError>((
                model,
                completion_cost(
                    prompt,
                    output,
                    built_in,
                    additional,
                    provider,
                    request.discount_config,
                    request.margin_config,
                ),
            ))
        })
        .collect::<Result<_, _>>()?;
    let model = priced_groups
        .first()
        .map(|(model, _)| model.clone())
        .or_else(|| prepared.model_candidates.iter().flatten().next().cloned())
        .ok_or(CompletionResponseCostError::MissingModel)?;
    let zero = completion_cost(0.0, 0.0, 0.0, &[], None, &Value::Null, &Value::Null);
    let cost = priced_groups
        .into_iter()
        .map(|(_, cost)| cost)
        .fold(zero, add_completion_costs);
    Ok(PricedCompletionResponse {
        model,
        prepared,
        cost,
        token_breakdown: None,
        named_additional_costs: BTreeMap::new(),
    })
}

pub fn completion_cost_from_response(
    catalog: &ModelInfoCatalog,
    request: CompletionResponseCostRequest<'_>,
) -> Result<PricedCompletionResponse, CompletionResponseCostError> {
    let prepared = catalog
        .prepare_completion_input(request.input)
        .map_err(CompletionResponseCostError::Usage)?;
    let hidden_params = request.input.model_selection.hidden_params;
    let explicit_provider = hidden_params
        .and_then(|hidden| hidden.get("custom_llm_provider"))
        .and_then(Value::as_str)
        .or(request.provider)
        .or(request.input.model_selection.provider);
    let inferred_provider = explicit_provider.map(str::to_owned).or_else(|| {
        prepared
            .model_candidates
            .iter()
            .flatten()
            .find_map(|model| {
                catalog.get_provider_for_cost_calc(
                    Some(model),
                    None,
                    request.input.model_selection.known_providers,
                )
            })
    });
    let provider = inferred_provider.as_deref();
    let deployment_info = request
        .input
        .model_selection
        .custom_pricing
        .then_some(request.deployment_info)
        .flatten();
    if matches!(
        prepared.call_type.as_str(),
        "send_message" | "asend_message"
    ) {
        let model = prepared
            .model_candidates
            .iter()
            .flatten()
            .next()
            .cloned()
            .unwrap_or_default();
        let total = calculate_a2a_cost(request.logging_details)
            .map_err(CompletionResponseCostError::A2A)?;
        return Ok(flat_priced(prepared, model, total));
    }
    if prepared.call_type == "call_mcp_tool" {
        let model = prepared
            .model_candidates
            .iter()
            .flatten()
            .next()
            .cloned()
            .ok_or(CompletionResponseCostError::MissingModel)?;
        let total = calculate_mcp_tool_call_cost(request.logging_details);
        return Ok(flat_priced(prepared, model, total));
    }
    if call_type_has_image_response(&prepared.call_type)
        && request.input.response_kind == Some(ResponseKind::ImageGeneration)
    {
        return price_image_response(catalog, request, prepared, provider, deployment_info);
    }
    if is_video_call(&prepared.call_type) {
        return price_video_response(catalog, request, prepared, provider, deployment_info);
    }
    if prepared.call_type == "_arealtime" {
        return price_realtime_response(catalog, request, prepared, provider);
    }
    if prepared.call_type == "_aresponses_websocket" {
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
        return price_responses_websocket(catalog, request, prepared, provider, region);
    }
    if !matches!(prepared.call_type.as_str(), "search" | "asearch")
        && let Some((model, total)) =
            unregistered_replicate_cost(catalog, request, &prepared, provider)
    {
        return Ok(flat_priced(prepared, model, total));
    }
    let needs_token_usage = matches!(
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
    );
    let counted_usage = if needs_token_usage
        && request.input.model_selection.response.is_none()
        && prepared.usage.is_none()
        && request.fallback_usage.is_none()
    {
        request.text_input.map(count_text_usage).transpose()?
    } else {
        None
    };
    let empty_usage = ChatUsage::default();
    let usage = prepared
        .usage
        .as_ref()
        .or(request.fallback_usage)
        .or(counted_usage.as_ref())
        .unwrap_or(&empty_usage);
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
    let is_search = matches!(prepared.call_type.as_str(), "search" | "asearch");
    let custom_pricing = if is_search {
        CustomPricing::NONE
    } else {
        request.custom_cost
    };
    let hidden_model = hidden_params.and_then(|hidden| {
        hidden
            .get("model")
            .and_then(Value::as_str)
            .filter(|model| !model.is_empty())
            .or_else(|| hidden.get("litellm_model_name").and_then(Value::as_str))
    });
    let request_model = if provider == Some("azure_ai") {
        hidden_model
            .filter(|model| is_azure_model_router(model))
            .or(request.request_model)
    } else {
        request.request_model
    };
    let empty_params = Value::Null;
    let (model, (prompt, output)) = price_with_custom(
        &prepared.model_candidates,
        usage,
        custom_pricing,
        response_time_ms_for_cost(
            request.input.model_selection.response,
            request.response_time_ms,
        ),
        |model| {
            let token_request_model =
                if provider == Some("azure_ai") && !is_azure_model_router(model) {
                    None
                } else {
                    request_model
                };
            let call = cost_call(
                &prepared.call_type,
                request,
                token_request_model,
                &empty_params,
            )?;
            let cost_request = ModelCostRequest {
                model,
                provider,
                region,
                usage,
                service_tier: prepared.service_tier.as_deref(),
                data_residency: request.data_residency,
                vertex_location: request.vertex_location,
                at: request.at,
                response_time_ms: response_time_ms_for_cost(
                    request.input.model_selection.response,
                    request.response_time_ms,
                ),
            };
            catalog
                .cost_per_token_for_call(cost_request, call)
                .map_err(CompletionResponseCostError::Cost)
        },
    )?;
    let router_fee = (!is_search && provider == Some("azure_ai") && !is_azure_model_router(&model))
        .then(|| catalog.azure_ai_router_fee(&model, request_model, usage.prompt_tokens))
        .and_then(Result::ok)
        .flatten();
    let supplied_additional_costs = if is_search {
        &[]
    } else {
        request.additional_costs
    };
    let additional_costs = supplied_additional_costs
        .iter()
        .copied()
        .chain(router_fee)
        .collect::<Vec<_>>();
    let cost = completion_cost(
        prompt,
        output,
        if is_search {
            0.0
        } else {
            built_in_tool_cost(catalog, request, &model, provider, region, Some(usage))
        },
        &additional_costs,
        provider,
        request.discount_config,
        request.margin_config,
    );
    let token_breakdown = prepared.usage.as_ref().map(|response_usage| {
        catalog.get_token_type_cost_breakdown(
            ModelCostRequest {
                model: &model,
                provider,
                region,
                usage: response_usage,
                service_tier: prepared.service_tier.as_deref(),
                data_residency: request.data_residency,
                vertex_location: request.vertex_location,
                at: request.at,
                response_time_ms: None,
            },
            request.custom_cost.token,
        )
    });
    Ok(PricedCompletionResponse {
        model,
        prepared,
        cost,
        token_breakdown,
        named_additional_costs: router_fee
            .map(|fee| BTreeMap::from([("Azure Model Router Flat Cost".to_owned(), fee)]))
            .unwrap_or_default(),
    })
}

pub fn response_cost_calculator_from_response(
    catalog: &ModelInfoCatalog,
    request: CompletionResponseCostRequest<'_>,
    cache_hit: bool,
) -> Result<f64, CompletionResponseCostError> {
    if cache_hit {
        return Ok(0.0);
    }
    if request.input.model_selection.response.is_some()
        && let Some(hidden) = request.input.model_selection.hidden_params
        && let Some(cost) = get_response_cost_from_hidden_params(hidden)
            .map_err(CompletionResponseCostError::ProviderCost)?
    {
        return Ok(cost);
    }
    completion_cost_from_response(catalog, request).map(|priced| priced.cost.total)
}
