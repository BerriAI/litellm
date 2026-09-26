use std::collections::BTreeMap;

use jiff::Timestamp;
use litellm_token_counter::{CountableRequest, TokenCounter};
use serde_json::Value;

use crate::a2a_cost::calculate_a2a_cost;
use crate::azure_ai_cost::{azure_ai_router_fee, is_azure_model_router};
use crate::billed_token_rates::{TokenTypeCostBreakdown, get_token_type_cost_breakdown};
use crate::call_type::CallTypes;
use crate::catalog::{CostCall, ModelCostRequest, ModelInfoCatalog};
use crate::completion_cost::{
    CompletionCost, completion_cost, get_response_cost_from_hidden_params,
};
use crate::completion_input::{
    CompletionInputRequest, PreparedCompletionInput, ResponseKind, normalize_service_tier,
};
use crate::cost_calculator::{
    cost_per_token, cost_per_token_for_call, default_video_cost_calculator,
    handle_realtime_stream_cost_calculation,
};
use crate::custom_pricing::{CustomPricing, cost_from_chat_usage};
use crate::error::CostError;
use crate::image_cost_router::{
    ImageCostRouteRequest, call_type_has_image_response, route_image_generation_cost_calculator,
};
use crate::mcp_cost::calculate_mcp_tool_call_cost;
use crate::model_selection::get_provider_for_cost_calc;
use crate::per_second::{DEFAULT_REPLICATE_GPU_PRICE_PER_SECOND, get_replicate_completion_pricing};
use crate::provider::LlmProviders;
use crate::realtime_cost::{
    collect_and_combine_usage_from_realtime_stream_results, combine_usage_objects, event_usage,
    partition_results_by_service_tier,
};
use crate::responses_usage::ChatUsage;
use crate::search_cost::search_provider_cost_per_query;
use crate::speech_cost::count_characters;
use crate::together_cost::together_pricing_model;
use crate::tool_call_cost_tracking::{DefaultToolRates, ResponseKind as ToolResponseKind};
use crate::tool_cost_dispatch::{BuiltInToolCostRequest, get_cost_for_built_in_tools};
use crate::wire::py_float;

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

fn without_provider_stated_cost(usage: &ChatUsage) -> ChatUsage {
    ChatUsage {
        cost: None,
        ..usage.clone()
    }
}

fn price_with_custom(
    candidates: &[Option<String>; 3],
    usage: &ChatUsage,
    pricing: CustomPricing,
    response_time_ms: Option<f64>,
    catalog_price: impl Fn(&str) -> Result<(f64, f64), CostError>,
) -> Result<(String, (f64, f64)), CostError> {
    match cost_from_chat_usage(usage, pricing, response_time_ms)? {
        Some(cost) => candidates
            .iter()
            .flatten()
            .next()
            .cloned()
            .map(|model| (model, (cost.input, cost.output)))
            .ok_or(CostError::MissingModel),
        None => price_candidates(candidates, catalog_price).map_err(|error| match error {
            CandidatePriceError::MissingModel => CostError::MissingModel,
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
    value.and_then(py_float)
}

fn count_text_usage(input: CompletionTextInput<'_>) -> Result<ChatUsage, CostError> {
    let prompt = match input.messages {
        Some(Value::Array(messages)) if !messages.is_empty() => {
            let body = serde_json::to_vec(&serde_json::json!({"messages": messages}))
                .map_err(|_| CostError::TokenCount)?;
            let request = CountableRequest::parse(&body).map_err(|_| CostError::TokenCount)?;
            input
                .counter
                .count_request(&request)
                .map_err(|_| CostError::TokenCount)?
                .input_tokens
        }
        Some(Value::Array(_)) | None => input
            .counter
            .count_text(input.prompt)
            .map_err(|_| CostError::TokenCount)?,
        Some(_) => return Err(CostError::TokenCount),
    };
    let completion = input
        .counter
        .count_text(input.completion)
        .map_err(|_| CostError::TokenCount)?;
    let prompt_tokens = u64::try_from(prompt).map_err(|_| CostError::TokenCountOverflow)?;
    let completion_tokens = u64::try_from(completion).map_err(|_| CostError::TokenCountOverflow)?;
    let total_tokens = prompt_tokens
        .checked_add(completion_tokens)
        .ok_or(CostError::TokenCountOverflow)?;
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
    model: &'a str,
    request_model: Option<&'a str>,
    empty_params: &'a Value,
) -> Result<CostCall<'a>, CostError> {
    let deployment_info = || {
        request
            .input
            .model_selection
            .custom_pricing
            .then_some(request.deployment_info)
            .flatten()
    };
    match call_type.parse::<CallTypes>().ok() {
        Some(CallTypes::speech | CallTypes::aspeech) => Ok(CostCall::Speech {
            prompt_characters: request.prompt_characters.or_else(|| {
                request
                    .speech_prompt
                    .map(|prompt| count_characters(prompt) as f64)
            }),
        }),
        Some(CallTypes::transcription | CallTypes::atranscription) => Ok(CostCall::Transcription {
            duration_seconds: duration(request),
        }),
        Some(CallTypes::rerank | CallTypes::arerank) => Ok(CostCall::Rerank {
            billed_units: request
                .input
                .model_selection
                .response
                .and_then(|response| response.pointer("/meta/billed_units")),
        }),
        Some(CallTypes::vector_store_search | CallTypes::avector_store_search) => {
            Ok(CostCall::VectorStoreSearch {
                api_type: model
                    .split_once('/')
                    .and_then(|(prefix, api_type)| (prefix == "vertex_ai").then_some(api_type)),
            })
        }
        Some(CallTypes::search | CallTypes::asearch) => Ok(CostCall::Search {
            number_of_queries: Some(number_of_queries(request.input.optional_params)),
            optional_params: request.input.optional_params.unwrap_or(empty_params),
        }),
        Some(CallTypes::ocr | CallTypes::aocr) => Ok(CostCall::Ocr {
            response: request
                .input
                .model_selection
                .response
                .ok_or(CostError::MissingUsage)?,
            deployment_info: deployment_info(),
        }),
        Some(CallTypes::retrieve_batch | CallTypes::aretrieve_batch) => Ok(CostCall::Batch {
            deployment_info: deployment_info(),
        }),
        _ => Ok(CostCall::Token {
            call_type,
            prompt_characters: request.prompt_characters,
            completion_characters: request.completion_characters,
            request_model,
        }),
    }
}

fn price_image_response(
    catalog: &ModelInfoCatalog,
    request: CompletionResponseCostRequest<'_>,
    prepared: PreparedCompletionInput,
    provider: Option<&str>,
    deployment_info: Option<&Value>,
) -> Result<PricedCompletionResponse, CostError> {
    let response = request
        .input
        .model_selection
        .response
        .ok_or(CostError::MissingUsage)?;
    let empty_params = Value::Null;
    let (model, total) = price_candidates(&prepared.model_candidates, |model| {
        route_image_generation_cost_calculator(
            catalog,
            ImageCostRouteRequest {
                model,
                provider,
                image_response: response,
                call_type: prepared.call_type.parse::<CallTypes>().ok(),
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
        CandidatePriceError::MissingModel => CostError::MissingModel,
        CandidatePriceError::Price(error) => error,
    })?;
    Ok(flat_priced(prepared, model, total))
}

fn is_video_call(call_type: Option<CallTypes>) -> bool {
    matches!(
        call_type,
        Some(
            CallTypes::create_video
                | CallTypes::acreate_video
                | CallTypes::video_edit
                | CallTypes::avideo_edit
                | CallTypes::video_remix
                | CallTypes::avideo_remix
        )
    )
}

fn price_video_response(
    catalog: &ModelInfoCatalog,
    request: CompletionResponseCostRequest<'_>,
    prepared: PreparedCompletionInput,
    provider: Option<&str>,
    deployment_info: Option<&Value>,
) -> Result<PricedCompletionResponse, CostError> {
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
            .ok_or(CostError::MissingModel)?
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
        default_video_cost_calculator(
            catalog,
            model,
            duration,
            provider,
            deployment_info,
            resolution.as_deref(),
        )
        .map(|cost| cost * count as f64)
    })
    .map_err(|error| match error {
        CandidatePriceError::MissingModel => CostError::MissingModel,
        CandidatePriceError::Price(error) => error,
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
    if !(LlmProviders::REPLICATE.matches(provider) || model.contains("replicate"))
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

fn realtime_results(request: CompletionResponseCostRequest<'_>) -> Result<&[Value], CostError> {
    request
        .input
        .model_selection
        .response
        .and_then(|response| response.get("results"))
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or(CostError::MissingUsage)
}

fn price_realtime_response(
    catalog: &ModelInfoCatalog,
    request: CompletionResponseCostRequest<'_>,
    prepared: PreparedCompletionInput,
    provider: Option<&str>,
) -> Result<PricedCompletionResponse, CostError> {
    let provider = provider.ok_or(CostError::MissingProvider)?;
    let results = realtime_results(request)?;
    let combined = match prepared.usage.as_ref().or(request.fallback_usage) {
        Some(usage) => usage.clone(),
        None => collect_and_combine_usage_from_realtime_stream_results(results)?,
    };
    let model = prepared
        .model_candidates
        .iter()
        .flatten()
        .next()
        .ok_or(CostError::MissingModel)?
        .clone();
    let total = handle_realtime_stream_cost_calculation(
        catalog,
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

fn config_driven_tool_cost(
    catalog: &ModelInfoCatalog,
    request: CompletionResponseCostRequest<'_>,
    model: &str,
    provider: Option<&str>,
    region: Option<&str>,
    usage: Option<&ChatUsage>,
) -> f64 {
    let Some(config) = request.built_in_tool_config else {
        return 0.0;
    };
    let Some(response) = request.input.model_selection.response else {
        return 0.0;
    };
    get_cost_for_built_in_tools(
        catalog,
        model,
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
) -> Result<PricedCompletionResponse, CostError> {
    let results = realtime_results(request)?;
    let requested_tier = normalize_service_tier(request.input.service_tier.or_else(|| {
        request
            .input
            .optional_params
            .and_then(|params| params.get("service_tier"))
    }));
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
            let stripped_usage = (request.custom_cost != CustomPricing::NONE
                || request.input.model_selection.custom_pricing)
                .then(|| without_provider_stated_cost(&usage));
            let usage = stripped_usage.as_ref().unwrap_or(&usage);
            let (model, (prompt, output)) = price_with_custom(
                &prepared.model_candidates,
                usage,
                request.custom_cost,
                response_time_ms_for_cost(
                    request.input.model_selection.response,
                    request.response_time_ms,
                ),
                |model| {
                    cost_per_token(
                        catalog,
                        ModelCostRequest {
                            model,
                            provider,
                            region,
                            usage,
                            service_tier: requested_tier
                                .or_else(|| tier.filter(|tier| !tier.eq_ignore_ascii_case("auto"))),
                            data_residency: request.data_residency,
                            vertex_location: request.vertex_location,
                            at: request.at,
                            response_time_ms: None,
                        },
                    )
                },
            )?;
            let built_in = if index == 0 {
                request.built_in_tool_cost
            } else {
                0.0
            } + config_driven_tool_cost(
                catalog,
                request,
                &model,
                provider,
                region,
                Some(usage),
            );
            let additional = if index == 0 {
                request.additional_costs
            } else {
                &[]
            };
            Ok::<_, CostError>((
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
        .ok_or(CostError::MissingModel)?;
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
) -> Result<PricedCompletionResponse, CostError> {
    let prepared = catalog.prepare_completion_input(request.input)?;
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
            .find_map(|model| get_provider_for_cost_calc(Some(model), None, catalog))
    });
    let provider = inferred_provider.as_deref();
    let deployment_info = request
        .input
        .model_selection
        .custom_pricing
        .then_some(request.deployment_info)
        .flatten();
    let call_type = prepared.call_type.parse::<CallTypes>().ok();
    if matches!(
        call_type,
        Some(CallTypes::send_message | CallTypes::asend_message)
    ) {
        let model = prepared
            .model_candidates
            .iter()
            .flatten()
            .next()
            .cloned()
            .unwrap_or_default();
        let total = calculate_a2a_cost(request.logging_details)?;
        return Ok(flat_priced(prepared, model, total));
    }
    if call_type == Some(CallTypes::call_mcp_tool) {
        let model = prepared
            .model_candidates
            .iter()
            .flatten()
            .next()
            .cloned()
            .ok_or(CostError::MissingModel)?;
        let total = calculate_mcp_tool_call_cost(request.logging_details);
        return Ok(flat_priced(prepared, model, total));
    }
    if call_type_has_image_response(&prepared.call_type)
        && request.input.response_kind == Some(ResponseKind::ImageGeneration)
    {
        return price_image_response(catalog, request, prepared, provider, deployment_info);
    }
    if is_video_call(call_type) {
        return price_video_response(catalog, request, prepared, provider, deployment_info);
    }
    if call_type == Some(CallTypes::arealtime) {
        return price_realtime_response(catalog, request, prepared, provider);
    }
    if call_type == Some(CallTypes::aresponses_websocket) {
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
    let is_search = matches!(call_type, Some(CallTypes::search | CallTypes::asearch));
    if !is_search
        && let Some((model, total)) =
            unregistered_replicate_cost(catalog, request, &prepared, provider)
    {
        return Ok(flat_priced(prepared, model, total));
    }
    let needs_token_usage = matches!(
        call_type,
        Some(
            CallTypes::completion
                | CallTypes::acompletion
                | CallTypes::embedding
                | CallTypes::aembedding
                | CallTypes::text_completion
                | CallTypes::atext_completion
                | CallTypes::responses
                | CallTypes::aresponses
                | CallTypes::moderation
                | CallTypes::amoderation
                | CallTypes::generate_content
                | CallTypes::agenerate_content
                | CallTypes::retrieve_batch
                | CallTypes::aretrieve_batch
        )
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
    let request_model = if LlmProviders::AZURE_AI.matches(provider) {
        hidden_model
            .filter(|model| is_azure_model_router(model))
            .or(request.request_model)
    } else {
        request.request_model
    };
    let empty_params = Value::Null;
    let stripped_usage = (custom_pricing != CustomPricing::NONE
        || request.input.model_selection.custom_pricing)
        .then(|| without_provider_stated_cost(usage));
    let usage = stripped_usage.as_ref().unwrap_or(usage);
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
                if LlmProviders::AZURE_AI.matches(provider) && !is_azure_model_router(model) {
                    None
                } else {
                    request_model
                };
            let pricing_model =
                together_pricing_model(catalog, model, provider, &prepared.call_type);
            let call = cost_call(
                &prepared.call_type,
                request,
                &pricing_model,
                token_request_model,
                &empty_params,
            )?;
            let cost_request = ModelCostRequest {
                model: &pricing_model,
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
            if let CostCall::Search {
                number_of_queries,
                optional_params,
            } = call
            {
                let search_model = match provider {
                    Some(provider) if !provider.is_empty() && !model.contains('/') => {
                        format!("{provider}/search")
                    }
                    _ => model.to_owned(),
                };
                return search_provider_cost_per_query(
                    catalog,
                    &search_model,
                    provider,
                    number_of_queries.unwrap_or(1),
                    optional_params,
                );
            }
            cost_per_token_for_call(catalog, cost_request, call)
        },
    )?;
    let router_fee =
        (!is_search && LlmProviders::AZURE_AI.matches(provider) && !is_azure_model_router(&model))
            .then(|| azure_ai_router_fee(catalog, &model, request_model, usage.prompt_tokens))
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
            request.built_in_tool_cost
                + config_driven_tool_cost(catalog, request, &model, provider, region, Some(usage))
        },
        &additional_costs,
        provider,
        request.discount_config,
        request.margin_config,
    );
    let token_breakdown = prepared.usage.as_ref().map(|response_usage| {
        get_token_type_cost_breakdown(
            catalog,
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
) -> Result<f64, CostError> {
    if cache_hit {
        return Ok(0.0);
    }
    if request.input.model_selection.response.is_some()
        && let Some(hidden) = request.input.model_selection.hidden_params
        && let Some(cost) = get_response_cost_from_hidden_params(hidden)?
    {
        return Ok(cost);
    }
    completion_cost_from_response(catalog, request).map(|priced| priced.cost.total)
}
