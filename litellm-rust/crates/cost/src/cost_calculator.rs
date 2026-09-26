//! The public cost entry points mirroring litellm/cost_calculator.py.

use jiff::Timestamp;
use serde_json::Value;

use crate::anthropic_cost::fast_speed_multiplier;
use crate::azure_ai_cost::azure_ai_cost_per_token;
use crate::azure_cost::output_per_second_cost;
use crate::batch::batch_cost_from_model_info;
use crate::catalog::{CostCall, ModelCostRequest, ModelInfoCatalog};
use crate::completion_cost::{
    CompletionCost, completion_cost as apply_completion_cost_adjustments,
    get_response_cost_from_hidden_params,
};
use crate::custom_pricing::{CustomPricing, cost_from_chat_usage};
use crate::dashscope_cost::cost_per_token as dashscope_cost_per_token;
use crate::databricks_cost::databricks_cost_per_token;
use crate::error::CostError;
use crate::fireworks_cost::{
    FireworksThresholds, cost_per_token as fireworks_cost_per_token, get_base_model_for_pricing,
};
use crate::generic_cost::{GenericCostRequest, generic_cost_per_token};
use crate::generic_input::get_cost_per_unit;
use crate::lemonade_cost::lemonade_cost_per_token;
use crate::non_token::{ImageRates, ImageUsage, calculate_image};
use crate::ocr_cost::ocr_cost;
use crate::openai_cost::video_generation_cost;
use crate::per_second::{has_token_or_tiered_pricing, per_second_pricing_cost};
use crate::perplexity_cost::cost_per_token as perplexity_cost_per_token;
use crate::pricing::Rate;
use crate::provider::LlmProviders;
use crate::provider_cache::apply_provider_cache_read_default;
use crate::realtime_cost::{
    combine_usage_objects, event_usage, get_transcription_model_name_from_results,
    partition_results_by_service_tier, transcription_usage_cost,
};
use crate::regional_uplift::get_provider_specific_geo_multiplier;
use crate::responses_usage::ChatUsage;
use crate::retrieval_cost::{rerank_cost, vector_store_search_cost};
use crate::search_cost::search_provider_cost_per_query;
use crate::speech_cost::{
    SpeechCostMetric, cost_per_second, generic_cost_per_character, lyria_generation_cost,
    select_cost_metric_for_model, transcription_usage_has_token_details,
};
use crate::together_cost::together_pricing_model;
use crate::tool_cost_dispatch::{BuiltInToolCostRequest, get_cost_for_built_in_tools};
use crate::vertex_cost::{cost_per_token as vertex_cost_per_token, vertex_cost};
use crate::xai_cost::{cost_per_token as xai_cost_per_token, reported_cost as xai_reported_cost};

#[derive(Clone, Copy, Debug)]
pub struct CompletionCostRequest<'a> {
    pub token: ModelCostRequest<'a>,
    pub built_in_tools: BuiltInToolCharge<'a>,
    pub additional_costs: &'a [f64],
    pub discount_config: &'a Value,
    pub margin_config: &'a Value,
}

#[derive(Clone, Copy, Debug)]
pub enum BuiltInToolCharge<'a> {
    Provided(f64),
    FromResponse(BuiltInToolCostRequest<'a>),
}

#[derive(Clone, Copy, Debug)]
pub struct ResponseCostRequest<'a> {
    pub completion: CompletionCostRequest<'a>,
    pub cache_hit: bool,
    pub hidden_params: &'a Value,
}

pub fn cost_per_token_with_custom(
    catalog: &ModelInfoCatalog,
    request: ModelCostRequest<'_>,
    pricing: CustomPricing,
) -> Result<(f64, f64), CostError> {
    match cost_from_chat_usage(request.usage, pricing, request.response_time_ms)? {
        Some(cost) => Ok((cost.input, cost.output)),
        None => cost_per_token(catalog, request),
    }
}

pub fn cost_per_token_for_call_with_custom(
    catalog: &ModelInfoCatalog,
    request: ModelCostRequest<'_>,
    call: CostCall<'_>,
    pricing: CustomPricing,
) -> Result<(f64, f64), CostError> {
    match cost_from_chat_usage(request.usage, pricing, request.response_time_ms)? {
        Some(cost) => Ok((cost.input, cost.output)),
        None => cost_per_token_for_call(catalog, request, call),
    }
}

pub fn cost_per_token_for_call(
    catalog: &ModelInfoCatalog,
    request: ModelCostRequest<'_>,
    call: CostCall<'_>,
) -> Result<(f64, f64), CostError> {
    with_inferred_provider(catalog, request, |catalog, request| {
        cost_per_token_for_provider(catalog, request, call)
    })
}

fn cost_per_token_for_provider(
    catalog: &ModelInfoCatalog,
    request: ModelCostRequest<'_>,
    call: CostCall<'_>,
) -> Result<(f64, f64), CostError> {
    let provider = request
        .provider
        .and_then(|provider| provider.parse::<LlmProviders>().ok());
    match call {
        CostCall::Token {
            call_type,
            prompt_characters,
            completion_characters,
            request_model,
        } => {
            if provider == Some(LlmProviders::VERTEX_AI) {
                return vertex_cost(
                    catalog,
                    request,
                    call_type,
                    prompt_characters,
                    completion_characters,
                );
            }
            if provider == Some(LlmProviders::AZURE_AI) {
                return azure_ai_cost_per_token(catalog, request, request_model);
            }
            token_cost_per_token(catalog, request)
        }
        CostCall::Speech { prompt_characters } => speech_cost(catalog, request, prompt_characters),
        CostCall::Transcription { duration_seconds } => {
            transcription_cost(catalog, request, duration_seconds)
        }
        CostCall::Rerank { billed_units } => {
            let provider = request.provider.ok_or(CostError::MissingProvider)?;
            Ok(rerank_cost(
                catalog,
                request.model,
                provider,
                request.region,
                billed_units,
            ))
        }
        CostCall::VectorStoreSearch { api_type } => {
            let provider = request.provider.ok_or(CostError::MissingProvider)?;
            Ok(vector_store_search_cost(catalog, provider, api_type))
        }
        CostCall::Search {
            number_of_queries,
            optional_params,
        } => search_provider_cost_per_query(
            catalog,
            request.model,
            request.provider,
            number_of_queries.filter(|count| *count > 0).unwrap_or(1),
            optional_params,
        ),
        CostCall::Ocr {
            response,
            deployment_info,
        } => {
            let published = catalog.entry(request.model, request.provider, request.region);
            Ok(ocr_cost(response, deployment_info, published.as_deref())?)
        }
        CostCall::Batch { deployment_info } => {
            let published = catalog.entry(request.model, request.provider, request.region);
            let has_deployment_rates = deployment_info.is_some_and(|info| {
                [
                    "input_cost_per_token_batches",
                    "input_cost_per_token",
                    "output_cost_per_token_batches",
                    "output_cost_per_token",
                ]
                .into_iter()
                .any(|key| info.get(key).is_some_and(|value| !value.is_null()))
            });
            let info = if has_deployment_rates {
                deployment_info
            } else {
                published.as_deref().or(deployment_info)
            };
            let Some(info) = info else {
                return Ok((0.0, 0.0));
            };
            let cost = batch_cost_from_model_info(
                info,
                request.usage,
                request.provider,
                request.data_residency,
            )?;
            Ok((cost.prompt, cost.completion))
        }
    }
}

pub fn cost_per_token(
    catalog: &ModelInfoCatalog,
    request: ModelCostRequest<'_>,
) -> Result<(f64, f64), CostError> {
    with_inferred_provider(catalog, request, token_cost_per_token)
}

fn with_inferred_provider(
    catalog: &ModelInfoCatalog,
    request: ModelCostRequest<'_>,
    price: impl FnOnce(&ModelInfoCatalog, ModelCostRequest<'_>) -> Result<(f64, f64), CostError>,
) -> Result<(f64, f64), CostError> {
    if request.provider.is_some() {
        return price(catalog, request);
    }
    let inferred = catalog
        .get_llm_provider(request.model)
        .ok_or(CostError::MissingProvider)?;
    let selected = catalog.cost_map_model(request.model, None, request.region);
    price(
        catalog,
        ModelCostRequest {
            model: &selected,
            provider: Some(&inferred.custom_llm_provider),
            ..request
        },
    )
}

fn token_cost_per_token(
    catalog: &ModelInfoCatalog,
    request: ModelCostRequest<'_>,
) -> Result<(f64, f64), CostError> {
    if let Some(cost) = catalog
        .entry(request.model, request.provider, request.region)
        .and_then(|info| per_second_pricing_cost(&info, request.response_time_ms))
    {
        return Ok(cost);
    }
    let provider = request
        .provider
        .and_then(|provider| provider.parse::<LlmProviders>().ok());
    if provider == Some(LlmProviders::AZURE_AI) {
        return azure_ai_cost_per_token(catalog, request, None);
    }
    if provider == Some(LlmProviders::DATABRICKS) {
        return databricks_cost_per_token(catalog, request);
    }
    if provider == Some(LlmProviders::LEMONADE) {
        return Ok(lemonade_cost_per_token(catalog, request));
    }
    if provider == Some(LlmProviders::PERPLEXITY)
        && let Some(cost) = request.usage.cost
    {
        return Ok((0.0, cost));
    }
    if provider == Some(LlmProviders::XAI)
        && let Some(cost) = xai_reported_cost(request.usage)
    {
        return Ok((0.0, cost));
    }
    let selected = catalog
        .select_model_info(request.model, request.provider, request.region)
        .or_else(|| {
            (provider == Some(LlmProviders::FIREWORKS_AI))
                .then(|| get_base_model_for_pricing(request.model, FireworksThresholds::default()))
                .and_then(|category| {
                    catalog.select_model_info(category, request.provider, request.region)
                })
        })
        .ok_or(CostError::ModelNotFound)?;
    let model_info = apply_provider_cache_read_default(&selected.info, request.provider);
    if let Some(cost) = per_second_pricing_cost(&model_info, request.response_time_ms) {
        return Ok(cost);
    }
    if provider == Some(LlmProviders::AZURE)
        && let Some(cost) = output_per_second_cost(&model_info, request.response_time_ms)
    {
        return Ok(cost);
    }
    match provider {
        Some(LlmProviders::PERPLEXITY) => {
            return Ok(perplexity_cost_per_token(
                request.usage,
                &model_info,
                request.at,
            ));
        }
        Some(LlmProviders::XAI) => {
            return Ok(xai_cost_per_token(request.usage, &model_info, request.at));
        }
        Some(
            LlmProviders::DASHSCOPE | LlmProviders::QWENCLOUD | LlmProviders::QWEN_AI_PLATFORM,
        ) => {
            return Ok(dashscope_cost_per_token(
                request.usage,
                &model_info,
                request.at,
            ));
        }
        Some(LlmProviders::FIREWORKS_AI) => {
            return Ok(fireworks_cost_per_token(
                request.usage,
                &selected.info,
                request.at,
            ));
        }
        Some(LlmProviders::VERTEX_AI) => {
            return Ok(vertex_cost_per_token(
                request.usage,
                &model_info,
                request.service_tier,
                request.vertex_location,
                request.at,
            ));
        }
        _ => {}
    }
    let dispatches_before_the_gate = matches!(
        provider,
        Some(
            LlmProviders::VERTEX_AI
                | LlmProviders::ANTHROPIC
                | LlmProviders::BEDROCK
                | LlmProviders::OPENAI
                | LlmProviders::DATABRICKS
                | LlmProviders::FIREWORKS_AI
                | LlmProviders::AZURE
                | LlmProviders::GEMINI
                | LlmProviders::DEEPSEEK
                | LlmProviders::TENCENT
                | LlmProviders::PERPLEXITY
                | LlmProviders::XAI
                | LlmProviders::LEMONADE
                | LlmProviders::DASHSCOPE
                | LlmProviders::QWENCLOUD
                | LlmProviders::QWEN_AI_PLATFORM
                | LlmProviders::AZURE_AI
        )
    );
    if !dispatches_before_the_gate && !has_token_or_tiered_pricing(&model_info) {
        return Ok((0.0, 0.0));
    }
    let (service_tier, data_residency) = match provider {
        Some(
            LlmProviders::ANTHROPIC
            | LlmProviders::BEDROCK
            | LlmProviders::AZURE
            | LlmProviders::GEMINI,
        ) => (request.service_tier, None),
        Some(LlmProviders::DEEPSEEK | LlmProviders::TENCENT) => (None, None),
        _ => (request.service_tier, request.data_residency),
    };
    let cost = generic_cost_per_token(GenericCostRequest {
        model_info: &selected.info,
        usage: request.usage,
        provider: request.provider,
        service_tier,
        data_residency,
        vertex_location: None,
        at: request.at,
    });
    let speed = if provider == Some(LlmProviders::ANTHROPIC) {
        fast_speed_multiplier(&model_info, request.usage)
            * get_provider_specific_geo_multiplier(
                &model_info,
                request
                    .usage
                    .extra
                    .get("inference_geo")
                    .and_then(Value::as_str),
            )
    } else {
        1.0
    };
    Ok((cost.0 * speed, cost.1 * speed))
}

pub fn speech_cost(
    catalog: &ModelInfoCatalog,
    request: ModelCostRequest<'_>,
    prompt_characters: Option<f64>,
) -> Result<(f64, f64), CostError> {
    let provider = request
        .provider
        .and_then(|provider| provider.parse::<LlmProviders>().ok());
    if matches!(
        provider,
        Some(LlmProviders::VERTEX_AI | LlmProviders::VERTEX_AI_BETA)
    ) {
        let model_without_prefix = request
            .model
            .split_once('/')
            .map_or(request.model, |(_, model)| model);
        let lyria_key = if model_without_prefix.starts_with("vertex_ai/") {
            model_without_prefix.to_owned()
        } else {
            format!("vertex_ai/{model_without_prefix}")
        };
        if let Some(cost) = catalog
            .entries()
            .get(&lyria_key)
            .and_then(lyria_generation_cost)
        {
            return Ok((0.0, cost));
        }
    }
    let model_info = catalog
        .entry(request.model, request.provider, request.region)
        .ok_or(CostError::ModelNotFound)?;
    let model_info = &*model_info;
    match select_cost_metric_for_model(model_info)? {
        SpeechCostMetric::PerCharacter => {
            let characters = prompt_characters.ok_or(CostError::MissingPromptCharacters)?;
            let (prompt, completion) =
                generic_cost_per_character(model_info, characters, 0.0, None, Some(0.0));
            Ok((
                prompt.ok_or(CostError::MissingInputCharacterRate)?,
                completion.unwrap_or(0.0),
            ))
        }
        SpeechCostMetric::PerToken => Ok(generic_cost_per_token(GenericCostRequest {
            model_info,
            usage: request.usage,
            provider: request.provider,
            service_tier: request.service_tier,
            data_residency: request.data_residency,
            vertex_location: None,
            at: request.at,
        })),
    }
}

pub fn transcription_cost(
    catalog: &ModelInfoCatalog,
    request: ModelCostRequest<'_>,
    duration_seconds: f64,
) -> Result<(f64, f64), CostError> {
    let model_info = catalog
        .entry(request.model, request.provider, request.region)
        .ok_or(CostError::ModelNotFound)?;
    let model_info = &*model_info;
    if transcription_usage_has_token_details(request.usage) {
        return Ok(generic_cost_per_token(GenericCostRequest {
            model_info,
            usage: request.usage,
            provider: request.provider,
            service_tier: request.service_tier,
            data_residency: request.data_residency,
            vertex_location: None,
            at: request.at,
        }));
    }
    Ok(cost_per_second(model_info, duration_seconds))
}

pub fn handle_realtime_transcription_cost_calculation(
    catalog: &ModelInfoCatalog,
    results: &[Value],
    provider: &str,
    requested_model: &str,
) -> f64 {
    let model = get_transcription_model_name_from_results(results).unwrap_or(requested_model);
    let model_info = catalog.entry(model, Some(provider), None);
    let model_info = model_info.as_deref();
    results
        .iter()
        .filter(|event| {
            event.get("type").and_then(Value::as_str)
                == Some("conversation.item.input_audio_transcription.completed")
        })
        .map(|event| transcription_usage_cost(&event["usage"], model_info))
        .sum()
}

fn cost_map_entry_declares_pricing(
    catalog: &ModelInfoCatalog,
    model: &str,
    provider: &str,
) -> bool {
    [
        catalog.entries().get(model),
        catalog.entries().get(&format!("{provider}/{model}")),
    ]
    .into_iter()
    .flatten()
    .filter_map(Value::as_object)
    .any(|fields| {
        fields
            .iter()
            .any(|(field, value)| field.contains("cost_per") && !value.is_null())
    })
}

pub fn handle_realtime_stream_cost_calculation(
    catalog: &ModelInfoCatalog,
    results: &[Value],
    combined_usage: &ChatUsage,
    provider: &str,
    requested_model: &str,
    data_residency: Option<&str>,
    at: Timestamp,
) -> f64 {
    let (prompt, completion) = results
        .iter()
        .filter(|event| event.get("type").and_then(Value::as_str) == Some("session.created"))
        .filter_map(|event| event.pointer("/session/model").and_then(Value::as_str))
        .chain(std::iter::once(requested_model))
        .find_map(|model| {
            let model_info = catalog.entry(model, Some(provider), None)?;
            let cost = generic_cost_per_token(GenericCostRequest {
                model_info: &model_info,
                usage: combined_usage,
                provider: Some(provider),
                service_tier: None,
                data_residency,
                vertex_location: None,
                at,
            });
            (cost.0 + cost.1 > 0.0 || cost_map_entry_declares_pricing(catalog, model, provider))
                .then_some(cost)
        })
        .unwrap_or((0.0, 0.0));
    prompt
        + completion
        + handle_realtime_transcription_cost_calculation(
            catalog,
            results,
            provider,
            requested_model,
        )
}

pub fn responses_ws_token_cost_by_tier(
    catalog: &ModelInfoCatalog,
    results: &[Value],
    model: &str,
    provider: Option<&str>,
    region: Option<&str>,
    data_residency: Option<&str>,
    at: Timestamp,
) -> Result<f64, CostError> {
    partition_results_by_service_tier(results)
        .into_iter()
        .map(|(service_tier, events)| {
            let usage = combine_usage_objects(
                events
                    .into_iter()
                    .map(event_usage)
                    .collect::<Result<Vec<_>, _>>()?,
            )?;
            let (prompt, completion) = cost_per_token(
                catalog,
                ModelCostRequest {
                    model,
                    provider,
                    region,
                    usage: &usage,
                    service_tier,
                    data_residency,
                    vertex_location: None,
                    at,
                    response_time_ms: None,
                },
            )?;
            Ok(prompt + completion)
        })
        .sum()
}

pub fn completion_cost(
    catalog: &ModelInfoCatalog,
    request: CompletionCostRequest<'_>,
) -> Result<CompletionCost, CostError> {
    let pricing_model = together_pricing_model(
        catalog,
        request.token.model,
        request.token.provider,
        "completion",
    );
    let (prompt, output) = cost_per_token(
        catalog,
        ModelCostRequest {
            model: &pricing_model,
            ..request.token
        },
    )?;
    let built_in_tools = match request.built_in_tools {
        BuiltInToolCharge::Provided(cost) => cost,
        BuiltInToolCharge::FromResponse(tool_request) => get_cost_for_built_in_tools(
            catalog,
            request.token.model,
            request.token.region,
            tool_request,
        ),
    };
    Ok(apply_completion_cost_adjustments(
        prompt,
        output,
        built_in_tools,
        request.additional_costs,
        request.token.provider,
        request.discount_config,
        request.margin_config,
    ))
}

pub fn response_cost_calculator(
    catalog: &ModelInfoCatalog,
    request: ResponseCostRequest<'_>,
) -> Result<f64, CostError> {
    if request.cache_hit {
        return Ok(0.0);
    }
    if let Some(reported) = get_response_cost_from_hidden_params(request.hidden_params)? {
        return Ok(reported);
    }
    Ok(completion_cost(catalog, request.completion)?.total)
}

fn model_without_provider_prefix(model: &str, provider: Option<&str>) -> Option<String> {
    let prefix = format!("{}/", provider.filter(|provider| !provider.is_empty())?);
    model
        .starts_with(&prefix)
        .then(|| model.replace(&prefix, ""))
}

#[derive(Clone, Copy, Debug)]
pub struct DefaultImageCostRequest<'a> {
    pub model: &'a str,
    pub provider: Option<&'a str>,
    pub quality: Option<&'a str>,
    pub n: Option<u64>,
    pub size: Option<&'a str>,
    pub supplied_model_info: Option<&'a Value>,
}

fn image_dimensions(size: &str) -> Result<(u32, u32), CostError> {
    let (height, width) = size.split_once("-x-").ok_or(CostError::InvalidQuantity)?;
    let parse = |value: &str| value.parse::<u32>().map_err(|_| CostError::InvalidQuantity);
    Ok((parse(height)?, parse(width)?))
}

pub fn default_image_cost_calculator(
    catalog: &ModelInfoCatalog,
    request: DefaultImageCostRequest<'_>,
) -> Result<f64, CostError> {
    let raw_size = request.size.unwrap_or("1024-x-1024");
    let size = if raw_size.contains("-x-") {
        raw_size.to_owned()
    } else {
        raw_size.replace('x', "-x-")
    };
    let (height, width) = image_dimensions(&size)?;
    let without_provider = model_without_provider_prefix(request.model, request.provider);
    let base = match (request.provider, without_provider.as_deref()) {
        (Some(provider), Some(model)) => format!("{provider}/{size}/{model}"),
        _ => format!("{size}/{}", request.model),
    };
    let model_tail = request.model.rsplit('/').next().unwrap_or(request.model);
    let without_prefix = format!("{size}/{model_tail}");
    let candidates = [
        request.quality.map(|quality| format!("{quality}/{base}")),
        request
            .provider
            .zip(request.quality)
            .map(|(provider, quality)| {
                format!(
                    "{provider}/{quality}/{size}/{}",
                    without_provider.as_deref().unwrap_or(request.model)
                )
            }),
        Some(base.clone()),
        Some(format!("high/{base}")),
        request
            .quality
            .map(|quality| format!("{quality}/{without_prefix}")),
        Some(without_prefix),
        Some(request.model.to_owned()),
        without_provider.clone(),
    ];
    let shared = candidates
        .iter()
        .flatten()
        .find_map(|candidate| catalog.entries().get(candidate));
    if shared.is_none() && request.supplied_model_info.is_none() {
        return Err(CostError::ModelNotFound);
    }
    let rate = |info: &Value, key: &str| {
        get_cost_per_unit(info, key, None).map_or(Rate::Missing, Rate::Value)
    };
    let tables = [request.supplied_model_info, shared]
        .into_iter()
        .flatten()
        .map(|info| ImageRates {
            input_per_image: rate(info, "input_cost_per_image"),
            output_per_image: rate(info, "output_cost_per_image"),
            input_per_pixel: rate(info, "input_cost_per_pixel"),
        })
        .collect::<Vec<_>>();
    Ok(calculate_image(
        &tables,
        ImageUsage {
            count: request.n.unwrap_or(1),
            width,
            height,
        },
    )?
    .total)
}

pub fn default_video_cost_calculator(
    catalog: &ModelInfoCatalog,
    model: &str,
    duration_seconds: f64,
    provider: Option<&str>,
    deployment_info: Option<&Value>,
    video_resolution: Option<&str>,
) -> Result<f64, CostError> {
    let model_info = match deployment_info {
        Some(info) => info,
        None => {
            let without_provider = model_without_provider_prefix(model, provider);
            let base_model = match (provider, without_provider.as_deref()) {
                (Some(provider), Some(bare)) => format!("{provider}/{bare}"),
                _ => model.to_owned(),
            };
            let model_tail = model.rsplit('/').next().unwrap_or(model);
            [
                Some(base_model.as_str()),
                Some(model),
                Some(model_tail),
                without_provider.as_deref(),
            ]
            .into_iter()
            .flatten()
            .find_map(|candidate| catalog.entries().get(candidate))
            .or_else(|| {
                provider.and_then(|provider| catalog.entries().get(&format!("{provider}/{model}")))
            })
            .ok_or(CostError::ModelNotFound)?
        }
    };
    video_generation_cost(model_info, duration_seconds, video_resolution)
}
