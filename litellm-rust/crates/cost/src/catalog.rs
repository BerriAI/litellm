use std::collections::HashMap;

use jiff::Timestamp;
use serde_json::Value;

use crate::billed_token_rates::{
    BilledRatesRequest, BilledTokenRates, TokenTypeCostBreakdown,
    get_billed_token_rates as calculate_billed_token_rates,
    get_token_type_cost_breakdown as calculate_token_type_cost_breakdown,
};
use crate::completion_cost::{
    CompletionCost, ResponseCostError, completion_cost, get_response_cost_from_hidden_params,
};
use crate::custom_pricing::CustomTokenRates;
use crate::dashscope_cost::cost_per_token as dashscope_cost_per_token;
use crate::fireworks_cost::{
    FireworksThresholds, cost_per_token as fireworks_cost_per_token, get_base_model_for_pricing,
};
use crate::generic_cost::calculate_generic_cost_from_model_info_with_region;
use crate::image_response_cost::calculate_image_response_cost_from_usage;
use crate::per_second::per_second_pricing_cost;
use crate::perplexity_cost::cost_per_token as perplexity_cost_per_token;
use crate::prompt_caching_savings::{
    PromptCachingSavingsRequest, calculate_prompt_caching_savings,
};
use crate::provider_cache::apply_provider_cache_read_default;
use crate::responses_usage::ChatUsage;
use crate::retrieval_cost::{rerank_cost, vector_store_search_cost};
use crate::tool_cost_dispatch::{BuiltInToolCostRequest, get_cost_for_built_in_tools};
use crate::xai_cost::{cost_per_token as xai_cost_per_token, reported_cost as xai_reported_cost};
use crate::{Cost, Pricing, PricingError, Rates, Request, calculate};

#[derive(Clone, Debug, Default)]
pub struct CostCatalog {
    entries: HashMap<String, Rates>,
}

#[derive(Clone, Debug, Default)]
pub struct ModelInfoCatalog {
    entries: HashMap<String, Value>,
}

#[derive(Clone, Copy, Debug)]
pub struct ModelCostRequest<'a> {
    pub model: &'a str,
    pub provider: Option<&'a str>,
    pub region: Option<&'a str>,
    pub usage: &'a ChatUsage,
    pub service_tier: Option<&'a str>,
    pub data_residency: Option<&'a str>,
    pub vertex_location: Option<&'a str>,
    pub at: Timestamp,
    pub response_time_ms: Option<f64>,
}

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CatalogError {
    ModelNotFound,
    Pricing(PricingError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CatalogResponseError {
    Catalog(CatalogError),
    ProviderCost(ResponseCostError),
}

impl From<PricingError> for CatalogError {
    fn from(value: PricingError) -> Self {
        Self::Pricing(value)
    }
}

impl From<CatalogError> for CatalogResponseError {
    fn from(value: CatalogError) -> Self {
        Self::Catalog(value)
    }
}

impl From<ResponseCostError> for CatalogResponseError {
    fn from(value: ResponseCostError) -> Self {
        Self::ProviderCost(value)
    }
}

fn select_model_key<'a, T>(
    entries: &'a HashMap<String, T>,
    model: &str,
    provider: Option<&str>,
    region: Option<&str>,
) -> Option<&'a str> {
    let normalized = match provider {
        Some(provider) => {
            let prefix = format!("{provider}/");
            let mut name = model;
            while let Some(remainder) = name.strip_prefix(&prefix) {
                if !remainder.starts_with(&prefix) {
                    break;
                }
                name = remainder;
            }
            name
        }
        None => model,
    };
    let model_with_provider = match provider {
        Some(provider) => {
            let prefix = format!("{provider}/");
            let bare = normalized.strip_prefix(&prefix).unwrap_or(normalized);
            let regional = region.map(|region| format!("{provider}/{region}/{bare}"));
            if let Some(regional) = regional.filter(|key| entries.contains_key(key)) {
                regional
            } else if normalized.starts_with(&prefix) {
                normalized.to_owned()
            } else {
                format!("{provider}/{normalized}")
            }
        }
        None => normalized.to_owned(),
    };
    let without_prefix = normalized
        .split_once('/')
        .map_or(normalized, |(_, remainder)| remainder);
    [model_with_provider.as_str(), normalized, without_prefix]
        .into_iter()
        .find_map(|candidate| {
            entries
                .get_key_value(candidate)
                .map(|(key, _)| key.as_str())
        })
}

impl CostCatalog {
    pub fn new(entries: HashMap<String, Rates>) -> Self {
        Self { entries }
    }

    pub fn select_model_key<'a>(
        &'a self,
        model: &str,
        provider: Option<&str>,
        region: Option<&str>,
    ) -> Option<&'a str> {
        select_model_key(&self.entries, model, provider, region)
    }

    pub fn cost_per_token(
        &self,
        model: &str,
        provider: Option<&str>,
        region: Option<&str>,
        request: &Request,
    ) -> Result<Cost, CatalogError> {
        let key = self
            .select_model_key(model, provider, region)
            .ok_or(CatalogError::ModelNotFound)?;
        let pricing = Pricing {
            standard: self.entries[key],
            tiers: &[],
            thresholds: &[],
            off_peak: None,
        };
        Ok(calculate(&pricing, request)?)
    }
}

impl ModelInfoCatalog {
    pub fn new(entries: HashMap<String, Value>) -> Self {
        Self { entries }
    }

    pub fn select_model_key<'a>(
        &'a self,
        model: &str,
        provider: Option<&str>,
        region: Option<&str>,
    ) -> Option<&'a str> {
        select_model_key(&self.entries, model, provider, region)
    }

    pub fn cost_per_token(
        &self,
        request: ModelCostRequest<'_>,
    ) -> Result<(f64, f64), CatalogError> {
        if request.provider == Some("perplexity")
            && let Some(cost) = request.usage.cost
        {
            return Ok((0.0, cost));
        }
        if request.provider == Some("xai")
            && let Some(cost) = xai_reported_cost(request.usage)
        {
            return Ok((0.0, cost));
        }
        let key = self
            .select_model_key(request.model, request.provider, request.region)
            .or_else(|| {
                (request.provider == Some("fireworks_ai"))
                    .then(|| {
                        get_base_model_for_pricing(request.model, FireworksThresholds::default())
                    })
                    .and_then(|category| {
                        self.select_model_key(category, request.provider, request.region)
                    })
            })
            .ok_or(CatalogError::ModelNotFound)?;
        let model_info = apply_provider_cache_read_default(&self.entries[key], request.provider);
        if let Some(cost) = per_second_pricing_cost(&model_info, request.response_time_ms) {
            return Ok(cost);
        }
        if request.provider == Some("perplexity") {
            return Ok(perplexity_cost_per_token(
                request.usage,
                &model_info,
                request.at,
            ));
        }
        if request.provider == Some("xai") {
            return Ok(xai_cost_per_token(request.usage, &model_info, request.at));
        }
        if matches!(
            request.provider,
            Some("dashscope" | "qwencloud" | "qwen_ai_platform")
        ) {
            return Ok(dashscope_cost_per_token(
                request.usage,
                &model_info,
                request.at,
            ));
        }
        if request.provider == Some("fireworks_ai") {
            return Ok(fireworks_cost_per_token(
                request.usage,
                &self.entries[key],
                request.at,
            ));
        }
        Ok(calculate_generic_cost_from_model_info_with_region(
            request.usage,
            &model_info,
            request.service_tier,
            request.provider == Some("xai"),
            request.data_residency,
            request.vertex_location,
            request.at,
        ))
    }

    pub fn get_billed_token_rates(
        &self,
        request: ModelCostRequest<'_>,
        custom_cost_per_token: Option<CustomTokenRates>,
    ) -> Option<BilledTokenRates> {
        let model_info = self
            .select_model_key(request.model, request.provider, request.region)
            .and_then(|key| self.entries.get(key));
        calculate_billed_token_rates(BilledRatesRequest {
            model_info,
            usage: request.usage,
            provider: request.provider,
            service_tier: request.service_tier,
            data_residency: request.data_residency,
            vertex_location: request.vertex_location,
            at: request.at,
            custom_cost_per_token,
        })
    }

    pub fn get_token_type_cost_breakdown(
        &self,
        request: ModelCostRequest<'_>,
        custom_cost_per_token: Option<CustomTokenRates>,
    ) -> TokenTypeCostBreakdown {
        let model_info = self
            .select_model_key(request.model, request.provider, request.region)
            .and_then(|key| self.entries.get(key));
        calculate_token_type_cost_breakdown(BilledRatesRequest {
            model_info,
            usage: request.usage,
            provider: request.provider,
            service_tier: request.service_tier,
            data_residency: request.data_residency,
            vertex_location: request.vertex_location,
            at: request.at,
            custom_cost_per_token,
        })
    }

    pub fn calculate_prompt_caching_savings(&self, request: ModelCostRequest<'_>) -> Option<f64> {
        let model_info = self
            .select_model_key(request.model, request.provider, request.region)
            .and_then(|key| self.entries.get(key))?;
        Some(calculate_prompt_caching_savings(
            PromptCachingSavingsRequest {
                model_info,
                usage: request.usage,
                provider: request.provider,
                service_tier: request.service_tier,
                data_residency: request.data_residency,
                vertex_location: request.vertex_location,
                at: request.at,
            },
        ))
    }

    pub fn calculate_image_response_cost_from_usage(
        &self,
        model: &str,
        provider: Option<&str>,
        region: Option<&str>,
        image_response: &Value,
        at: Timestamp,
    ) -> Option<f64> {
        let model_info = self
            .select_model_key(model, provider, region)
            .and_then(|key| self.entries.get(key))?;
        calculate_image_response_cost_from_usage(image_response, model_info, provider, at)
    }

    pub fn rerank_cost(
        &self,
        model: &str,
        provider: &str,
        region: Option<&str>,
        billed_units: Option<&Value>,
    ) -> (f64, f64) {
        let model_info = self
            .select_model_key(model, Some(provider), region)
            .and_then(|key| self.entries.get(key));
        rerank_cost(provider, model_info, billed_units)
    }

    pub fn vector_store_search_cost(&self, provider: &str, api_type: Option<&str>) -> (f64, f64) {
        vector_store_search_cost(provider, api_type, self.entries.get("vertex_ai/search_api"))
    }

    pub fn built_in_tool_cost(
        &self,
        model: &str,
        provider: Option<&str>,
        region: Option<&str>,
        request: BuiltInToolCostRequest<'_>,
    ) -> f64 {
        let model_info = self
            .select_model_key(model, provider, region)
            .and_then(|key| self.entries.get(key));
        get_cost_for_built_in_tools(request, model_info)
    }

    pub fn completion_cost(
        &self,
        request: CompletionCostRequest<'_>,
    ) -> Result<CompletionCost, CatalogError> {
        let (prompt, output) = self.cost_per_token(request.token)?;
        let built_in_tools = match request.built_in_tools {
            BuiltInToolCharge::Provided(cost) => cost,
            BuiltInToolCharge::FromResponse(tool_request) => self.built_in_tool_cost(
                request.token.model,
                request.token.provider,
                request.token.region,
                tool_request,
            ),
        };
        Ok(completion_cost(
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
        &self,
        request: ResponseCostRequest<'_>,
    ) -> Result<f64, CatalogResponseError> {
        if request.cache_hit {
            return Ok(0.0);
        }
        if let Some(reported) = get_response_cost_from_hidden_params(request.hidden_params)? {
            return Ok(reported);
        }
        Ok(self.completion_cost(request.completion)?.total)
    }
}
