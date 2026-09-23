use std::collections::HashMap;

use jiff::Timestamp;
use serde_json::Value;

use crate::generic_cost::calculate_generic_cost_from_model_info_with_region;
use crate::per_second::per_second_pricing_cost;
use crate::responses_usage::ChatUsage;
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CatalogError {
    ModelNotFound,
    Pricing(PricingError),
}

impl From<PricingError> for CatalogError {
    fn from(value: PricingError) -> Self {
        Self::Pricing(value)
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
        let key = self
            .select_model_key(request.model, request.provider, request.region)
            .ok_or(CatalogError::ModelNotFound)?;
        let model_info = &self.entries[key];
        if let Some(cost) = per_second_pricing_cost(model_info, request.response_time_ms) {
            return Ok(cost);
        }
        Ok(calculate_generic_cost_from_model_info_with_region(
            request.usage,
            model_info,
            request.service_tier,
            request.provider == Some("xai"),
            request.data_residency,
            request.vertex_location,
            request.at,
        ))
    }
}
