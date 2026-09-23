use std::collections::HashMap;

use crate::{Cost, Pricing, PricingError, Rates, Request, calculate};

#[derive(Clone, Debug, Default)]
pub struct CostCatalog {
    entries: HashMap<String, Rates>,
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
                if let Some(regional) = regional.filter(|key| self.entries.contains_key(key)) {
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
                self.entries
                    .get_key_value(candidate)
                    .map(|(key, _)| key.as_str())
            })
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
