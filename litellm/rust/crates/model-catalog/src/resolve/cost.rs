//! The candidates main's `completion_cost` walks and the rows it prices them with, for every provider: the names
//! (`select_model_name_for_cost_calc`, `_base_model_pricing_provider`, `_routed_model_pricing_provider`,
//! `_bedrock_pricing_info`, `_priced_candidates_before_unpriced_rules`), the provider each is priced as
//! (`_pricing_provider_for_cost_candidate` and the loop's own provider), the key normalisation of `cost_per_token`
//! and the `get_model_info` walk (`get_model_info_helper`) in `litellm/cost_calculator.py` and `litellm/utils.py`.
//! A candidate main prices through code only Python has is `AfterRows::Deferred` with its `Deferral`

use std::sync::Arc;

use serde_json::Value;

use super::bedrock::is_bedrock_cross_region_profile_of;
use super::{BACKFILL_MODES, Candidates, provider_matches};
use crate::snapshot::{LiveTable, RowProvenance, RowRef, Snapshot, prices_anything};
use crate::wire::Fields;

/// Model names main prices by another path: an Azure Model Router fee, Together AI categories, Replicate run time
const OTHER_PATHS: [&str; 5] = [
    "model-router",
    "model_router",
    "togethercomputer",
    "together_ai",
    "replicate",
];

/// Providers whose `cost_per_token` branch finds its row through code of its own, which stays in Python
const PYTHON_ROW_PROVIDERS: [&str; 5] = [
    "databricks",
    "fireworks_ai",
    "perplexity",
    "xai",
    "lemonade",
];

/// Providers whose own `get_model_info` serves a row ahead of the table, and Hugging Face, whose row main builds. Each
/// lookup may reach the provider, so the walk leaves them to Python, which looks up only the candidates main reached
const SERVED_PROVIDERS: [&str; 4] = ["ollama", "ollama_chat", "lemonade", "huggingface"];

/// Providers `get_llm_provider` would authenticate to resolve, so a lookup takes the provider the name declares
const AUTHENTICATING_PROVIDERS: [&str; 2] = ["github_copilot", "chatgpt"];

/// What only the host can answer about a name
pub trait NameHost {
    /// `litellm.get_llm_provider(model)`: the model without its provider and the provider, `None` where it raises
    fn infer(&self, model: &str) -> Option<Inferred>;
    /// The row a provider's own `get_model_info` serves for `model` (Ollama and Lemonade read it live), or the row
    /// main builds for a Hugging Face model. `None` when it serves none and the table answers
    fn served_row(&self, model: &str, provider: &str) -> Option<Fields>;
}

/// The model and provider `get_llm_provider` reads off a name
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Inferred {
    pub model: String,
    pub provider: String,
}

/// A host that knows nothing beyond the table, for a caller whose calls always name a table provider
pub struct TableOnly;

impl NameHost for TableOnly {
    fn infer(&self, _model: &str) -> Option<Inferred> {
        None
    }

    fn served_row(&self, _model: &str, _provider: &str) -> Option<Fields> {
        None
    }
}

/// The facts of a finished call that pick its candidates, as `completion_cost` reads them
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CallIdentity {
    /// `custom_llm_provider`, `pricing_provider` when the deployment set one
    pub provider: Option<String>,
    pub request_model: Option<String>,
    pub response_model: Option<String>,
    pub base_model: Option<String>,
    pub custom_pricing: bool,
    pub deployment_id: Option<String>,
    /// The `region_name` argument
    pub region: Option<String>,
    /// Bedrock's `model_id` param, an inference profile main prices from
    pub invoked_model_id: Option<String>,
    /// The response's `_hidden_params`, `None` when it carries none
    pub hidden: Option<HiddenParams>,
    /// The provider main's candidate loop prices with once it read the response's hidden params
    pub loop_provider: Option<String>,
    /// The region main's candidate loop prices with once it read the response's hidden params
    pub loop_region: Option<String>,
}

/// The `_hidden_params` strings that pick a candidate, each a non-empty string or `None`
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HiddenParams {
    pub model: Option<String>,
    pub litellm_model_name: Option<String>,
    pub provider_response_model: Option<String>,
    pub region: Option<String>,
}

/// The names main's loop tries for a call, and the name it selected before ordering them, which `completion_cost`
/// hands the realtime calculator
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CostCandidates {
    pub selected: Option<String>,
    pub names: Vec<CostCandidate>,
}

/// One name main's loop tries and the provider it pairs it with, in main's order
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CostCandidate {
    pub model: Option<String>,
    pub provider: Option<String>,
}

/// The rows to try in order, each tried only when the one before it could not be priced, then what is left
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateRows {
    pub rows: Vec<Tried>,
    pub then: AfterRows,
}

/// A candidate main tries: its place among the names, the model name it hands `cost_per_token`, which keys the
/// discount, the provider it prices it as, the provider its loop holds, the row and the row its breakdown reads
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tried {
    pub candidate: usize,
    pub model: String,
    pub provider: String,
    pub loop_provider: String,
    pub row: TriedRow,
    pub breakdown: Option<TriedRow>,
}

/// A row in the table, or the fields `get_model_info` answers with when no name is a key (a capability rule or a
/// row the provider serves)
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TriedRow {
    Table(RowRef),
    Fields { name: String, fields: Arc<Fields> },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AfterRows {
    /// Main raises once every row failed
    Exhausted,
    /// Main prices the call by a path not ported here, once every row before failed
    Deferred(Deferral),
}

/// Why a candidate is left to Python
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Deferral {
    /// A provider whose `cost_per_token` branch finds its row in code of its own, or a call with no provider
    Provider,
    /// A model name main prices by another path
    ModelName,
    /// A Bedrock inference profile priced with a `register_model` price merged over it
    BedrockPricing,
}

/// The model names `cost_per_token` looks a model up with
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PricedNames {
    pub model: String,
    pub model_without_prefix: String,
}

/// The Bedrock row a call prices every Bedrock candidate with (`_bedrock_pricing_info`): the key of the invoked
/// inference profile or of the row a `register_model` price landed on, and the registration main merges over the
/// profile when both price
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BedrockPricing {
    pub key: String,
    /// The row a `register_model` price landed on, whether it is the key or merged over the profile
    pub registration: Option<String>,
    pub merged: bool,
}

/// The row `cost_per_token`'s provider branch prices a model with, and the name it prices it under
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PricedRow {
    pub model: String,
    pub row: TriedRow,
}

/// What `get_model_info(model, provider)` finds
#[derive(Clone, Debug, PartialEq)]
enum Lookup {
    Row(String),
    Rule { name: String, fields: Fields },
    Served(Fields),
    Missing,
}

impl<T: LiveTable> Snapshot<T> {
    /// The Bedrock row `completion_cost` hands every candidate of `call`, `None` when it hands none
    pub fn bedrock_pricing(
        &self,
        call: &CallIdentity,
        host: &dyn NameHost,
    ) -> Option<BedrockPricing> {
        let explicit = call.custom_pricing || call.base_model.is_some();
        let configured = call
            .request_model
            .as_deref()
            .or(call.response_model.as_deref());
        let provider = self.provider_for(
            call.request_model.as_deref(),
            call.provider.as_deref(),
            host,
        );
        self.bedrock_pricing_row(
            configured,
            call.invoked_model_id.as_deref(),
            provider.as_deref(),
            explicit,
            host,
        )
    }

    /// Port of `select_model_name_for_cost_calc` with the provider `call` names
    pub fn selected_cost_model(&self, call: &CallIdentity, host: &dyn NameHost) -> Option<String> {
        self.selected_model(call, call.provider.as_deref(), host)
    }

    /// The names main's loop tries for `call`, each with the provider it pairs it with, in main's order
    pub fn cost_candidates(&self, call: &CallIdentity, host: &dyn NameHost) -> CostCandidates {
        let request_provider = call.provider.as_deref();
        let inferred = self.provider_for(call.request_model.as_deref(), request_provider, host);
        let base_model_provider = match (&call.base_model, call.custom_pricing) {
            (Some(base_model), false) => {
                self.base_model_pricing_provider(base_model, inferred.as_deref(), host)
            }
            _ => None,
        };
        let pricing_override = base_model_provider.or_else(|| {
            (call.base_model.is_none() && !call.custom_pricing)
                .then(|| self.routed_model_pricing_provider(call, host))
                .flatten()
        });
        let selected_provider = pricing_override.as_deref().or(request_provider);
        let selected = self.selected_model(call, selected_provider, host);
        let raw = [
            (selected.clone(), pricing_override.or(call.provider.clone())),
            (call.response_model.clone(), call.provider.clone()),
        ]
        .into_iter()
        .chain(
            call.request_model
                .clone()
                .map(|model| (Some(model), call.provider.clone())),
        );
        let raw: Vec<CostCandidate> = raw
            .map(|(model, provider)| CostCandidate { model, provider })
            .collect();
        if call.custom_pricing {
            return CostCandidates {
                selected,
                names: raw,
            };
        }
        let (unpriced_rules, priced): (Vec<CostCandidate>, Vec<CostCandidate>) = raw
            .into_iter()
            .partition(|candidate| self.is_unpriced_rule(candidate, host));
        CostCandidates {
            selected,
            names: priced.into_iter().chain(unpriced_rules).collect(),
        }
    }

    /// The rows main's loop prices `call`'s candidates with, in its order. A token call stops at the first that
    /// prices, and a candidate main prices by a path only Python has defers the rest of the walk, as does a call whose
    /// loop holds no provider, which main infers per candidate
    pub fn candidates(&self, call: &CallIdentity, host: &dyn NameHost) -> CandidateRows {
        let names = self.cost_candidates(call, host).names;
        let explicit = call.custom_pricing || call.base_model.is_some();
        let bedrock_row = match self.bedrock_pricing(call, host) {
            Some(BedrockPricing { merged: true, .. }) => {
                return CandidateRows {
                    rows: Vec::new(),
                    then: AfterRows::Deferred(Deferral::BedrockPricing),
                };
            }
            Some(BedrockPricing { key, .. }) => Some(key),
            None => None,
        };
        let region = call.loop_region.as_deref().filter(|_| !explicit);
        let Some(held) = call.loop_provider.clone() else {
            return CandidateRows {
                rows: Vec::new(),
                then: AfterRows::Deferred(Deferral::Provider),
            };
        };
        let walked = names.iter().enumerate().try_fold(
            Vec::new(),
            |rows: Vec<Tried>, (index, candidate)| {
                let Some(model) = candidate.model.as_deref() else {
                    return Ok(rows);
                };
                let provider = match candidate.provider != call.provider {
                    true => candidate.provider.clone(),
                    false => Some(held.clone()),
                };
                let Some(provider) = provider else {
                    return Err((rows, Deferral::Provider));
                };
                if self.priced_elsewhere(model) || held == "together_ai" {
                    return Err((rows, Deferral::ModelName));
                }
                if PYTHON_ROW_PROVIDERS.contains(&provider.as_str())
                    || SERVED_PROVIDERS.contains(&provider.as_str())
                {
                    return Err((rows, Deferral::Provider));
                }
                let priced = match (&bedrock_row, provider.as_str()) {
                    (Some(key), "bedrock") => Some(PricedRow {
                        model: self.priced_names(model, Some(&provider), region).model,
                        row: TriedRow::Table(RowRef {
                            key: key.clone(),
                            provider: provider.clone(),
                            provenance: RowProvenance::Registered,
                        }),
                    }),
                    _ => self.priced_key(model, &provider, region, host),
                };
                let breakdown = match &bedrock_row {
                    Some(key) => Some(TriedRow::Table(RowRef {
                        key: key.clone(),
                        provider: provider.clone(),
                        provenance: RowProvenance::Registered,
                    })),
                    None => self.tried_row(self.lookup(model, Some(&provider), host), &provider),
                };
                let tried = |PricedRow { row, .. }: PricedRow| Tried {
                    candidate: index,
                    model: model.to_owned(),
                    provider: provider.clone(),
                    loop_provider: held.clone(),
                    row: match row {
                        TriedRow::Table(row)
                            if call.custom_pricing
                                && call.deployment_id.as_deref() == Some(&row.key) =>
                        {
                            TriedRow::Table(RowRef {
                                provenance: RowProvenance::Registered,
                                ..row
                            })
                        }
                        row => row,
                    },
                    breakdown,
                };
                Ok(rows.into_iter().chain(priced.map(tried)).collect())
            },
        );
        match walked {
            Ok(rows) => CandidateRows {
                rows,
                then: AfterRows::Exhausted,
            },
            Err((rows, deferral)) => CandidateRows {
                rows,
                then: AfterRows::Deferred(deferral),
            },
        }
    }

    /// The row `cost_per_token(model, custom_llm_provider=provider, region_name=region)` prices with, for a caller that
    /// names one model. `Err` when main prices it by another path
    pub fn priced_row(
        &self,
        model: &str,
        provider: &str,
        region: Option<&str>,
        host: &dyn NameHost,
    ) -> Result<Option<PricedRow>, Deferral> {
        if PYTHON_ROW_PROVIDERS.contains(&provider) {
            return Err(Deferral::Provider);
        }
        Ok(self.priced_key(model, provider, region, host))
    }

    /// The names `cost_per_token` looks `model` up with: the key main settles on (`model`) and the name without a
    /// leading provider segment (`model_without_prefix`). Only a caller that names `provider` gets the provider joined
    /// on and a repeated provider prefix dropped
    pub fn priced_names(
        &self,
        model: &str,
        provider: Option<&str>,
        region: Option<&str>,
    ) -> PricedNames {
        let Some(provider) = provider else {
            let without_prefix = model.split_once('/').map_or(model, |(_, rest)| rest);
            let chosen = [model, without_prefix]
                .into_iter()
                .find(|name| self.table.contains(name))
                .unwrap_or(model);
            return PricedNames {
                model: chosen.to_owned(),
                model_without_prefix: without_prefix.to_owned(),
            };
        };
        let prefix = format!("{provider}/");
        let model = deduplicated_prefix(model, &prefix);
        let provider_relative = format!("{prefix}{model}");
        let regional = region
            .map(|region| {
                format!(
                    "{provider}/{region}/{}",
                    model.strip_prefix(&prefix).unwrap_or(model)
                )
            })
            .filter(|regional| self.table.contains(regional));
        let with_provider = match (regional, model.starts_with(&prefix)) {
            (Some(regional), _) => regional,
            (None, true) => model.to_owned(),
            (None, false) => provider_relative.clone(),
        };
        let without_prefix = match model.split_once('/') {
            Some((_, rest))
                if !(with_provider == provider_relative && self.table.contains(&with_provider)) =>
            {
                rest
            }
            _ => model,
        };
        let chosen = [with_provider.as_str(), model, without_prefix]
            .into_iter()
            .find(|name| self.table.contains(name))
            .unwrap_or(model);
        PricedNames {
            model: chosen.to_owned(),
            model_without_prefix: without_prefix.to_owned(),
        }
    }

    /// `_get_provider_for_cost_calc`: the provider the caller named, else the one `get_llm_provider` infers
    fn provider_for(
        &self,
        model: Option<&str>,
        provider: Option<&str>,
        host: &dyn NameHost,
    ) -> Option<String> {
        match provider {
            Some(provider) => Some(provider.to_owned()),
            None => host.infer(model?).map(|found| found.provider),
        }
    }

    /// Port of `select_model_name_for_cost_calc`
    fn selected_model(
        &self,
        call: &CallIdentity,
        provider: Option<&str>,
        host: &dyn NameHost,
    ) -> Option<String> {
        let provider = self.provider_for(call.request_model.as_deref(), provider, host);
        let provider = provider.as_deref();
        let hidden = call.hidden.as_ref();
        let provider_response_model =
            hidden.and_then(|hidden| hidden.provider_response_model.as_deref());
        let requested = call
            .request_model
            .as_deref()
            .or(hidden.and_then(|hidden| hidden.litellm_model_name.as_deref()));
        let explicit = call.custom_pricing || call.base_model.is_some();
        let configured = call
            .request_model
            .as_deref()
            .or(call.response_model.as_deref());
        let bedrock_key = self
            .bedrock_pricing_row(
                configured,
                call.invoked_model_id.as_deref(),
                provider,
                explicit,
                host,
            )
            .map(|pricing| pricing.key);
        let priced_from_response =
            provider_response_model.is_some() || call.response_model.is_some();
        let priced_region = (!explicit && priced_from_response)
            .then(|| {
                hidden
                    .and_then(|hidden| hidden.region.as_deref())
                    .or(call.region.as_deref())
            })
            .flatten();
        let served = |model: &str| {
            self.served_or_priced_requested(model, requested, provider, priced_region, host)
        };
        let chosen = if call.custom_pricing {
            call.deployment_id
                .as_deref()
                .filter(|id| {
                    self.table.contains(id)
                        && self.fields(id).is_some_and(|row| prices_anything(&row))
                })
                .or(call.request_model.as_deref())
        } else if let Some(priced) = call.base_model.as_deref().or(bedrock_key.as_deref()) {
            Some(priced)
        } else if let Some(served_model) = provider_response_model {
            return Some(served(served_model));
        } else {
            match (&call.response_model, hidden) {
                (None, Some(hidden)) => hidden.model.as_deref(),
                _ => None,
            }
        };
        if chosen.is_none()
            && let Some(response_model) = &call.response_model
        {
            return Some(served(response_model));
        }
        chosen
            .or(call.request_model.as_deref())
            .map(|model| self.cost_map_model_name(model, provider, priced_region))
    }

    /// Port of `_base_model_pricing_provider`: the provider an explicit base model is priced under when the
    /// deployment's own provider has no rates for it but the base model's own row does
    fn base_model_pricing_provider(
        &self,
        base_model: &str,
        provider: Option<&str>,
        host: &dyn NameHost,
    ) -> Option<String> {
        let provider = provider?;
        if self.declares_rates(&self.lookup(base_model, Some(provider), host)) {
            return None;
        }
        let own = self.lookup(base_model, None, host);
        if !self.declares_rates(&own) {
            return None;
        }
        match own {
            Lookup::Row(key) => self
                .fields(&key)?
                .get("litellm_provider")
                .and_then(Value::as_str)
                .map(str::to_owned),
            Lookup::Served(fields) => fields
                .get("litellm_provider")
                .and_then(Value::as_str)
                .map(str::to_owned),
            Lookup::Rule { .. } | Lookup::Missing => None,
        }
    }

    /// Port of `_routed_model_pricing_provider`: an Azure Model Router call whose served model only Azure AI prices
    fn routed_model_pricing_provider(
        &self,
        call: &CallIdentity,
        host: &dyn NameHost,
    ) -> Option<String> {
        let (Some(request_model), Some(served_model)) = (&call.request_model, &call.response_model)
        else {
            return None;
        };
        if call.provider.as_deref() != Some("azure")
            || !is_model_router(request_model)
            || self.resolves_to_rates(served_model, Some("azure"), host)
        {
            return None;
        }
        self.resolves_to_rates(served_model, Some("azure_ai"), host)
            .then(|| "azure_ai".to_owned())
    }

    /// Port of `_served_or_priced_requested_model`
    fn served_or_priced_requested(
        &self,
        served_model: &str,
        requested_model: Option<&str>,
        provider: Option<&str>,
        region: Option<&str>,
        host: &dyn NameHost,
    ) -> String {
        let served = self.cost_map_model_name(served_model, provider, region);
        let Some(requested_model) = requested_model else {
            return served;
        };
        let requested = self.cost_map_model_name(requested_model, provider, None);
        let requested_tail = requested.split_once('/').map_or("", |(_, tail)| tail);
        if served == requested || served == requested_tail {
            return served;
        }
        let pricing_provider = self.provider_for(Some(&requested), provider, host);
        let pricing_provider = pricing_provider.as_deref();
        let profile_of_served = pricing_provider == Some("bedrock")
            && is_bedrock_cross_region_profile_of(requested_model, served_model);
        if profile_of_served && self.resolves_to_rates(&requested, pricing_provider, host) {
            return requested;
        }
        if self.resolves_to_rates(&served, pricing_provider, host) {
            return served;
        }
        match self.resolves_to_rates(&requested, pricing_provider, host) {
            true => requested,
            false => served,
        }
    }

    /// Port of `_cost_map_model_name`
    fn cost_map_model_name(
        &self,
        model: &str,
        provider: Option<&str>,
        region: Option<&str>,
    ) -> String {
        let first_segment = model.split('/').next().unwrap_or(model);
        let Some(provider) = provider.filter(|_| !self.names.providers.contains(first_segment))
        else {
            return model.to_owned();
        };
        let prefix = match region {
            Some(region) => format!("{provider}/{region}"),
            None => provider.to_owned(),
        };
        self.strip_unregistered_leading_segments(&format!("{prefix}/{model}"), region)
    }

    /// Port of `_strip_unregistered_leading_segments`
    fn strip_unregistered_leading_segments(&self, model: &str, region: Option<&str>) -> String {
        let segments: Vec<&str> = model.split('/').collect();
        if self.table.contains(&segments[1..].join("/")) {
            return model.to_owned();
        }
        let head_len = match region {
            Some(region) if segments.len() > 2 && segments[1] == region => 2,
            _ => 1,
        };
        let head = segments[..head_len].join("/");
        let tail = &segments[head_len..];
        let strippable = tail
            .iter()
            .position(|segment| self.names.providers.contains(*segment))
            .unwrap_or(tail.len());
        let last_start = strippable.min(tail.len().saturating_sub(1));
        (0..=last_start)
            .map(|start| format!("{head}/{}", tail[start..].join("/")))
            .find(|candidate| self.table.contains(candidate))
            .unwrap_or_else(|| model.to_owned())
    }

    /// The row `cost_per_token` prices `model` with as `provider`, its provider branch included
    fn priced_key(
        &self,
        model: &str,
        provider: &str,
        region: Option<&str>,
        host: &dyn NameHost,
    ) -> Option<PricedRow> {
        let names = self.priced_names(model, Some(provider), region);
        let priced_model = match provider {
            "vertex_ai" => names.model_without_prefix,
            _ => names.model,
        };
        if provider == "bedrock"
            && let Some(key) = self.bedrock_registration(&priced_model)
        {
            return Some(PricedRow {
                model: priced_model,
                row: TriedRow::Table(RowRef {
                    key,
                    provider: provider.to_owned(),
                    provenance: RowProvenance::Registered,
                }),
            });
        }
        let row = self.tried_row(self.lookup(&priced_model, Some(provider), host), provider)?;
        Some(PricedRow {
            model: priced_model,
            row,
        })
    }

    fn tried_row(&self, lookup: Lookup, provider: &str) -> Option<TriedRow> {
        match lookup {
            Lookup::Row(key) => Some(match self.filled(&key) {
                Some(fields) => TriedRow::Fields {
                    name: key,
                    fields: Arc::new(fields),
                },
                None => TriedRow::Table(RowRef {
                    key,
                    provider: provider.to_owned(),
                    provenance: RowProvenance::Catalog,
                }),
            }),
            Lookup::Rule { name, fields } => Some(TriedRow::Fields {
                name,
                fields: Arc::new(fields),
            }),
            Lookup::Served(fields) => Some(TriedRow::Fields {
                name: fields
                    .get("key")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                fields: Arc::new(fields),
            }),
            Lookup::Missing => None,
        }
    }

    /// Port of `get_model_info_helper`'s walk: the renames, a provider's own row, the first of the seven names whose
    /// row's provider matches, else a capability rule when none of the names is a key
    fn lookup(&self, model: &str, provider: Option<&str>, host: &dyn NameHost) -> Lookup {
        let model = self
            .names
            .renamed_models
            .get(model)
            .map_or(model, String::as_str);
        let provider = match provider {
            Some("vertex_ai_beta") => Some("vertex_ai"),
            provider => provider,
        };
        let model = match provider {
            Some("vertex_ai") => self
                .names
                .vertex_renamed
                .get(model)
                .map_or(model, String::as_str),
            _ => model,
        };
        let provider = provider.or_else(|| declared_authenticating_provider(model));
        let (candidates, provider) = Candidates::resolved(model, provider, host);
        let served = provider
            .as_deref()
            .filter(|provider| SERVED_PROVIDERS.contains(provider))
            .and_then(|provider| host.served_row(model, provider));
        if let Some(served) = served.filter(|served| {
            served
                .get("key")
                .and_then(Value::as_str)
                .is_none_or(|key| !self.table.contains(key))
        }) {
            return Lookup::Served(served);
        }
        let provider = provider.unwrap_or_default();
        let registered = candidates.in_lookup_order().find_map(|name| {
            let key = self.table.find_key(name)?;
            let row = self.fields(&key)?;
            provider_matches(
                row.get("litellm_provider").and_then(Value::as_str),
                &provider,
            )
            .then_some(key)
        });
        if let Some(key) = registered {
            return Lookup::Row(key);
        }
        self.generalization(&candidates)
            .map_or(Lookup::Missing, |(name, fields)| Lookup::Rule {
                name,
                fields,
            })
    }

    /// The row under `key` with the fields `get_model_info` fills in from the capability rules
    /// (`mapped_model_fill_missing`), `None` when no rule fills a rate the row lacks, as only a rate changes its price
    fn filled(&self, key: &str) -> Option<Fields> {
        let row = self.fields(key)?;
        let mode = row.get("mode").map_or(Some("chat"), Value::as_str)?;
        if !BACKFILL_MODES.contains(&mode) {
            return None;
        }
        let provider = match row.get("litellm_provider") {
            None => "",
            Some(provider) => provider.as_str()?,
        };
        let missing: Fields = self
            .rules
            .filling(key, provider)
            .into_iter()
            .filter(|(field, _)| !row.contains_key(field))
            .collect();
        prices_anything(&missing).then(|| missing.into_iter().chain((*row).clone()).collect())
    }

    /// Port of `_get_model_info_from_generalization`: rules lose to any name that is a key, whatever its provider
    fn generalization(&self, candidates: &Candidates) -> Option<(String, Fields)> {
        if candidates
            .in_lookup_order()
            .any(|name| self.table.find_key(name).is_some())
        {
            return None;
        }
        candidates
            .in_lookup_order()
            .find_map(|name| Some((name.to_owned(), self.rules.matching(name)?)))
    }

    /// Port of `_is_unpriced_generalized_candidate`
    fn is_unpriced_rule(&self, candidate: &CostCandidate, host: &dyn NameHost) -> bool {
        let Some(model) = candidate.model.as_deref() else {
            return false;
        };
        let (names, _) = Candidates::resolved(model, candidate.provider.as_deref(), host);
        self.generalization(&names)
            .is_some_and(|(_, rule)| !prices_anything(&rule))
    }

    /// Port of `_resolves_to_rates`: the row `model` resolves to declares a rate, or the rule it falls back to does
    fn resolves_to_rates(&self, model: &str, provider: Option<&str>, host: &dyn NameHost) -> bool {
        let unmapped = || {
            let (names, _) = Candidates::resolved(model, provider, host);
            self.generalization(&names)
                .is_some_and(|(_, rule)| prices_anything(&rule))
        };
        match self.lookup(model, provider, host) {
            Lookup::Row(key) => match self.filled(&key) {
                Some(filled) => prices_anything(&filled),
                None => self.fields(&key).is_some_and(|row| prices_anything(&row)),
            },
            Lookup::Rule { .. } => unmapped(),
            Lookup::Served(fields) => match fields
                .get("key")
                .and_then(Value::as_str)
                .and_then(|key| self.fields(key))
            {
                Some(row) => prices_anything(&row),
                None => unmapped(),
            },
            Lookup::Missing => false,
        }
    }

    /// Port of `_declares_rates`: the key a lookup found holds a row that prices anything
    fn declares_rates(&self, lookup: &Lookup) -> bool {
        let key = match lookup {
            Lookup::Row(key) => Some(key.as_str()),
            Lookup::Served(fields) => fields.get("key").and_then(Value::as_str),
            Lookup::Rule { name, .. } => Some(name.as_str()),
            Lookup::Missing => None,
        };
        key.and_then(|key| self.fields(key))
            .is_some_and(|row| prices_anything(&row))
    }

    /// Port of `_bedrock_pricing_info`: the key of the inference profile a call invoked when that prices, else the
    /// row a `register_model` price landed on, and the registration main merges over the profile
    fn bedrock_pricing_row(
        &self,
        model: Option<&str>,
        invoked_model_id: Option<&str>,
        provider: Option<&str>,
        explicit: bool,
        host: &dyn NameHost,
    ) -> Option<BedrockPricing> {
        let model =
            model.filter(|model| !explicit && provider == Some("bedrock") && !model.is_empty())?;
        let registration = self.bedrock_registration(model);
        let Some(profile) = self.bedrock_profile_key(invoked_model_id, host) else {
            return registration.map(|key| BedrockPricing {
                registration: Some(key.clone()),
                key,
                merged: false,
            });
        };
        Some(BedrockPricing {
            merged: registration
                .as_ref()
                .is_some_and(|registered| *registered != profile),
            registration,
            key: profile,
        })
    }

    /// The key of the inference profile a Bedrock call invoked, when its row declares a rate
    fn bedrock_profile_key(
        &self,
        invoked_model_id: Option<&str>,
        host: &dyn NameHost,
    ) -> Option<String> {
        let lookup = self.lookup(invoked_model_id?, Some("bedrock"), host);
        if !self.declares_rates(&lookup) {
            return None;
        }
        match lookup {
            Lookup::Row(key) => Some(key),
            Lookup::Rule { .. } | Lookup::Served(_) | Lookup::Missing => None,
        }
    }

    /// Port of `_bedrock_cost_registration`: the first row `model` names whose `register_model` fields price anything,
    /// which main bills whatever the candidate. A cache rate counts, as every `cache_*_cost` field `register_model`
    /// writes is one main derives for `/model/info`
    fn bedrock_registration(&self, model: &str) -> Option<String> {
        let candidates = Candidates::potential(model, "bedrock");
        let split = candidates.split_model.as_deref().unwrap_or(model);
        let keys = [
            model.to_owned(),
            candidates.combined_model_name.clone().unwrap_or_default(),
            format!("bedrock/{split}"),
            split.to_owned(),
            candidates.stripped_model_name.clone().unwrap_or_default(),
        ];
        keys.into_iter().find(|key| {
            let registered = self.registered_fields(key);
            let Some(row) = self.fields(key).filter(|_| !registered.is_empty()) else {
                return false;
            };
            let written: Fields = row
                .iter()
                .filter(|(field, value)| registered.contains(*field) && !value.is_null())
                .map(|(field, value)| (field.clone(), value.clone()))
                .collect();
            prices_anything(&written)
                || written
                    .keys()
                    .any(|field| field.starts_with("cache_") && field.contains("cost"))
        })
    }

    fn priced_elsewhere(&self, name: &str) -> bool {
        let lower = name.to_lowercase();
        OTHER_PATHS.iter().any(|path| lower.contains(path))
            || self.names.replicate_models.contains(name) && !self.table.contains(name)
    }
}

/// Port of `is_azure_model_router`
fn is_model_router(model: &str) -> bool {
    let lower = model.to_lowercase();
    lower.contains("model-router")
        || lower.contains("model_router")
        || lower == "azure-model-router"
}

/// Port of `declared_authenticating_provider`: the provider a name declares, when resolving it would authenticate
fn declared_authenticating_provider(model: &str) -> Option<&str> {
    let (declared, _) = model.split_once('/')?;
    AUTHENTICATING_PROVIDERS
        .contains(&declared)
        .then_some(declared)
}

/// `cost_per_token` keeps one `{provider}/` of a repeated chain such as `openai/openai/gpt-5.5`
fn deduplicated_prefix<'a>(model: &'a str, prefix: &str) -> &'a str {
    match model.strip_prefix(prefix) {
        Some(rest) if rest.starts_with(prefix) => deduplicated_prefix(rest, prefix),
        _ => model,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashMap, HashSet};
    use std::sync::LazyLock;

    use indexmap::IndexMap;
    use rstest::rstest;
    use serde::Deserialize;
    use serde_json::json;

    use super::*;
    use crate::snapshot::{KnownNames, TableRow};
    use crate::store::Store;
    use crate::test_support::{TestRow, TestRows, bundled_rows, names, snapshot};

    /// A row `register_model` wrote: its fields as main served them and the keys it registered
    #[derive(Deserialize)]
    struct Registered {
        fields: Fields,
        registered_fields: BTreeSet<String>,
    }

    /// Main's walk for realistic calls across providers, captured once, see `captured_from` in `cost_walk.json`
    #[derive(Deserialize)]
    struct Walk {
        known_names: WalkNames,
        inferred: HashMap<String, Option<(String, String)>>,
        registered: IndexMap<String, Registered>,
        cases: Vec<WalkCase>,
    }

    #[derive(Deserialize)]
    struct WalkNames {
        providers: HashSet<String>,
        renamed_models: HashMap<String, String>,
        vertex_renamed: HashMap<String, String>,
        replicate_models: HashSet<String>,
    }

    #[derive(Deserialize)]
    struct WalkCase {
        name: String,
        identity: WalkIdentity,
        names: Vec<(Option<String>, Option<String>)>,
        priced: Option<usize>,
        keys: Vec<String>,
    }

    #[derive(Deserialize)]
    struct WalkIdentity {
        provider: Option<String>,
        request_model: Option<String>,
        response_model: Option<String>,
        base_model: Option<String>,
        custom_pricing: bool,
        deployment_id: Option<String>,
        region: Option<String>,
        invoked_model_id: Option<String>,
        hidden: WalkHidden,
        loop_provider: Option<String>,
        loop_region: Option<String>,
    }

    #[derive(Deserialize)]
    struct WalkHidden {
        model: Option<String>,
        litellm_model_name: Option<String>,
        provider_response_model: Option<String>,
        region: Option<String>,
    }

    impl From<&WalkIdentity> for CallIdentity {
        fn from(identity: &WalkIdentity) -> Self {
            Self {
                provider: identity.provider.clone(),
                request_model: identity.request_model.clone(),
                response_model: identity.response_model.clone(),
                base_model: identity.base_model.clone(),
                custom_pricing: identity.custom_pricing,
                deployment_id: identity.deployment_id.clone(),
                region: identity.region.clone(),
                invoked_model_id: identity.invoked_model_id.clone(),
                hidden: Some(HiddenParams {
                    model: identity.hidden.model.clone(),
                    litellm_model_name: identity.hidden.litellm_model_name.clone(),
                    provider_response_model: identity.hidden.provider_response_model.clone(),
                    region: identity.hidden.region.clone(),
                }),
                loop_provider: identity.loop_provider.clone(),
                loop_region: identity.loop_region.clone(),
            }
        }
    }

    /// `get_llm_provider`'s answers main gave for the names the cases use
    struct Inferring(&'static HashMap<String, Option<(String, String)>>);

    impl NameHost for Inferring {
        fn infer(&self, model: &str) -> Option<Inferred> {
            let (model, provider) = self.0.get(model)?.clone()?;
            Some(Inferred { model, provider })
        }

        fn served_row(&self, _model: &str, _provider: &str) -> Option<Fields> {
            None
        }
    }

    static WALK: LazyLock<Walk> = LazyLock::new(|| {
        serde_json::from_str(include_str!("cost_walk.json")).expect("the fixture parses")
    });

    static WALKED: LazyLock<Snapshot<Store<TestRows>>> = LazyLock::new(|| {
        let registered = WALK.registered.iter().map(|(key, row)| {
            (
                key.clone(),
                TestRow::new(row.fields.clone(), row.registered_fields.clone()),
            )
        });
        let known = &WALK.known_names;
        snapshot(
            TestRows(bundled_rows().0.into_iter().chain(registered).collect()),
            KnownNames {
                providers: known.providers.clone(),
                renamed_models: known.renamed_models.clone(),
                vertex_renamed: known.vertex_renamed.clone(),
                replicate_models: known.replicate_models.clone(),
            },
        )
    });

    fn walk_case(name: &str) -> &'static WalkCase {
        WALK.cases
            .iter()
            .find(|case| case.name == name)
            .expect("the case is in the fixture")
    }

    fn row_name(row: &TriedRow) -> &str {
        match row {
            TriedRow::Table(row) => &row.key,
            TriedRow::Fields { name, .. } => name,
        }
    }

    #[rstest]
    #[case::openai_plain("openai_plain")]
    #[case::openai_same("openai_same")]
    #[case::openai_no_response("openai_no_response")]
    #[case::openai_prefixed("openai_prefixed")]
    #[case::openai_repeated_prefix("openai_repeated_prefix")]
    #[case::openai_unmapped_request("openai_unmapped_request")]
    #[case::openai_unmapped_both("openai_unmapped_both")]
    #[case::openai_finetune("openai_finetune")]
    #[case::openai_dated_unmapped("openai_dated_unmapped")]
    #[case::openai_cached("openai_cached")]
    #[case::openai_custom_priced("openai_custom_priced")]
    #[case::openai_custom_input_only("openai_custom_input_only")]
    #[case::openai_custom_unpriced("openai_custom_unpriced")]
    #[case::openai_custom_missing("openai_custom_missing")]
    #[case::openai_base_model("openai_base_model")]
    #[case::openai_registered_custom_name("openai_registered_custom_name")]
    #[case::openai_provider_response_model("openai_provider_response_model")]
    #[case::openai_hidden_model("openai_hidden_model")]
    #[case::azure_plain("azure_plain")]
    #[case::azure_deployment_name("azure_deployment_name")]
    #[case::azure_base_model("azure_base_model")]
    #[case::azure_eu_region("azure_eu_region")]
    #[case::azure_renamed("azure_renamed")]
    #[case::azure_model_router("azure_model_router")]
    #[case::azure_custom("azure_custom")]
    #[case::azure_ai_claude("azure_ai_claude")]
    #[case::azure_ai_other("azure_ai_other")]
    #[case::azure_ai_deepseek("azure_ai_deepseek")]
    #[case::azure_ai_router("azure_ai_router")]
    #[case::anthropic_dated("anthropic_dated")]
    #[case::anthropic_unmapped_request("anthropic_unmapped_request")]
    #[case::bedrock_profile("bedrock_profile")]
    #[case::bedrock_converse_route("bedrock_converse_route")]
    #[case::bedrock_llama("bedrock_llama")]
    #[case::bedrock_region("bedrock_region")]
    #[case::bedrock_nova("bedrock_nova")]
    #[case::bedrock_converse_provider("bedrock_converse_provider")]
    #[case::bedrock_invoked_profile("bedrock_invoked_profile")]
    #[case::mantle("mantle")]
    #[case::vertex_gemini("vertex_gemini")]
    #[case::vertex_gemini_version("vertex_gemini_version")]
    #[case::vertex_claude("vertex_claude")]
    #[case::vertex_mistral("vertex_mistral")]
    #[case::vertex_llama("vertex_llama")]
    #[case::vertex_prefixed("vertex_prefixed")]
    #[case::vertex_slash_alias("vertex_slash_alias")]
    #[case::vertex_beta("vertex_beta")]
    #[case::gemini_plain("gemini_plain")]
    #[case::gemini_prefixed("gemini_prefixed")]
    #[case::gemini_dated("gemini_dated")]
    #[case::groq("groq")]
    #[case::groq_nested("groq_nested")]
    #[case::mistral("mistral")]
    #[case::deepseek("deepseek")]
    #[case::deepinfra_nested("deepinfra_nested")]
    #[case::novita("novita")]
    #[case::vercel("vercel")]
    #[case::nvidia_nim("nvidia_nim")]
    #[case::dashscope("dashscope")]
    #[case::qwencloud("qwencloud")]
    #[case::qwen_ai_platform("qwen_ai_platform")]
    #[case::tencent("tencent")]
    #[case::oci("oci")]
    #[case::github_copilot("github_copilot")]
    #[case::snowflake("snowflake")]
    #[case::watsonx("watsonx")]
    #[case::cloudflare("cloudflare")]
    #[case::cohere_chat("cohere_chat")]
    #[case::cerebras("cerebras")]
    #[case::sambanova("sambanova")]
    #[case::moonshot("moonshot")]
    #[case::zai("zai")]
    #[case::minimax("minimax")]
    #[case::nebius("nebius")]
    #[case::friendliai("friendliai")]
    #[case::openrouter("openrouter")]
    #[case::openrouter_unmapped("openrouter_unmapped")]
    #[case::hosted_vllm_unmapped("hosted_vllm_unmapped")]
    #[case::litellm_proxy("litellm_proxy")]
    #[case::opencode_go("opencode_go")]
    #[case::text_completion_openai("text_completion_openai")]
    #[case::xai("xai")]
    #[case::perplexity("perplexity")]
    #[case::fireworks("fireworks")]
    #[case::databricks("databricks")]
    #[case::together("together")]
    #[case::replicate("replicate")]
    #[case::lemonade("lemonade")]
    #[case::chatgpt_base_model("chatgpt_base_model")]
    #[case::pricing_provider("pricing_provider")]
    #[case::no_provider("no_provider")]
    #[case::no_provider_prefixed("no_provider_prefixed")]
    #[case::hidden_provider_differs("hidden_provider_differs")]
    #[case::hidden_region("hidden_region")]
    #[case::rule_only("rule_only")]
    fn the_walk_tries_mains_names_and_prices_the_candidate_main_priced(#[case] name: &str) {
        let case = walk_case(name);
        let call = CallIdentity::from(&case.identity);
        let host = Inferring(&WALK.inferred);

        let names: Vec<(Option<String>, Option<String>)> = WALKED
            .cost_candidates(&call, &host)
            .names
            .into_iter()
            .map(|candidate| (candidate.model, candidate.provider))
            .collect();
        let found = WALKED.candidates(&call, &host);

        assert_eq!(names, case.names);
        match (found.rows.first(), case.priced) {
            (Some(first), Some(priced)) => {
                assert_eq!(first.candidate, priced);
                assert!(
                    case.keys.iter().any(|key| key == row_name(&first.row)),
                    "{} not among {:?}",
                    row_name(&first.row),
                    case.keys
                );
            }
            (Some(first), None) => panic!(
                "main priced nothing, the walk priced {}",
                row_name(&first.row)
            ),
            (None, _) => {
                assert!(case.priced.is_none() || matches!(found.then, AfterRows::Deferred(_)))
            }
        }
    }

    /// Main's billed rows for realistic `/v1/messages` calls, captured once, see `captured_from` in the file
    #[derive(Deserialize)]
    struct Fixture {
        known_names: FixtureNames,
        registered: IndexMap<String, Registered>,
        cases: Vec<Case>,
        bedrock_registration: BedrockRegistration,
    }

    /// A `register_model` price on a Bedrock row a profile's stripped name reaches
    #[derive(Deserialize)]
    struct BedrockRegistration {
        key: String,
        registered: Fields,
        request_model: String,
        response_model: String,
        cases: Vec<RegistrationCase>,
    }

    #[derive(Deserialize)]
    struct RegistrationCase {
        custom_pricing: bool,
        tried: Vec<String>,
    }

    #[derive(Deserialize)]
    struct FixtureNames {
        providers: Vec<String>,
        renamed_models: Vec<String>,
    }

    #[derive(Deserialize)]
    struct Case {
        name: String,
        identity: FixtureIdentity,
        billed_key: Option<String>,
        billed_provider: Option<String>,
    }

    #[derive(Deserialize)]
    struct FixtureIdentity {
        provider: String,
        request_model: Option<String>,
        response_model: Option<String>,
        base_model: Option<String>,
        custom_pricing: bool,
        deployment_id: Option<String>,
        region: Option<String>,
    }

    static FIXTURE: LazyLock<Fixture> = LazyLock::new(|| {
        serde_json::from_str(include_str!("cost_candidates.json")).expect("the fixture parses")
    });

    static MAIN: LazyLock<Snapshot<Store<TestRows>>> = LazyLock::new(|| {
        let registered = FIXTURE.registered.iter().map(|(key, row)| {
            (
                key.clone(),
                TestRow::new(row.fields.clone(), row.registered_fields.clone()),
            )
        });
        snapshot(
            TestRows(bundled_rows().0.into_iter().chain(registered).collect()),
            KnownNames {
                providers: FIXTURE.known_names.providers.iter().cloned().collect(),
                renamed_models: FIXTURE
                    .known_names
                    .renamed_models
                    .iter()
                    .map(|name| (name.clone(), name.clone()))
                    .collect(),
                ..KnownNames::default()
            },
        )
    });

    fn call(provider: &str, request: &str, response: &str) -> CallIdentity {
        CallIdentity {
            provider: Some(provider.to_owned()),
            request_model: Some(request.to_owned()),
            response_model: Some(response.to_owned()),
            loop_provider: Some(provider.to_owned()),
            ..CallIdentity::default()
        }
    }

    const REGISTERED: &str = "us.anthropic.claude-opus-5-5";

    fn table_rows(found: &CandidateRows) -> Vec<(&str, RowProvenance)> {
        found
            .rows
            .iter()
            .filter_map(|tried| match &tried.row {
                TriedRow::Table(row) => Some((row.key.as_str(), row.provenance)),
                TriedRow::Fields { .. } => None,
            })
            .collect()
    }

    #[rstest]
    #[case::anthropic_dated_response("anthropic_dated_response")]
    #[case::anthropic_same_response("anthropic_same_response")]
    #[case::anthropic_prefixed_request("anthropic_prefixed_request")]
    #[case::anthropic_no_response_model("anthropic_no_response_model")]
    #[case::anthropic_unmapped_both("anthropic_unmapped_both")]
    #[case::anthropic_unmapped_request("anthropic_unmapped_request")]
    #[case::anthropic_unmapped_response("anthropic_unmapped_response")]
    #[case::anthropic_unmapped_dated_response("anthropic_unmapped_dated_response")]
    #[case::anthropic_case_folded("anthropic_case_folded")]
    #[case::anthropic_custom_priced_deployment("anthropic_custom_priced_deployment")]
    #[case::anthropic_custom_unpriced_deployment("anthropic_custom_unpriced_deployment")]
    #[case::anthropic_custom_missing_deployment("anthropic_custom_missing_deployment")]
    #[case::anthropic_uncustom_deployment_id_ignored("anthropic_uncustom_deployment_id_ignored")]
    #[case::anthropic_base_model("anthropic_base_model")]
    #[case::anthropic_repeated_prefix("anthropic_repeated_prefix")]
    #[case::anthropic_thrice_repeated_prefix("anthropic_thrice_repeated_prefix")]
    #[case::bedrock_profile_over_served_base_id("bedrock_profile_over_served_base_id")]
    #[case::bedrock_us_profile("bedrock_us_profile")]
    #[case::bedrock_invoke_us_profile("bedrock_invoke_us_profile")]
    #[case::bedrock_global_profile("bedrock_global_profile")]
    #[case::bedrock_eu_dated_profile("bedrock_eu_dated_profile")]
    #[case::bedrock_base_id("bedrock_base_id")]
    #[case::bedrock_claude_platform("bedrock_claude_platform")]
    #[case::bedrock_prefixed_no_response("bedrock_prefixed_no_response")]
    #[case::bedrock_region_path("bedrock_region_path")]
    #[case::bedrock_invoke_apac("bedrock_invoke_apac")]
    #[case::bedrock_mantle_route("bedrock_mantle_route")]
    #[case::bedrock_context_suffix("bedrock_context_suffix")]
    #[case::bedrock_throughput_suffix("bedrock_throughput_suffix")]
    #[case::bedrock_unmapped_profile("bedrock_unmapped_profile")]
    #[case::bedrock_custom_priced_deployment("bedrock_custom_priced_deployment")]
    #[case::mantle_us_east("mantle_us_east")]
    #[case::mantle_gov_region_row("mantle_gov_region_row")]
    #[case::mantle_region_in_model("mantle_region_in_model")]
    #[case::mantle_no_region("mantle_no_region")]
    #[case::mantle_prefixed("mantle_prefixed")]
    #[case::mantle_unmapped("mantle_unmapped")]
    #[case::azure_ai_plain("azure_ai_plain")]
    #[case::azure_ai_deployment_name("azure_ai_deployment_name")]
    #[case::azure_ai_base_model("azure_ai_base_model")]
    #[case::azure_ai_bare_base_model("azure_ai_bare_base_model")]
    #[case::azure_ai_fast_variant("azure_ai_fast_variant")]
    #[case::vertex_plain("vertex_plain")]
    #[case::vertex_versioned("vertex_versioned")]
    #[case::vertex_default_version("vertex_default_version")]
    #[case::vertex_unknown_version("vertex_unknown_version")]
    #[case::vertex_prefixed("vertex_prefixed")]
    #[case::vertex_slash_alias("vertex_slash_alias")]
    fn the_first_row_is_the_one_main_billed_a_messages_call_with(#[case] name: &str) {
        let case = FIXTURE
            .cases
            .iter()
            .find(|case| case.name == name)
            .expect("the case is in the fixture");
        let identity = &case.identity;
        let found = MAIN.candidates(
            &CallIdentity {
                base_model: identity.base_model.clone(),
                custom_pricing: identity.custom_pricing,
                deployment_id: identity.deployment_id.clone(),
                region: identity.region.clone(),
                loop_region: identity.region.clone(),
                request_model: identity.request_model.clone(),
                response_model: identity.response_model.clone(),
                ..call(&identity.provider, "", "")
            },
            &TableOnly,
        );
        let main_billed = case
            .billed_key
            .as_deref()
            .zip(case.billed_provider.as_deref())
            .filter(|(key, _)| MAIN.table.contains(key));

        assert_eq!(
            found.rows.first().and_then(|tried| match &tried.row {
                TriedRow::Table(row) => Some((row.key.as_str(), row.provider.as_str())),
                TriedRow::Fields { .. } => None,
            }),
            main_billed
        );
    }

    #[rstest]
    fn a_bedrock_registration_on_a_stripped_name_is_billed_like_main(
        #[values(true, false)] custom_pricing: bool,
    ) {
        let registration = &FIXTURE.bedrock_registration;
        let case = registration
            .cases
            .iter()
            .find(|case| case.custom_pricing == custom_pricing)
            .expect("the case is in the fixture");
        let rows = bundled_rows();
        let written = TestRow::new(
            rows.0[registration.key.as_str()]
                .fields()
                .into_iter()
                .chain(registration.registered.clone())
                .collect(),
            registration.registered.keys().cloned().collect(),
        );
        let snapshot = snapshot(
            TestRows(
                rows.0
                    .into_iter()
                    .chain([(registration.key.clone(), written)])
                    .collect(),
            ),
            names(&["anthropic", "bedrock"]),
        );

        let found = snapshot.candidates(
            &CallIdentity {
                custom_pricing,
                ..call(
                    "bedrock",
                    &registration.request_model,
                    &registration.response_model,
                )
            },
            &TableOnly,
        );

        assert_eq!(
            table_rows(&found),
            case.tried
                .iter()
                .map(|key| (key.as_str(), RowProvenance::Registered))
                .collect::<Vec<_>>()
        );
    }

    #[rstest]
    fn a_custom_priced_deployment_row_comes_first_as_registered() {
        let found = MAIN.candidates(
            &CallIdentity {
                custom_pricing: true,
                deployment_id: Some("dep-claude-custom".to_owned()),
                ..call("anthropic", "claude-opus-5-5", "claude-opus-5-5")
            },
            &TableOnly,
        );

        assert_eq!(
            table_rows(&found),
            [
                ("dep-claude-custom", RowProvenance::Registered),
                ("claude-opus-5-5", RowProvenance::Catalog),
                ("claude-opus-5-5", RowProvenance::Catalog),
            ]
        );
    }

    #[rstest]
    #[case::python_row_provider(
        call("databricks", "databricks-claude-haiku-4-5", "x"),
        Deferral::Provider
    )]
    #[case::no_loop_provider(
        CallIdentity { loop_provider: None, ..call("openai", "gpt-4o", "gpt-4o") },
        Deferral::Provider
    )]
    #[case::azure_model_router(
        call("azure_ai", "model-router", "model-router"),
        Deferral::ModelName
    )]
    #[case::together_category(
        call("together_ai", "meta-llama/Llama-3-8b", "meta-llama/Llama-3-8b"),
        Deferral::ModelName
    )]
    fn calls_main_prices_another_way_are_deferred(
        #[case] call: CallIdentity,
        #[case] deferral: Deferral,
    ) {
        let found = MAIN.candidates(&call, &TableOnly);

        assert_eq!(found.rows, Vec::new());
        assert_eq!(found.then, AfterRows::Deferred(deferral));
    }

    #[rstest]
    #[case::priced_registration_bills_every_candidate(
        json!({"input_cost_per_token": 1e-6}),
        &[(REGISTERED, RowProvenance::Registered); 3]
    )]
    #[case::cache_only_registration_is_a_price(
        json!({"cache_read_input_token_cost": 1e-7}),
        &[(REGISTERED, RowProvenance::Registered); 3]
    )]
    #[case::capability_registration_is_not_a_price(
        json!({"supports_vision": true, "mode": "chat"}),
        &[(REGISTERED, RowProvenance::Catalog); 2]
    )]
    fn a_bedrock_register_model_price_is_billed_like_main(
        #[case] registered: serde_json::Value,
        #[case] expected: &[(&str, RowProvenance)],
    ) {
        let serde_json::Value::Object(registered) = registered else {
            panic!("a registration is an object")
        };
        let rows = bundled_rows();
        let written = TestRow::new(
            rows.0[REGISTERED]
                .fields()
                .into_iter()
                .chain(registered.clone())
                .collect(),
            registered.keys().cloned().collect(),
        );
        let snapshot = snapshot(
            TestRows(
                rows.0
                    .into_iter()
                    .chain([(REGISTERED.to_owned(), written)])
                    .collect(),
            ),
            names(&["anthropic", "bedrock"]),
        );

        let found =
            snapshot.candidates(&call("bedrock", REGISTERED, "claude-opus-5-5"), &TableOnly);

        assert_eq!(table_rows(&found), expected);
    }

    #[rstest]
    #[case::fireworks_short_name(
        "llama-v3p1-8b-instruct",
        "fireworks_ai/accounts/fireworks/models/llama-v3p1-8b-instruct"
    )]
    #[case::fireworks_router("auto", "fireworks_ai/accounts/fireworks/routers/auto")]
    #[case::fireworks_fast("glm-5-fast", "fireworks_ai/accounts/fireworks/routers/glm-5-fast")]
    #[case::fireworks_account("accounts/me/models/x", "fireworks_ai/accounts/me/models/x")]
    #[case::fireworks_models_path("models/x", "fireworks_ai/accounts/fireworks/models/x")]
    fn fireworks_names_reach_the_key_its_model_info_maps_them_to(
        #[case] model: &str,
        #[case] key: &str,
    ) {
        let (names, _) = Candidates::resolved(model, Some("fireworks_ai"), &TableOnly);

        assert_eq!(names.provider_prefixed_model_name.as_deref(), Some(key));
    }
}
