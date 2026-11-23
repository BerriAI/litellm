//! The rows main tries when it prices a call, in its order: `select_model_name_for_cost_calc` and the candidate loop
//! of `completion_cost` in `litellm/cost_calculator.py` and the key normalisation of `cost_per_token`, for the
//! providers of a `/v1/messages` call. Any call main would price another way is `Deferred`, so its pricing stays in
//! Python. The table is reached only through `CandidateTable`, which a catalog implements

pub mod bedrock;
mod pricing;

pub use pricing::*;

use std::str::FromStr;

use bedrock::is_bedrock_cross_region_profile_of;

/// Model names main prices by another path: an Azure Model Router fee, Together AI categories, Replicate run time
const OTHER_PATHS: [&str; 5] = [
    "model-router",
    "model_router",
    "togethercomputer",
    "together_ai",
    "replicate",
];

/// The providers whose calls this module resolves, the ones `/v1/messages` prices Claude through
#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::EnumString, strum::IntoStaticStr)]
pub enum CostProvider {
    #[strum(serialize = "anthropic")]
    Anthropic,
    #[strum(serialize = "bedrock")]
    Bedrock,
    #[strum(serialize = "bedrock_mantle")]
    BedrockMantle,
    #[strum(serialize = "azure_ai")]
    AzureAi,
    #[strum(serialize = "vertex_ai")]
    VertexAi,
}

/// The facts of a finished call that pick its row, as `Logging.response_cost_calculator` hands them to
/// `completion_cost`
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize), serde(default))]
pub struct CallIdentity {
    pub provider: String,
    pub request_model: Option<String>,
    pub response_model: Option<String>,
    pub base_model: Option<String>,
    pub custom_pricing: bool,
    pub deployment_id: Option<String>,
    /// The Bedrock Mantle region, from `_resolve_mantle_region_for_cost`
    pub region: Option<String>,
    /// Bedrock's `model_id` param, an inference profile main prices from
    pub invoked_model_id: Option<String>,
    /// `_hidden_params["provider_response_model"]`
    pub provider_response_model: Option<String>,
    /// `_hidden_params["model"]`, read when the response names no model
    pub hidden_model: Option<String>,
}

/// The rows to try in order, each tried only when the one before it could not be priced, then what is left
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateRows {
    pub rows: Vec<Tried>,
    pub then: AfterRows,
}

/// A row main tries, with the model name it hands `cost_per_token` for it, which keys the discount
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tried {
    pub model: String,
    pub row: RowRef,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AfterRows {
    /// Main raises once every row failed
    Exhausted,
    /// Main prices the call by a path not ported here, once every row before failed
    Deferred(RustRefusal),
}

/// Why a call is left to Python
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RustRefusal {
    /// A provider other than the `/v1/messages` Claude providers
    Provider,
    /// A model name main prices by another path, or renames before its lookup
    ModelName,
    /// A Bedrock inference profile or a `register_model` price on a Bedrock row, read by `_bedrock_pricing_info`
    BedrockPricing,
    /// A `base_model` whose own provider's row would price it
    BaseModelProvider,
    /// A hidden response model or region main reads off the response
    HiddenParams,
    /// A model only a `fallback_generalizations` rule knows, which main bills from the rule
    Generalization,
}

/// What a catalog's `get_model_info(model, provider)` walk finds
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelInfoLookup {
    /// The first name whose row's provider matches, as its key
    Row(String),
    /// No name is a key and a capability rule matches, which prices anything or not
    Rule { prices_anything: bool },
    /// No name is a key and no capability rule matches
    Missing,
}

/// What `register_model` wrote onto a row
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Registration {
    /// Nothing, or no field that prices or is a cache rate
    Unpriced,
    /// A field that prices the call
    Priced,
    /// Cache rates alone
    CacheRatesOnly,
}

/// The names `get_model_info` derives from a model that the Bedrock registration walk reads
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NameForms {
    pub combined_model_name: Option<String>,
    pub split_model: Option<String>,
    pub stripped_model_name: Option<String>,
}

/// What the candidate walk asks of a host's catalog
pub trait CandidateTable {
    /// The table has exactly `key`, as `key in litellm.model_cost`
    fn contains(&self, key: &str) -> bool;
    /// `get_model_info(model, provider)`'s lookup
    fn lookup(&self, model: &str, provider: &str) -> ModelInfoLookup;
    /// The row under exactly `key` declares a rate (`_cost_map_entry_prices_anything`)
    fn prices_anything(&self, key: &str) -> bool;
    /// What `register_model` wrote onto the row under exactly `key`
    fn registration(&self, key: &str) -> Registration;
    fn name_forms(&self, model: &str, provider: &str) -> NameForms;
    /// `model` names a provider, as `LlmProvidersSet` holds it
    fn is_provider(&self, name: &str) -> bool;
    /// `get_model_info` renames `name` before its lookup
    fn is_renamed_model(&self, name: &str) -> bool;
}

/// Where a row came from. A `Registered` row is a deployment's row as the router registered it under its id
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RowProvenance {
    Catalog,
    Registered,
}

/// A row to price: its key in the table, the provider it is billed as, and where it came from
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RowRef {
    pub key: String,
    pub provider: String,
    pub provenance: RowProvenance,
}

/// One step of the candidate loop
enum Step {
    Row(String, RowProvenance),
    Skip,
    Defer(RustRefusal),
}

/// The rows main would try to price `call` with, in its order
pub fn candidates<T: CandidateTable + ?Sized>(table: &T, call: &CallIdentity) -> CandidateRows {
    Walk { table }.candidates(call)
}

/// The row the router registered under deployment id `id`, exactly as stored, billed as `provider`
pub fn deployment_row<T: CandidateTable + ?Sized>(
    table: &T,
    id: &str,
    provider: &str,
) -> Option<RowRef> {
    table.contains(id).then(|| RowRef {
        key: id.to_owned(),
        provider: provider.to_owned(),
        provenance: RowProvenance::Registered,
    })
}

/// The row `cost_per_token(model, custom_llm_provider=provider, region_name=region)` prices with, for a caller that
/// names one model. `Err` when main prices it by another path
pub fn priced_row<T: CandidateTable + ?Sized>(
    table: &T,
    model: &str,
    provider: &str,
    region: Option<&str>,
) -> Result<Option<RowRef>, RustRefusal> {
    Walk { table }.priced_row(model, provider, region)
}

struct Walk<'a, T: ?Sized> {
    table: &'a T,
}

impl<T: CandidateTable + ?Sized> Walk<'_, T> {
    fn candidates(&self, call: &CallIdentity) -> CandidateRows {
        match self.cost_candidates(call) {
            Ok(rows) => rows,
            Err(deferral) => CandidateRows {
                rows: Vec::new(),
                then: AfterRows::Deferred(deferral),
            },
        }
    }

    fn cost_candidates(&self, call: &CallIdentity) -> Result<CandidateRows, RustRefusal> {
        let provider = CostProvider::from_str(&call.provider).map_err(|_| RustRefusal::Provider)?;
        if call.provider_response_model.is_some() || call.hidden_model.is_some() {
            return Err(RustRefusal::HiddenParams);
        }
        let names = [
            &call.request_model,
            &call.response_model,
            &call.base_model,
            &call.deployment_id,
        ];
        if names
            .into_iter()
            .flatten()
            .any(|name| self.priced_elsewhere(name))
        {
            return Err(RustRefusal::ModelName);
        }
        let p = call.provider.as_str();
        if let Some(base_model) = &call.base_model
            && !call.custom_pricing
            && !self.declares_rates(&self.lookup(base_model, p))
        {
            return Err(RustRefusal::BaseModelProvider);
        }
        let explicit = call.custom_pricing || call.base_model.is_some();
        let configured = call
            .request_model
            .as_deref()
            .or(call.response_model.as_deref());
        let bedrock_registration = match (provider, explicit, configured) {
            (CostProvider::Bedrock, false, Some(_)) if call.invoked_model_id.is_some() => {
                return Err(RustRefusal::BedrockPricing);
            }
            (CostProvider::Bedrock, false, Some(configured)) => {
                self.bedrock_registration(configured)?
            }
            _ => None,
        };
        let selected = self.selected_model(call, bedrock_registration.as_deref());
        let raw: Vec<&str> = [
            selected.as_deref(),
            call.response_model.as_deref(),
            call.request_model.as_deref(),
        ]
        .into_iter()
        .flatten()
        .collect();
        let ordered: Vec<&str> = match call.custom_pricing {
            true => raw,
            false => {
                let (unpriced_rules, others): (Vec<&str>, Vec<&str>) = raw
                    .into_iter()
                    .partition(|model| self.is_unpriced_rule(model, p));
                others.into_iter().chain(unpriced_rules).collect()
            }
        };
        let region = call.region.as_deref().filter(|_| !explicit);
        let step = |model| match &bedrock_registration {
            Some(key) => Step::Row(key.clone(), RowProvenance::Registered),
            None => self.priced_key(model, provider, region),
        };
        let tried = |model: &str, key: String, provenance: RowProvenance| Tried {
            model: model.to_owned(),
            row: RowRef {
                provenance: match call.custom_pricing && call.deployment_id.as_deref() == Some(&key)
                {
                    true => RowProvenance::Registered,
                    false => provenance,
                },
                key,
                provider: call.provider.clone(),
            },
        };
        let walked =
            ordered
                .into_iter()
                .try_fold(Vec::new(), |rows: Vec<Tried>, model| match step(model) {
                    Step::Row(key, provenance) => Ok(rows
                        .into_iter()
                        .chain([tried(model, key, provenance)])
                        .collect()),
                    Step::Skip => Ok(rows),
                    Step::Defer(deferral) => Err((rows, deferral)),
                });
        Ok(match walked {
            Ok(rows) => CandidateRows {
                rows,
                then: AfterRows::Exhausted,
            },
            Err((rows, deferral)) => CandidateRows {
                rows,
                then: AfterRows::Deferred(deferral),
            },
        })
    }

    /// The row `cost_per_token(model, custom_llm_provider=provider, region_name=region)` prices with, for a caller that
    /// names one model. `Err` when main prices it by another path
    fn priced_row(
        &self,
        model: &str,
        provider: &str,
        region: Option<&str>,
    ) -> Result<Option<RowRef>, RustRefusal> {
        let cost_provider = CostProvider::from_str(provider).map_err(|_| RustRefusal::Provider)?;
        if self.priced_elsewhere(model) {
            return Err(RustRefusal::ModelName);
        }
        match self.priced_key(model, cost_provider, region) {
            Step::Row(key, provenance) => Ok(Some(RowRef {
                key,
                provider: provider.to_owned(),
                provenance,
            })),
            Step::Skip => Ok(None),
            Step::Defer(deferral) => Err(deferral),
        }
    }

    /// Port of `select_model_name_for_cost_calc` for a call with no hidden response facts. `bedrock_registration` is
    /// the row `_bedrock_pricing_info` found
    fn selected_model(
        &self,
        call: &CallIdentity,
        bedrock_registration: Option<&str>,
    ) -> Option<String> {
        let p = call.provider.as_str();
        let explicit = call.custom_pricing || call.base_model.is_some();
        let priced_region = call
            .region
            .as_deref()
            .filter(|_| !explicit && call.response_model.is_some());
        let chosen = match (
            call.custom_pricing,
            call.base_model.as_deref().or(bedrock_registration),
        ) {
            (true, _) => call
                .deployment_id
                .as_deref()
                .filter(|id| self.table.prices_anything(id))
                .or(call.request_model.as_deref()),
            (false, priced) => priced,
        };
        if chosen.is_none()
            && let Some(served) = &call.response_model
        {
            return Some(self.served_or_priced_requested(
                served,
                call.request_model.as_deref(),
                p,
                priced_region,
            ));
        }
        chosen
            .or(call.request_model.as_deref())
            .map(|model| self.cost_map_model_name(model, p, priced_region))
    }

    /// Port of `_served_or_priced_requested_model`
    fn served_or_priced_requested(
        &self,
        served_model: &str,
        requested_model: Option<&str>,
        provider: &str,
        region: Option<&str>,
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
        let profile_of_served = provider == "bedrock"
            && is_bedrock_cross_region_profile_of(requested_model, served_model);
        if profile_of_served && self.resolves_to_rates(&requested, provider) {
            return requested;
        }
        if self.resolves_to_rates(&served, provider) {
            return served;
        }
        match self.resolves_to_rates(&requested, provider) {
            true => requested,
            false => served,
        }
    }

    /// Port of `_cost_map_model_name`
    fn cost_map_model_name(&self, model: &str, provider: &str, region: Option<&str>) -> String {
        let first_segment = model.split('/').next().unwrap_or(model);
        if self.table.is_provider(first_segment) {
            return model.to_owned();
        }
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
            .position(|segment| self.table.is_provider(segment))
            .unwrap_or(tail.len());
        let last_start = strippable.min(tail.len().saturating_sub(1));
        (0..=last_start)
            .map(|start| format!("{head}/{}", tail[start..].join("/")))
            .find(|candidate| self.table.contains(candidate))
            .unwrap_or_else(|| model.to_owned())
    }

    /// The key `cost_per_token` prices `model` under, its provider branch included
    fn priced_key(&self, model: &str, provider: CostProvider, region: Option<&str>) -> Step {
        let p: &'static str = provider.into();
        let prefix = format!("{p}/");
        let model = deduplicated_prefix(model, &prefix);
        let provider_relative = format!("{prefix}{model}");
        let regional = region
            .map(|region| {
                format!(
                    "{p}/{region}/{}",
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
        let lookup = match provider {
            CostProvider::VertexAi if !without_prefix.contains("claude") => {
                return Step::Defer(RustRefusal::ModelName);
            }
            CostProvider::VertexAi => self.lookup(without_prefix, p),
            CostProvider::Bedrock => match self.bedrock_registration(chosen) {
                Ok(Some(key)) => return Step::Row(key, RowProvenance::Registered),
                Ok(None) => self.lookup(chosen, p),
                Err(deferral) => return Step::Defer(deferral),
            },
            _ => self.lookup(chosen, p),
        };
        match lookup {
            ModelInfoLookup::Row(key) => Step::Row(key, RowProvenance::Catalog),
            ModelInfoLookup::Rule { .. } => Step::Defer(RustRefusal::Generalization),
            ModelInfoLookup::Missing => Step::Skip,
        }
    }

    fn lookup(&self, model: &str, provider: &str) -> ModelInfoLookup {
        self.table.lookup(model, provider)
    }

    /// Port of `_is_unpriced_generalized_candidate`
    fn is_unpriced_rule(&self, model: &str, provider: &str) -> bool {
        matches!(
            self.lookup(model, provider),
            ModelInfoLookup::Rule {
                prices_anything: false
            }
        )
    }

    /// Port of `_resolves_to_rates`
    fn resolves_to_rates(&self, model: &str, provider: &str) -> bool {
        match self.lookup(model, provider) {
            ModelInfoLookup::Row(key) => self.table.prices_anything(&key),
            ModelInfoLookup::Rule { prices_anything } => prices_anything,
            ModelInfoLookup::Missing => false,
        }
    }

    /// Port of `_declares_rates` over a lookup: a key whose own row prices anything
    fn declares_rates(&self, lookup: &ModelInfoLookup) -> bool {
        match lookup {
            ModelInfoLookup::Row(key) => self.table.prices_anything(key),
            ModelInfoLookup::Rule { .. } | ModelInfoLookup::Missing => false,
        }
    }

    /// Port of `_bedrock_cost_registration` with no inference profile: the first row `model` names whose
    /// `register_model` fields price anything, which main bills whatever the candidate. A registration that writes
    /// only cache rates is deferred, since main's server-derived cache field list is not ported
    fn bedrock_registration(&self, model: &str) -> Result<Option<String>, RustRefusal> {
        let forms = self.table.name_forms(model, "bedrock");
        let split = forms.split_model.as_deref().unwrap_or(model);
        let keys = [
            model.to_owned(),
            forms.combined_model_name.clone().unwrap_or_default(),
            format!("bedrock/{split}"),
            split.to_owned(),
            forms.stripped_model_name.clone().unwrap_or_default(),
        ];
        keys.into_iter()
            .find_map(|key| match self.table.registration(&key) {
                Registration::Priced => Some(Ok(key)),
                Registration::CacheRatesOnly => Some(Err(RustRefusal::BedrockPricing)),
                Registration::Unpriced => None,
            })
            .transpose()
    }

    fn priced_elsewhere(&self, name: &str) -> bool {
        let lower = name.to_lowercase();
        self.table.is_renamed_model(name) || OTHER_PATHS.iter().any(|path| lower.contains(path))
    }
}

/// `cost_per_token` keeps one `{provider}/` of a repeated chain such as `openai/openai/gpt-5.5`
fn deduplicated_prefix<'a>(model: &'a str, prefix: &str) -> &'a str {
    match model.strip_prefix(prefix) {
        Some(rest) if rest.starts_with(prefix) => deduplicated_prefix(rest, prefix),
        _ => model,
    }
}
