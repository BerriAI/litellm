//! The rows main tries when it prices a call, in its order: `select_model_name_for_cost_calc` and the candidate loop
//! of `completion_cost` in `litellm/cost_calculator.py`, the key normalisation of `cost_per_token` and the
//! `get_model_info` walk, for the providers of a `/v1/messages` call. Any call main would price another way is
//! `Deferred`, so its pricing stays in Python

use std::str::FromStr;

use serde_json::Value;

use super::Candidates;
use super::bedrock::is_bedrock_cross_region_profile_of;
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
    pub rows: Vec<RowRef>,
    pub then: AfterRows,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AfterRows {
    /// Main raises once every row failed
    Exhausted,
    /// Main prices the call by a path not ported here, once every row before failed
    Deferred(Deferral),
}

/// Why a call is left to Python
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Deferral {
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

/// What `get_model_info(model, provider)` finds
#[derive(Clone, Debug, PartialEq)]
enum Lookup {
    Row(String),
    Rule(Fields),
    Missing,
}

/// One step of the candidate loop
enum Step {
    Row(String, RowProvenance),
    Skip,
    Defer(Deferral),
}

impl<T: LiveTable> Snapshot<T> {
    /// The rows main would try to price `call` with, in its order
    pub fn candidates(&self, call: &CallIdentity) -> CandidateRows {
        match self.cost_candidates(call) {
            Ok(rows) => rows,
            Err(deferral) => CandidateRows {
                rows: Vec::new(),
                then: AfterRows::Deferred(deferral),
            },
        }
    }

    fn cost_candidates(&self, call: &CallIdentity) -> Result<CandidateRows, Deferral> {
        let provider = CostProvider::from_str(&call.provider).map_err(|_| Deferral::Provider)?;
        if call.provider_response_model.is_some() || call.hidden_model.is_some() {
            return Err(Deferral::HiddenParams);
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
            return Err(Deferral::ModelName);
        }
        let p = call.provider.as_str();
        if let Some(base_model) = &call.base_model
            && !call.custom_pricing
            && !self.declares_rates(&self.lookup(base_model, p))
        {
            return Err(Deferral::BaseModelProvider);
        }
        let explicit = call.custom_pricing || call.base_model.is_some();
        let configured = call
            .request_model
            .as_deref()
            .or(call.response_model.as_deref());
        let bedrock_registration = match (provider, explicit, configured) {
            (CostProvider::Bedrock, false, Some(_)) if call.invoked_model_id.is_some() => {
                return Err(Deferral::BedrockPricing);
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
        let steps = ordered
            .into_iter()
            .map(|model| match &bedrock_registration {
                Some(key) => Step::Row(key.clone(), RowProvenance::Registered),
                None => self.priced_key(model, provider, region),
            });
        let mut rows: Vec<RowRef> = Vec::new();
        for step in steps {
            match step {
                Step::Row(key, provenance) => rows.push(RowRef {
                    provenance: match call.custom_pricing
                        && call.deployment_id.as_deref() == Some(&key)
                    {
                        true => RowProvenance::Registered,
                        false => provenance,
                    },
                    key,
                    provider: call.provider.clone(),
                }),
                Step::Skip => {}
                Step::Defer(deferral) => {
                    return Ok(CandidateRows {
                        rows,
                        then: AfterRows::Deferred(deferral),
                    });
                }
            }
        }
        Ok(CandidateRows {
            rows,
            then: AfterRows::Exhausted,
        })
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
                .filter(|id| self.fields(id).is_some_and(|row| prices_anything(&row)))
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
        if self.names.providers.contains(first_segment) {
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
            .position(|segment| self.names.providers.contains(*segment))
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
                return Step::Defer(Deferral::ModelName);
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
            Lookup::Row(key) => Step::Row(key, RowProvenance::Catalog),
            Lookup::Rule(_) => Step::Defer(Deferral::Generalization),
            Lookup::Missing => Step::Skip,
        }
    }

    /// Port of the `get_model_info` walk for a known provider: the first of the seven names whose row's provider
    /// matches, else a capability rule when none of the names is a key
    fn lookup(&self, model: &str, provider: &str) -> Lookup {
        let candidates = Candidates::potential(model, provider);
        let registered = candidates.in_lookup_order().find_map(|name| {
            let key = self.table.find_key(name)?;
            let row = self.fields(&key)?;
            super::provider_matches(
                row.get("litellm_provider").and_then(Value::as_str),
                provider,
            )
            .then_some(key)
        });
        if let Some(key) = registered {
            return Lookup::Row(key);
        }
        self.generalization(&candidates)
            .map_or(Lookup::Missing, Lookup::Rule)
    }

    /// Port of `_get_model_info_from_generalization`: rules lose to any name that is a key, whatever its provider
    fn generalization(&self, candidates: &Candidates) -> Option<Fields> {
        if candidates
            .in_lookup_order()
            .any(|name| self.table.find_key(name).is_some())
        {
            return None;
        }
        candidates
            .in_lookup_order()
            .find_map(|name| self.rules.matching(name))
    }

    /// Port of `_is_unpriced_generalized_candidate`
    fn is_unpriced_rule(&self, model: &str, provider: &str) -> bool {
        self.generalization(&Candidates::potential(model, provider))
            .is_some_and(|rule| !prices_anything(&rule))
    }

    /// Port of `_resolves_to_rates`
    fn resolves_to_rates(&self, model: &str, provider: &str) -> bool {
        match self.lookup(model, provider) {
            Lookup::Row(key) => self.fields(&key).is_some_and(|row| prices_anything(&row)),
            Lookup::Rule(rule) => prices_anything(&rule),
            Lookup::Missing => false,
        }
    }

    /// Port of `_declares_rates` over a lookup: a key whose own row prices anything
    fn declares_rates(&self, lookup: &Lookup) -> bool {
        match lookup {
            Lookup::Row(key) => self.fields(key).is_some_and(|row| prices_anything(&row)),
            Lookup::Rule(_) | Lookup::Missing => false,
        }
    }

    /// Port of `_bedrock_cost_registration` with no inference profile: the first row `model` names whose
    /// `register_model` fields price anything, which main bills whatever the candidate. A registration that writes
    /// only cache rates is deferred, since main's server-derived cache field list is not ported
    fn bedrock_registration(&self, model: &str) -> Result<Option<String>, Deferral> {
        let candidates = Candidates::potential(model, "bedrock");
        let split = candidates.split_model.as_deref().unwrap_or(model);
        let keys = [
            model.to_owned(),
            candidates.combined_model_name.clone().unwrap_or_default(),
            format!("bedrock/{split}"),
            split.to_owned(),
            candidates.stripped_model_name.clone().unwrap_or_default(),
        ];
        keys.into_iter()
            .find_map(|key| {
                let registered = self.registered_fields(&key);
                let row = self.fields(&key).filter(|_| !registered.is_empty())?;
                let written: Fields = row
                    .iter()
                    .filter(|(field, value)| registered.contains(*field) && !value.is_null())
                    .map(|(field, value)| (field.clone(), value.clone()))
                    .collect();
                match (
                    prices_anything(&written),
                    written
                        .keys()
                        .any(|field| field.starts_with("cache_") && field.contains("cost")),
                ) {
                    (true, _) => Some(Ok(key)),
                    (false, true) => Some(Err(Deferral::BedrockPricing)),
                    (false, false) => None,
                }
            })
            .transpose()
    }

    fn priced_elsewhere(&self, name: &str) -> bool {
        let lower = name.to_lowercase();
        self.names.renamed_models.contains(name)
            || OTHER_PATHS.iter().any(|path| lower.contains(path))
    }
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
    use std::collections::BTreeSet;
    use std::sync::LazyLock;

    use indexmap::IndexMap;
    use rstest::rstest;
    use serde::Deserialize;
    use serde_json::json;

    use super::*;
    use crate::snapshot::{KnownNames, TableRow};
    use crate::store::Store;
    use crate::test_support::{TestRow, TestRows, bundled_rows, names, snapshot};

    /// Main's answers for realistic `/v1/messages` calls, captured once, see `captured_from` in the file
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
    struct Registered {
        fields: Fields,
        registered_fields: BTreeSet<String>,
    }

    #[derive(Deserialize)]
    struct Case {
        name: String,
        identity: FixtureIdentity,
        billed_key: Option<String>,
        billed_provider: Option<String>,
        tried: Vec<String>,
        after_tried: String,
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
                renamed_models: FIXTURE.known_names.renamed_models.iter().cloned().collect(),
            },
        )
    });

    fn call(provider: &str, request: &str, response: &str) -> CallIdentity {
        CallIdentity {
            provider: provider.to_owned(),
            request_model: Some(request.to_owned()),
            response_model: Some(response.to_owned()),
            ..CallIdentity::default()
        }
    }

    const REGISTERED: &str = "us.anthropic.claude-opus-5-5";

    fn keys(rows: &[RowRef]) -> Vec<&str> {
        rows.iter().map(|row| row.key.as_str()).collect()
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
    fn candidates_are_the_rows_main_tries_in_its_order(#[case] name: &str) {
        let case = FIXTURE
            .cases
            .iter()
            .find(|case| case.name == name)
            .expect("the case is in the fixture");
        let identity = &case.identity;
        let found = MAIN.candidates(&CallIdentity {
            provider: identity.provider.clone(),
            request_model: identity.request_model.clone(),
            response_model: identity.response_model.clone(),
            base_model: identity.base_model.clone(),
            custom_pricing: identity.custom_pricing,
            deployment_id: identity.deployment_id.clone(),
            region: identity.region.clone(),
            ..CallIdentity::default()
        });
        let rows_main_tried: Vec<&str> = case
            .tried
            .iter()
            .map(String::as_str)
            .take_while(|key| MAIN.table.contains(key))
            .collect();
        let main_went_past_rows =
            rows_main_tried.len() < case.tried.len() || case.after_tried == "returned";
        let main_billed = case
            .billed_key
            .as_deref()
            .zip(case.billed_provider.as_deref())
            .filter(|(key, _)| MAIN.table.contains(key));

        assert_eq!(keys(&found.rows), rows_main_tried);
        assert_eq!(
            matches!(found.then, AfterRows::Deferred(_)),
            main_went_past_rows
        );
        assert_eq!(
            found
                .rows
                .first()
                .map(|row| (row.key.as_str(), row.provider.as_str())),
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

        let found = snapshot.candidates(&CallIdentity {
            custom_pricing,
            ..call(
                "bedrock",
                &registration.request_model,
                &registration.response_model,
            )
        });

        assert_eq!(
            found
                .rows
                .iter()
                .map(|row| (row.key.as_str(), row.provenance))
                .collect::<Vec<_>>(),
            case.tried
                .iter()
                .map(|key| (key.as_str(), RowProvenance::Registered))
                .collect::<Vec<_>>()
        );
    }

    #[rstest]
    fn a_custom_priced_deployment_row_comes_first_as_registered() {
        let found = MAIN.candidates(&CallIdentity {
            custom_pricing: true,
            deployment_id: Some("dep-claude-custom".to_owned()),
            ..call("anthropic", "claude-opus-5-5", "claude-opus-5-5")
        });

        assert_eq!(
            found
                .rows
                .iter()
                .map(|row| (row.key.as_str(), row.provenance))
                .collect::<Vec<_>>(),
            [
                ("dep-claude-custom", RowProvenance::Registered),
                ("claude-opus-5-5", RowProvenance::Catalog),
                ("claude-opus-5-5", RowProvenance::Catalog),
            ]
        );
    }

    #[rstest]
    #[case::other_provider(call("openai", "gpt-4o", "gpt-4o"), Deferral::Provider)]
    #[case::bedrock_inference_profile(
        CallIdentity {
            invoked_model_id: Some("arn:aws:bedrock:us-east-1:1:application-inference-profile/x".to_owned()),
            ..call("bedrock", "us.anthropic.claude-opus-5-5", "claude-opus-5-5")
        },
        Deferral::BedrockPricing
    )]
    #[case::azure_model_router(
        call("azure_ai", "model-router", "claude-opus-5-5"),
        Deferral::ModelName
    )]
    #[case::renamed_before_lookup(
        call("azure_ai", "gpt-35-turbo", "claude-opus-5-5"),
        Deferral::ModelName
    )]
    #[case::hidden_response_model(
        CallIdentity {
            provider_response_model: Some("claude-opus-5-5".to_owned()),
            ..call("anthropic", "claude-opus-5-5", "claude-opus-5-5")
        },
        Deferral::HiddenParams
    )]
    #[case::base_model_its_provider_cannot_price(
        CallIdentity {
            base_model: Some("claude-opus-9-9".to_owned()),
            ..call("azure_ai", "prod-opus", "prod-opus")
        },
        Deferral::BaseModelProvider
    )]
    #[case::vertex_character_pricing(
        CallIdentity {
            custom_pricing: true,
            deployment_id: Some("dep-bedrock-custom".to_owned()),
            ..call("vertex_ai", "claude-opus-5-5", "claude-opus-5-5")
        },
        Deferral::ModelName
    )]
    fn calls_main_prices_another_way_are_deferred(
        #[case] call: CallIdentity,
        #[case] deferral: Deferral,
    ) {
        assert_eq!(
            MAIN.candidates(&call),
            CandidateRows {
                rows: Vec::new(),
                then: AfterRows::Deferred(deferral),
            }
        );
    }

    #[rstest]
    #[case::priced_registration_bills_every_candidate(
        json!({"input_cost_per_token": 1e-6}),
        &[(REGISTERED, RowProvenance::Registered); 3],
        AfterRows::Exhausted
    )]
    #[case::cache_only_registration(
        json!({"cache_read_input_token_cost": 1e-7}),
        &[],
        AfterRows::Deferred(Deferral::BedrockPricing)
    )]
    #[case::capability_registration_is_not_a_price(
        json!({"supports_vision": true, "mode": "chat"}),
        &[(REGISTERED, RowProvenance::Catalog); 2],
        AfterRows::Exhausted
    )]
    fn a_bedrock_register_model_price_is_billed_like_main(
        #[case] registered: Value,
        #[case] expected: &[(&str, RowProvenance)],
        #[case] then: AfterRows,
    ) {
        let Value::Object(registered) = registered else {
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

        let found = snapshot.candidates(&call("bedrock", REGISTERED, "claude-opus-5-5"));

        assert_eq!(
            found
                .rows
                .iter()
                .map(|row| (row.key.as_str(), row.provenance))
                .collect::<Vec<_>>(),
            expected
        );
        assert_eq!(found.then, then);
    }
}
