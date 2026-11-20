//! completion_cost's candidate walk in litellm-cost over this catalog's lookups, checked against main's answers in
//! `cost_candidates.json`. The walk lives in litellm-cost, which cannot see a catalog, so these run here
//!
//! `cost_candidates.json` is generated and has no generator in the tree. It was captured once from
//! `litellm.completion_cost(call_type="anthropic_messages")` at main 532df6ef85 plus #51566, with
//! `LITELLM_LOCAL_MODEL_COST_MAP=True` and `NativePrices` spied for the row billed and, with every pricing attempt
//! forced to fail, each row tried in order. Recapture it only against Python's own walk: from #52484 up,
//! completion_cost goes through this walk, so a capture there would record Rust's answers

use std::collections::BTreeSet;
use std::sync::LazyLock;

use indexmap::IndexMap;
use rstest::rstest;
use serde::Deserialize;
use serde_json::json;

use litellm_cost::candidates::{
    AfterRows, CallIdentity, CandidateRows, RustRefusal, RowProvenance, Tried, candidates,
};
use serde_json::Value;

use crate::CostMap;
use crate::store::Store;
use crate::test_support::{TestRow, TestRows, bundled_rows, context, names};
use crate::view::{CatalogView, CostLookup, KnownNames, PricingContext};
use crate::wire::Fields;

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

static FIXTURE_CONTEXT: LazyLock<PricingContext> = LazyLock::new(|| {
    context(KnownNames {
        providers: FIXTURE.known_names.providers.iter().cloned().collect(),
        renamed_models: FIXTURE.known_names.renamed_models.iter().cloned().collect(),
    })
});

static MAIN_TABLE: LazyLock<Store<TestRows>> = LazyLock::new(|| {
    let registered = FIXTURE.registered.iter().map(|(key, row)| {
        (
            key.clone(),
            TestRow::new(row.fields.clone(), row.registered_fields.clone()),
        )
    });
    Store::new(TestRows(
        bundled_rows().0.into_iter().chain(registered).collect(),
    ))
});

/// Main's table as the fixture captured it: the bundled document plus the rows its router registered
fn main() -> CostLookup<'static, Store<TestRows>> {
    CostLookup::new(&MAIN_TABLE, &FIXTURE_CONTEXT)
}

fn fixture_call(identity: &FixtureIdentity) -> CallIdentity {
    CallIdentity {
        provider: identity.provider.clone(),
        request_model: identity.request_model.clone(),
        response_model: identity.response_model.clone(),
        base_model: identity.base_model.clone(),
        custom_pricing: identity.custom_pricing,
        deployment_id: identity.deployment_id.clone(),
        region: identity.region.clone(),
        ..CallIdentity::default()
    }
}

fn call(provider: &str, request: &str, response: &str) -> CallIdentity {
    CallIdentity {
        provider: provider.to_owned(),
        request_model: Some(request.to_owned()),
        response_model: Some(response.to_owned()),
        ..CallIdentity::default()
    }
}

const REGISTERED: &str = "us.anthropic.claude-opus-5-5";

fn keys(rows: &[Tried]) -> Vec<&str> {
    rows.iter().map(|tried| tried.row.key.as_str()).collect()
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
    let found = candidates(&main(), &fixture_call(&case.identity));
    let rows_main_tried: Vec<&str> = case
        .tried
        .iter()
        .map(String::as_str)
        .take_while(|key| CatalogView::contains(&*MAIN_TABLE, key))
        .collect();
    let main_went_past_rows =
        rows_main_tried.len() < case.tried.len() || case.after_tried == "returned";
    let main_billed = case
        .billed_key
        .as_deref()
        .zip(case.billed_provider.as_deref())
        .filter(|(key, _)| CatalogView::contains(&*MAIN_TABLE, key));

    assert_eq!(keys(&found.rows), rows_main_tried);
    assert_eq!(
        matches!(found.then, AfterRows::Deferred(_)),
        main_went_past_rows
    );
    assert_eq!(
        found
            .rows
            .first()
            .map(|tried| (tried.row.key.as_str(), tried.row.provider.as_str())),
        main_billed
    );
}

/// The Rust host's `Catalog` and Python's table of the same document resolve every call to the same rows
#[rstest]
fn the_rust_hosts_catalog_resolves_like_pythons_table_of_the_same_document() {
    let cost_map = CostMap::bundled();
    let python = Store::new(bundled_rows());
    let resolved = |table: &dyn CatalogView| -> Vec<CandidateRows> {
        let lookup = CostLookup::new(table, &FIXTURE_CONTEXT);
        FIXTURE
            .cases
            .iter()
            .map(|case| candidates(&lookup, &fixture_call(&case.identity)))
            .collect()
    };

    let from_catalog = resolved(cost_map.catalog());

    assert_eq!(from_catalog, resolved(&python));
    assert!(from_catalog.iter().any(|found| !found.rows.is_empty()));
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
    let table = Store::new(TestRows(
        rows.0
            .into_iter()
            .chain([(registration.key.clone(), written)])
            .collect(),
    ));
    let context = context(names(&["anthropic", "bedrock"]));

    let found = candidates(
        &CostLookup::new(&table, &context),
        &CallIdentity {
            custom_pricing,
            ..call(
                "bedrock",
                &registration.request_model,
                &registration.response_model,
            )
        },
    );

    assert_eq!(
        found
            .rows
            .iter()
            .map(|tried| (tried.row.key.as_str(), tried.row.provenance))
            .collect::<Vec<_>>(),
        case.tried
            .iter()
            .map(|key| (key.as_str(), RowProvenance::Registered))
            .collect::<Vec<_>>()
    );
}

#[rstest]
fn a_custom_priced_deployment_row_comes_first_as_registered() {
    let found = candidates(
        &main(),
        &CallIdentity {
            custom_pricing: true,
            deployment_id: Some("dep-claude-custom".to_owned()),
            ..call("anthropic", "claude-opus-5-5", "claude-opus-5-5")
        },
    );

    assert_eq!(
        found
            .rows
            .iter()
            .map(|tried| (tried.row.key.as_str(), tried.row.provenance))
            .collect::<Vec<_>>(),
        [
            ("dep-claude-custom", RowProvenance::Registered),
            ("claude-opus-5-5", RowProvenance::Catalog),
            ("claude-opus-5-5", RowProvenance::Catalog),
        ]
    );
}

#[rstest]
#[case::other_provider(call("openai", "gpt-4o", "gpt-4o"), RustRefusal::Provider)]
#[case::bedrock_inference_profile(
    CallIdentity {
        invoked_model_id: Some("arn:aws:bedrock:us-east-1:1:application-inference-profile/x".to_owned()),
        ..call("bedrock", "us.anthropic.claude-opus-5-5", "claude-opus-5-5")
    },
    RustRefusal::BedrockPricing
)]
#[case::azure_model_router(
    call("azure_ai", "model-router", "claude-opus-5-5"),
    RustRefusal::ModelName
)]
#[case::renamed_before_lookup(
    call("azure_ai", "gpt-35-turbo", "claude-opus-5-5"),
    RustRefusal::ModelName
)]
#[case::hidden_response_model(
    CallIdentity {
        provider_response_model: Some("claude-opus-5-5".to_owned()),
        ..call("anthropic", "claude-opus-5-5", "claude-opus-5-5")
    },
    RustRefusal::HiddenParams
)]
#[case::base_model_its_provider_cannot_price(
    CallIdentity {
        base_model: Some("claude-opus-9-9".to_owned()),
        ..call("azure_ai", "prod-opus", "prod-opus")
    },
    RustRefusal::BaseModelProvider
)]
#[case::vertex_character_pricing(
    CallIdentity {
        custom_pricing: true,
        deployment_id: Some("dep-bedrock-custom".to_owned()),
        ..call("vertex_ai", "claude-opus-5-5", "claude-opus-5-5")
    },
    RustRefusal::ModelName
)]
fn calls_main_prices_another_way_are_deferred(
    #[case] call: CallIdentity,
    #[case] deferral: RustRefusal,
) {
    assert_eq!(
        candidates(&main(), &call),
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
    AfterRows::Deferred(RustRefusal::BedrockPricing)
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
    let table = Store::new(TestRows(
        rows.0
            .into_iter()
            .chain([(REGISTERED.to_owned(), written)])
            .collect(),
    ));
    let context = context(names(&["anthropic", "bedrock"]));

    let found = candidates(
        &CostLookup::new(&table, &context),
        &call("bedrock", REGISTERED, "claude-opus-5-5"),
    );

    assert_eq!(
        found
            .rows
            .iter()
            .map(|tried| (tried.row.key.as_str(), tried.row.provenance))
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(found.then, then);
}
