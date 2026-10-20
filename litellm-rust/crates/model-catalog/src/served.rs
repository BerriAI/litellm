use std::collections::HashMap;
use std::sync::Arc;

use indexmap::IndexMap;

use crate::catalog::Catalog;
use crate::error::Rejected;
use crate::model_info::{Fields, Model};

/// The registry document this crate was built with.
pub const BUNDLED_REGISTRY: &[u8] =
    include_bytes!("../../../../model_prices_and_context_window.json");

/// Per key, the owners in the order they last wrote it, each with the fields it set.
#[derive(Debug, Default)]
struct Overrides {
    by_key: IndexMap<String, IndexMap<String, Fields>>,
    lowercase_keys: HashMap<String, String>,
}

impl Overrides {
    fn indexed(by_key: IndexMap<String, IndexMap<String, Fields>>) -> Self {
        let lowercase_keys = by_key
            .keys()
            .map(|key| (key.to_lowercase(), key.clone()))
            .collect();
        Self {
            by_key,
            lowercase_keys,
        }
    }
}

/// The catalog a process serves at runtime: a registry, replaced by a price reload, with the
/// fields each owner registered on top of it.
///
/// Immutable: every change returns a new value and the host decides how to swap the one it
/// serves. An owner keeps only the fields it set, so a new registry shows through every field
/// nobody overrode, and unregistering an owner takes back exactly what it added.
#[derive(Clone, Debug, Default)]
pub struct ServedCatalog {
    registry: Option<Arc<Catalog>>,
    overrides: Arc<Overrides>,
}

impl ServedCatalog {
    pub fn new(registry: Arc<Catalog>) -> Self {
        Self {
            registry: Some(registry),
            overrides: Arc::default(),
        }
    }

    pub fn registry(&self) -> Option<&Arc<Catalog>> {
        self.registry.as_ref()
    }

    /// `registry` served underneath every registration made so far, rejected when a
    /// registration no longer reads cleanly on top of it.
    pub fn with_registry(&self, registry: Arc<Catalog>) -> Result<Self, Rejected> {
        let served = Self {
            registry: Some(registry),
            overrides: Arc::clone(&self.overrides),
        };
        served.checked(self.overrides.by_key.keys())
    }

    /// `entries` written on behalf of `owner`. Each merges over what `owner` set on that key
    /// before and becomes the latest write to it. Rejected when a merged entry has a known key
    /// with a bad value.
    pub fn register(
        &self,
        owner: &str,
        entries: impl IntoIterator<Item = (String, Fields)>,
    ) -> Result<Self, Rejected> {
        let mut by_key = self.overrides.by_key.clone();
        let mut written_keys = Vec::new();
        for (key, fields) in entries {
            let key = self.write_key(key);
            let owners = by_key.entry(key.clone()).or_default();
            let mut written = owners.shift_remove(owner).unwrap_or_default();
            written.extend(fields);
            owners.insert(owner.to_owned(), written);
            written_keys.push(key);
        }
        self.with_overrides(by_key).checked(&written_keys)
    }

    /// The entry served under exactly `key`, read into types.
    pub fn model(&self, key: &str) -> Option<Result<Model, Rejected>> {
        self.entry(key)
            .map(|fields| Model::from_fields(key, &fields).map_err(Rejected))
    }

    fn checked<'a>(self, keys: impl IntoIterator<Item = &'a String>) -> Result<Self, Rejected> {
        let problems: Vec<_> = keys
            .into_iter()
            .filter_map(|key| self.model(key)?.err())
            .flat_map(|Rejected(problems)| problems)
            .collect();
        if problems.is_empty() {
            Ok(self)
        } else {
            Err(Rejected(problems))
        }
    }

    /// Every field `owner` registered taken back. The other owners' writes stay in their order.
    pub fn unregister(&self, owner: &str) -> Self {
        let by_key = self
            .overrides
            .by_key
            .iter()
            .filter_map(|(key, owners)| {
                let kept: IndexMap<String, Fields> = owners
                    .iter()
                    .filter(|(writer, _)| writer.as_str() != owner)
                    .map(|(writer, fields)| (writer.clone(), fields.clone()))
                    .collect();
                (!kept.is_empty()).then(|| (key.clone(), kept))
            })
            .collect();
        self.with_overrides(by_key)
    }

    fn with_overrides(&self, by_key: IndexMap<String, IndexMap<String, Fields>>) -> Self {
        Self {
            registry: self.registry.clone(),
            overrides: Arc::new(Overrides::indexed(by_key)),
        }
    }

    /// The canonical registry key `key` names exactly, so a write to an alias reaches its group.
    fn registry_key(&self, key: &str) -> Option<&str> {
        self.registry
            .as_deref()?
            .lookup(key)
            .filter(|found| found.matched_key == key)
            .map(|found| found.canonical_key)
    }

    fn write_key(&self, key: String) -> String {
        self.registry_key(&key).map_or(key, str::to_owned)
    }

    pub fn contains(&self, key: &str) -> bool {
        self.overrides.by_key.contains_key(key) || self.registry_key(key).is_some()
    }

    /// The key `name` names, exactly or else case-insensitively, aliases included.
    pub fn key_for<'a>(&'a self, name: &'a str) -> Option<&'a str> {
        if self.contains(name) {
            return Some(name);
        }
        if let Some(key) = self.overrides.lowercase_keys.get(&name.to_lowercase()) {
            return Some(key);
        }
        self.registry
            .as_deref()?
            .lookup(name)
            .map(|found| found.matched_key)
    }

    /// The entry served under exactly `key`: its registry row, then each owner's fields in order.
    pub fn entry(&self, key: &str) -> Option<Fields> {
        let canonical = self.registry_key(key);
        let row = canonical.and_then(|canonical| {
            self.registry
                .as_deref()?
                .lookup(canonical)
                .map(|found| found.entry.fields().clone())
        });
        let owners = self.overrides.by_key.get(canonical.unwrap_or(key));
        if row.is_none() && owners.is_none() {
            return None;
        }
        let mut fields = row.unwrap_or_default();
        for written in owners.into_iter().flat_map(IndexMap::values) {
            fields.extend(
                written
                    .iter()
                    .map(|(name, value)| (name.clone(), value.clone())),
            );
        }
        Some(fields)
    }

    /// Every served key once: registry models and aliases, then keys only registrations add.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        let registry_keys = self.registry.as_deref().into_iter().flat_map(|registry| {
            registry
                .model_names()
                .chain(registry.aliases().keys().map(String::as_str))
        });
        let registered_only = self
            .overrides
            .by_key
            .keys()
            .map(String::as_str)
            .filter(|key| self.registry_key(key).is_none());
        registry_keys.chain(registered_only)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::Provenance;
    use crate::prices::{Clock, Meter, PriceKey};
    use crate::pricing::Usd;
    use crate::rate::{Charge, TokenKind, Unit};
    use rstest::{fixture, rstest};
    use serde_json::{Value, json};
    use std::collections::BTreeSet;

    trait Registering {
        fn registered(&self, owner: &str, entries: Vec<(String, Fields)>) -> ServedCatalog;
        fn reloaded(&self, registry: Arc<Catalog>) -> ServedCatalog;
    }

    impl Registering for ServedCatalog {
        fn registered(&self, owner: &str, entries: Vec<(String, Fields)>) -> ServedCatalog {
            self.register(owner, entries).unwrap()
        }
        fn reloaded(&self, registry: Arc<Catalog>) -> ServedCatalog {
            self.with_registry(registry).unwrap()
        }
    }

    const SDK: &str = "sdk";
    const DEPLOYMENT_A: &str = "router:1:dep-a";
    const DEPLOYMENT_B: &str = "router:1:dep-b";
    const OTHER_ROUTER_A: &str = "router:2:dep-a";

    fn fields(value: Value) -> Fields {
        let Value::Object(fields) = value else {
            panic!("test entry must be an object")
        };
        fields
    }

    fn registry(document: Value) -> Arc<Catalog> {
        Arc::new(
            Catalog::parse(
                &serde_json::to_vec(&document).unwrap(),
                Provenance::default(),
            )
            .unwrap(),
        )
    }

    fn entries(value: Value) -> Vec<(String, Fields)> {
        let Value::Object(entries) = value else {
            panic!("registrations must be an object")
        };
        entries
            .into_iter()
            .map(|(key, value)| (key, fields(value)))
            .collect()
    }

    fn keys(catalog: &ServedCatalog) -> BTreeSet<&str> {
        catalog.keys().collect()
    }

    #[fixture]
    fn served() -> ServedCatalog {
        ServedCatalog::new(registry(json!({
            "gpt-4o": {
                "litellm_provider": "openai",
                "input_cost_per_token": 1.0,
                "output_cost_per_token": 4.0,
                "aliases": ["gpt-4o-latest"]
            },
            "hosted_vllm/llama": {"litellm_provider": "hosted_vllm", "mode": "chat"}
        })))
    }

    #[rstest]
    fn an_sdk_override_changes_only_the_fields_it_sets(served: ServedCatalog) {
        let catalog = served.registered(
            SDK,
            entries(json!({"gpt-4o": {"input_cost_per_token": 2.0}})),
        );
        assert_eq!(
            catalog.entry("gpt-4o"),
            Some(fields(json!({
                "litellm_provider": "openai",
                "input_cost_per_token": 2.0,
                "output_cost_per_token": 4.0
            })))
        );
    }

    #[rstest]
    fn an_sdk_registration_adds_a_model_the_registry_lacks(served: ServedCatalog) {
        let catalog = served.registered(SDK, entries(json!({"my-model": {"mode": "chat"}})));
        assert_eq!(
            catalog.entry("my-model"),
            Some(fields(json!({"mode": "chat"})))
        );
        assert!(keys(&catalog).contains("my-model"));
    }

    #[rstest]
    fn an_owner_registering_a_key_again_merges_over_what_it_set_before(served: ServedCatalog) {
        let catalog = served
            .registered(
                SDK,
                entries(json!({"my-model": {"mode": "chat", "input_cost_per_token": 1.0}})),
            )
            .registered(
                SDK,
                entries(json!({"my-model": {"input_cost_per_token": 2.0}})),
            );
        assert_eq!(
            catalog.entry("my-model"),
            Some(fields(json!({"mode": "chat", "input_cost_per_token": 2.0})))
        );
    }

    #[rstest]
    #[case::nothing_unregistered(&[], 4.0)]
    #[case::latest_writer_unregistered(&[SDK], 3.0)]
    #[case::earlier_writer_unregistered(&[DEPLOYMENT_A], 4.0)]
    #[case::every_writer_unregistered(&[SDK, DEPLOYMENT_A], 1.0)]
    #[case::unknown_owner_unregistered(&["router:9:missing"], 4.0)]
    fn the_latest_remaining_write_to_a_field_wins(
        served: ServedCatalog,
        #[case] unregistered: &[&str],
        #[case] price: f64,
    ) {
        let catalog = served
            .registered(
                SDK,
                entries(json!({"gpt-4o": {"input_cost_per_token": 2.0}})),
            )
            .registered(
                DEPLOYMENT_A,
                entries(json!({"gpt-4o": {"input_cost_per_token": 3.0}})),
            )
            .registered(
                SDK,
                entries(json!({"gpt-4o": {"input_cost_per_token": 4.0}})),
            );
        let remaining = unregistered
            .iter()
            .fold(catalog, |catalog, owner| catalog.unregister(owner));
        assert_eq!(
            remaining.entry("gpt-4o").unwrap()["input_cost_per_token"],
            json!(price)
        );
    }

    #[rstest]
    fn a_reload_keeps_every_override_and_refreshes_every_other_field(served: ServedCatalog) {
        let reloaded = served
            .registered(SDK, entries(json!({"gpt-4o": {"input_cost_per_token": 2.0}})))
            .registered(DEPLOYMENT_A, entries(json!({"dep-a": {"input_cost_per_token": 5.0}})))
            .reloaded(registry(json!({
                "gpt-4o": {"litellm_provider": "openai", "input_cost_per_token": 1.5, "output_cost_per_token": 6.0}
            })));
        assert_eq!(
            reloaded.entry("gpt-4o"),
            Some(fields(json!({
                "litellm_provider": "openai",
                "input_cost_per_token": 2.0,
                "output_cost_per_token": 6.0
            })))
        );
        assert_eq!(
            reloaded.entry("dep-a"),
            Some(fields(json!({"input_cost_per_token": 5.0})))
        );
    }

    #[rstest]
    fn a_reload_that_drops_a_model_keeps_only_what_was_registered_for_it(served: ServedCatalog) {
        let reloaded = served
            .registered(
                SDK,
                entries(json!({"hosted_vllm/llama": {"max_tokens": 10}})),
            )
            .reloaded(registry(json!({"gpt-4o": {"litellm_provider": "openai"}})));
        assert_eq!(
            reloaded.entry("hosted_vllm/llama"),
            Some(fields(json!({"max_tokens": 10})))
        );
        assert_eq!(reloaded.entry("gpt-4o-latest"), None);
    }

    #[rstest]
    fn deleting_a_deployment_removes_its_entries_and_its_share_of_a_shared_key(
        served: ServedCatalog,
    ) {
        let catalog = served
            .registered(
                DEPLOYMENT_A,
                entries(json!({"dep-a": {"input_cost_per_token": 5.0}, "hosted_vllm/llama": {"supports_vision": true}})),
            )
            .registered(
                DEPLOYMENT_B,
                entries(json!({"dep-b": {"input_cost_per_token": 6.0}, "hosted_vllm/llama": {"max_tokens": 10}})),
            )
            .unregister(DEPLOYMENT_A);
        assert_eq!(catalog.entry("dep-a"), None);
        assert_eq!(catalog.key_for("DEP-A"), None);
        assert_eq!(
            catalog.entry("hosted_vllm/llama"),
            Some(fields(
                json!({"litellm_provider": "hosted_vllm", "mode": "chat", "max_tokens": 10})
            ))
        );
        assert_eq!(
            catalog.entry("dep-b"),
            Some(fields(json!({"input_cost_per_token": 6.0})))
        );
    }

    #[rstest]
    fn upserting_a_deployment_drops_the_fields_its_new_config_no_longer_sets(
        served: ServedCatalog,
    ) {
        let catalog = served
            .registered(
                DEPLOYMENT_A,
                entries(json!({"dep-a": {"input_cost_per_token": 5.0, "max_tokens": 10}})),
            )
            .unregister(DEPLOYMENT_A)
            .registered(
                DEPLOYMENT_A,
                entries(json!({"dep-a": {"input_cost_per_token": 7.0}})),
            );
        assert_eq!(
            catalog.entry("dep-a"),
            Some(fields(json!({"input_cost_per_token": 7.0})))
        );
    }

    #[rstest]
    fn a_deployment_registered_before_a_reload_is_deleted_after_it(served: ServedCatalog) {
        let catalog = served
            .registered(
                DEPLOYMENT_A,
                entries(json!({"dep-a": {"mode": "chat"}, "gpt-4o": {"max_tokens": 10}})),
            )
            .reloaded(registry(json!({"gpt-4o": {"litellm_provider": "openai"}})))
            .unregister(DEPLOYMENT_A);
        assert_eq!(keys(&catalog), BTreeSet::from(["gpt-4o"]));
        assert_eq!(
            catalog.entry("gpt-4o"),
            Some(fields(json!({"litellm_provider": "openai"})))
        );
    }

    #[rstest]
    fn two_routers_serving_the_same_deployment_id_release_it_independently(served: ServedCatalog) {
        let catalog = served
            .registered(
                DEPLOYMENT_A,
                entries(json!({"dep-a": {"input_cost_per_token": 5.0}})),
            )
            .registered(
                OTHER_ROUTER_A,
                entries(json!({"dep-a": {"input_cost_per_token": 5.0}})),
            )
            .unregister(DEPLOYMENT_A);
        assert_eq!(
            catalog.entry("dep-a"),
            Some(fields(json!({"input_cost_per_token": 5.0})))
        );
    }

    #[rstest]
    #[case::through_the_canonical_key("gpt-4o")]
    #[case::through_an_alias("gpt-4o-latest")]
    fn a_registration_on_any_alias_reaches_every_name_in_the_group(
        served: ServedCatalog,
        #[case] key: &str,
    ) {
        let catalog = served.registered(SDK, entries(json!({key: {"input_cost_per_token": 2.0}})));
        assert_eq!(catalog.entry("gpt-4o"), catalog.entry("gpt-4o-latest"));
        assert_eq!(
            catalog.entry("gpt-4o").unwrap()["input_cost_per_token"],
            json!(2.0)
        );
    }

    #[rstest]
    #[case::registry_exact("gpt-4o", Some("gpt-4o"))]
    #[case::registry_case_insensitive("GPT-4o", Some("gpt-4o"))]
    #[case::alias("gpt-4o-latest", Some("gpt-4o-latest"))]
    #[case::alias_case_insensitive("GPT-4O-LATEST", Some("gpt-4o-latest"))]
    #[case::registered_exact("My-Model", Some("My-Model"))]
    #[case::registered_case_insensitive("my-model", Some("My-Model"))]
    #[case::unknown("gpt-5", None)]
    fn key_for_resolves_exact_then_case_insensitive_names(
        served: ServedCatalog,
        #[case] name: &str,
        #[case] expected: Option<&str>,
    ) {
        let catalog = served.registered(SDK, entries(json!({"My-Model": {"mode": "chat"}})));
        assert_eq!(catalog.key_for(name), expected);
    }

    #[rstest]
    fn every_served_name_is_listed_once(served: ServedCatalog) {
        let catalog = served.registered(
            SDK,
            entries(json!({"gpt-4o": {"input_cost_per_token": 2.0}, "my-model": {"mode": "chat"}})),
        );
        assert_eq!(
            catalog.keys().collect::<Vec<_>>().len(),
            keys(&catalog).len(),
            "a key is listed twice"
        );
        assert_eq!(
            keys(&catalog),
            BTreeSet::from(["gpt-4o", "gpt-4o-latest", "hosted_vllm/llama", "my-model"])
        );
    }

    #[rstest]
    fn a_catalog_already_handed_out_never_changes(served: ServedCatalog) {
        let before = served.registered(
            SDK,
            entries(json!({"gpt-4o": {"input_cost_per_token": 2.0}})),
        );
        let snapshot = before.entry("gpt-4o");
        let _after = before
            .registered(
                DEPLOYMENT_A,
                entries(json!({"gpt-4o": {"input_cost_per_token": 3.0}})),
            )
            .unregister(SDK)
            .reloaded(registry(
                json!({"claude": {"litellm_provider": "anthropic", "mode": "chat"}}),
            ));
        assert_eq!(before.entry("gpt-4o"), snapshot);
    }

    #[rstest]
    fn a_catalog_without_a_registry_serves_only_its_registrations() {
        let catalog = ServedCatalog::default()
            .registered(SDK, entries(json!({"My-Model": {"mode": "chat"}})));
        assert_eq!(keys(&catalog), BTreeSet::from(["My-Model"]));
        assert_eq!(catalog.key_for("my-model"), Some("My-Model"));
    }

    fn input_token_price(model: &Model) -> Option<f64> {
        model
            .prices
            .get(&PriceKey {
                meter: Meter::Charge(Charge::Input(Unit::Token(TokenKind::Text))),
                tier: None,
                band: None,
                clock: Clock::Peak,
            })
            .map(Usd::get)
    }

    #[rstest]
    fn the_typed_view_reads_the_merged_entry(served: ServedCatalog) {
        let catalog = served.registered(
            SDK,
            entries(json!({"gpt-4o-latest": {"input_cost_per_token": "0.5"}})),
        );
        let model = catalog.model("gpt-4o").unwrap().unwrap();
        assert_eq!(input_token_price(&model), Some(0.5));
        assert_eq!(model.attributes.litellm_provider.as_deref(), Some("openai"));
    }

    #[rstest]
    fn a_registration_with_a_bad_known_value_is_rejected_by_model_and_key(served: ServedCatalog) {
        let rejected = served
            .register(
                DEPLOYMENT_A,
                entries(
                    json!({"dep-a": {"mode": "chatt", "input_cost_per_token": "abc", "id": "x"}}),
                ),
            )
            .unwrap_err()
            .to_string();
        assert!(
            rejected.contains("dep-a.mode: unknown variant `chatt`"),
            "{rejected}"
        );
        assert!(
            rejected.contains("dep-a.input_cost_per_token"),
            "{rejected}"
        );
        assert!(!rejected.contains("dep-a.id"), "{rejected}");
    }

    #[rstest]
    fn a_registration_that_prices_a_registry_band_twice_is_rejected(served: ServedCatalog) {
        let catalog = served.registered(
            SDK,
            entries(json!({"gpt-4o": {"input_cost_per_token_above_200k_tokens": 2.0}})),
        );
        let rejected = catalog.with_registry(registry(json!({"gpt-4o": {
            "litellm_provider": "openai",
            "tiered_pricing": [{"range": [200000, 400000], "input_cost_per_token": 3.0}]
        }})));
        assert!(rejected.unwrap_err().to_string().contains(
            "input_cost_per_token_above_200k_tokens and tiered_pricing[0].input_cost_per_token"
        ));
    }

    #[rstest]
    fn a_rejected_registration_leaves_the_catalog_unchanged(served: ServedCatalog) {
        assert!(
            served
                .register(SDK, entries(json!({"gpt-4o": {"max_tokens": -1}})))
                .is_err()
        );
        assert_eq!(served.entry("gpt-4o").unwrap().get("max_tokens"), None);
    }

    #[rstest]
    #[ignore]
    fn the_bundled_registry_parses() {
        assert!(
            Catalog::parse(BUNDLED_REGISTRY, Provenance::default())
                .unwrap()
                .model_count()
                > 0
        );
    }
}
