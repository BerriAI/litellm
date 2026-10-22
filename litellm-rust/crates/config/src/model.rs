use litellm_router_types::LitellmParams;
use serde::Deserialize;

use crate::{Object, Raw, Spelled};

#[derive(Clone, Debug, Deserialize)]
pub struct Model {
    pub model_name: String,
    pub litellm_params: LitellmParams,
    #[serde(default)]
    pub model_info: ModelInfo,
    pub blocked: Option<bool>,
    pub rpm: Option<Spelled<u64>>,
    pub tpm: Option<Spelled<u64>>,
    #[serde(flatten)]
    pub additional_fields: Object,
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct ModelInfo {
    pub id: Option<ModelId>,
    pub db_model: Option<Spelled<bool>>,
    pub updated_at: Option<Raw>,
    pub created_at: Option<Raw>,
    pub updated_by: Option<String>,
    pub created_by: Option<String>,
    pub base_model: Option<String>,
    pub pricing_provider: Option<String>,
    pub tier: Option<String>,
    pub team_id: Option<String>,
    pub team_public_model_name: Option<String>,
    pub member_auto_router: Option<Spelled<bool>>,
    pub blocked: Option<Spelled<bool>>,
    pub discoverable: Option<Spelled<bool>>,
    pub access_windows: Option<Box<[AccessWindow]>>,
    pub ptu_count: Option<Spelled<i64>>,
    pub cost_per_ptu_per_hour: Option<Spelled<f64>>,
    pub ptu_effective_from: Option<Raw>,
    pub ptu_effective_to: Option<Raw>,
    pub ptu_shares: Option<Object>,
    pub allow_fail_open: Option<Spelled<bool>>,
    pub enable_tag_filtering: Option<Spelled<bool>>,
    pub internal_router_model: Option<Spelled<bool>>,
    pub max_embedding_batch_size: Option<Spelled<i64>>,
    pub input_cost_per_token: Option<Spelled<f64>>,
    pub output_cost_per_token: Option<Spelled<f64>>,
    pub input_cost_per_character: Option<Spelled<f64>>,
    pub output_cost_per_character: Option<Spelled<f64>>,
    pub cache_read_input_token_cost: Option<Spelled<f64>>,
    pub explicit_cache_read_input_token_cost: Option<Spelled<f64>>,
    pub cache_creation_input_token_cost: Option<Spelled<f64>>,
    pub cache_creation_input_token_cost_above_1hr: Option<Spelled<f64>>,
    pub tiered_pricing: Option<Box<[Object]>>,
    pub mode: Option<String>,
    pub access_groups: Option<Box<[String]>>,
    pub supported_environments: Option<Box<[String]>>,
    #[serde(flatten)]
    pub additional_fields: Object,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum ModelId {
    Text(String),
    Number(i64),
}

#[macro_rules_attribute::apply(crate::config_section)]
pub struct AccessWindow {
    pub start: Option<Raw>,
    pub end: Option<Raw>,
    pub timezone: Option<String>,
    pub team_ids: Option<Box<[String]>>,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::{Config, Model, ModelId, Spelled};

    fn model(entry: &str) -> Model {
        let config = Config::from_yaml(&format!(
            "model_list:\n  - model_name: assistant\n    litellm_params: {{model: openai/test}}\n{entry}"
        ))
        .unwrap();
        config.model_list[0].clone()
    }

    #[rstest]
    #[case::text("    model_info: {id: primary}", Some(ModelId::Text("primary".to_string())))]
    #[case::number("    model_info: {id: 7}", Some(ModelId::Number(7)))]
    #[case::null("    model_info: {id: null}", None)]
    #[case::absent("    model_info: {mode: chat}", None)]
    fn model_info_id_accepts_what_python_accepts(#[case] entry: &str, #[case] id: Option<ModelId>) {
        assert_eq!(model(entry).model_info.id, id);
    }

    #[rstest]
    #[case::missing("")]
    #[case::null("    model_info:")]
    fn model_info_may_be_missing_or_null(#[case] entry: &str) {
        let model = model(entry);
        assert!(model.model_info.id.is_none());
        assert!(model.model_info.additional_fields.is_empty());
    }

    #[rstest]
    fn model_info_fields_land_typed() {
        let model = model(
            r#"    model_info:
      mode: embedding
      base_model: azure/test-base
      team_id: team-1
      access_groups: [internal, beta]
      db_model: "1"
      blocked: true
      ptu_count: 4
      cost_per_ptu_per_hour: 2
      input_cost_per_token: os.environ/INPUT_COST
      tiered_pricing: [{range: [0, 1000]}]
      access_windows:
        - {start: "09:00", end: "17:00", timezone: UTC, team_ids: [team-1]}
      supports_vision: true
"#,
        );
        let info = &model.model_info;
        assert_eq!(info.mode.as_deref(), Some("embedding"));
        assert_eq!(info.base_model.as_deref(), Some("azure/test-base"));
        assert_eq!(info.team_id.as_deref(), Some("team-1"));
        assert_eq!(
            info.access_groups.as_deref(),
            Some(&["internal".to_string(), "beta".to_string()][..])
        );
        assert_eq!(info.db_model, Some(Spelled::Text("1".to_string())));
        assert_eq!(info.blocked, Some(Spelled::Value(true)));
        assert_eq!(info.ptu_count, Some(Spelled::Value(4)));
        assert_eq!(info.cost_per_ptu_per_hour, Some(Spelled::Value(2.0)));
        assert_eq!(
            info.input_cost_per_token,
            Some(Spelled::Text("os.environ/INPUT_COST".to_string()))
        );
        assert_eq!(
            info.tiered_pricing.as_ref().map(|tiers| tiers.len()),
            Some(1)
        );
        let window = &info.access_windows.as_ref().unwrap()[0];
        assert_eq!(window.timezone.as_deref(), Some("UTC"));
        assert_eq!(window.start.as_ref().unwrap().get().as_str(), Some("09:00"));
        assert_eq!(
            window.team_ids.as_deref(),
            Some(&["team-1".to_string()][..])
        );
        assert!(info.additional_fields.contains_key("supports_vision"));
    }

    #[rstest]
    fn top_level_limits_and_unknown_keys_land_on_the_model() {
        let model = model("    rpm: 100\n    tpm: os.environ/TPM\n    team: research\n");
        assert_eq!(model.rpm, Some(Spelled::Value(100)));
        assert_eq!(model.tpm, Some(Spelled::Text("os.environ/TPM".to_string())));
        assert!(model.additional_fields.contains_key("team"));
        assert!(!model.additional_fields.contains_key("rpm"));
    }

    #[rstest]
    #[case::ptu_shares(
        "    model_info: {ptu_shares: {team-1: sentinel-share-value}}",
        "sentinel-share-value"
    )]
    #[case::created_at(
        "    model_info: {created_at: sentinel-created-at}",
        "sentinel-created-at"
    )]
    #[case::tiered_pricing(
        "    model_info: {tiered_pricing: [{label: sentinel-tier-value}]}",
        "sentinel-tier-value"
    )]
    #[case::window_start(
        "    model_info: {access_windows: [{start: sentinel-window-start, end: '17:00', timezone: UTC, team_ids: [t]}]}",
        "sentinel-window-start"
    )]
    #[case::model_extension("    team_token: sentinel-model-extension", "sentinel-model-extension")]
    fn opaque_model_info_values_stay_out_of_debug(#[case] entry: &str, #[case] sentinel: &str) {
        assert!(!format!("{:?}", model(entry)).contains(sentinel));
    }
}
