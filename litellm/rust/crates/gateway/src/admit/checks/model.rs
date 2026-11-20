//! The model a request names: its group, every group that group can fall back to, and each of
//! their deployments. The router says which groups those are and which host steps a call served by
//! each one makes (`Router::reach`), so nothing here reads a fallback list or an alias. The router
//! picks the deployment after the request is admitted, so one deployment the native app does not
//! serve as Python does sends its whole group to Python, and so does a fallback to such a group.

use axum::http::Uri;
use litellm_config::{LitellmParams, Model, Spelled};
use litellm_core_utils::get_llm_provider_logic::LlmProviders;
use litellm_model_catalog::Catalog;
use litellm_router_core::Reached;
use serde_json::{Map, Value};

use super::{
    Admit, Reason, Refused,
    fields::{IsSet, other, unknown},
};

/// The providers the Python proxy itself serves `/v1/messages` through the Rust route for, as
/// `RULES` in `litellm/rust_bridge/catalog.py` lists them. Python answers any other provider
/// through an adapter the gateway has not been compared with.
const MESSAGES_PROVIDERS: [LlmProviders; 2] = [LlmProviders::Anthropic, LlmProviders::AzureAi];

/// The `litellm_params` beyond the typed ones that the router reads as Python's does.
const HONORED_EXTRA_PARAMS: [&str; 1] = ["weight"];

/// The `model_info` keys the router reads as Python's does. `access_groups` only decides which
/// models a caller other than the master key may call, and `max_input_tokens` is the context
/// window the pre-call checks hold a prompt to.
const HONORED_MODEL_INFO: [&str; 3] = ["id", "access_groups", "max_input_tokens"];

pub(super) fn model(admit: &Admit, fields: &Map<String, Value>) -> Option<Reason> {
    let model = fields.get("model").and_then(Value::as_str)?;
    let Some(reached) = admit.router.reach(model) else {
        return Some(Reason::new(Refused::Model, "unknown_model"));
    };
    reached
        .iter()
        .enumerate()
        .find_map(|(hop, reached)| {
            let group = &reached.group;
            match (reached.served, hop) {
                (false, _) => Some(format!("router_settings fallback to {group}").into()),
                (true, 0) => unserved(admit, reached),
                (true, _) => unserved(admit, reached)
                    .map(|over| format!("router_settings fallback to {group}: {over}").into()),
            }
        })
        .map(|over| Reason::new(Refused::Model, over))
}

/// What a router setting the native host does not serve for the group `reached`, or the first of
/// its deployments the native app does not serve as Python does, is over.
fn unserved(admit: &Admit, reached: &Reached) -> Option<Box<str>> {
    let group = reached.group.as_str();
    super::router::group(&admit.config.router_settings, group)
        .or_else(|| admit.host.unanswered(reached))
        .or_else(|| {
            admit
                .config
                .model_list
                .iter()
                .filter(|model| model.model_name == group)
                .find_map(|model| deployment(model).into_iter().next())
        })
        .or_else(|| {
            admit
                .deployed
                .iter()
                .filter(|(deployed, _)| deployed == group)
                .find_map(|(_, provider)| match provider {
                    Ok(provider) if MESSAGES_PROVIDERS.contains(provider) => None,
                    Ok(provider) => Some(of_provider(provider.into())),
                    Err(unlisted) => Some(of_provider(unlisted)),
                })
        })
        .or_else(|| {
            admit
                .aliased
                .iter()
                .find(|(aliased, _)| aliased == group)
                .map(|(_, alias)| format!("model_list.litellm_params.model alias {alias}").into())
        })
}

/// The name `catalog` holds `model` of `provider` under as another model's alias, as it holds
/// `claude-haiku-latest`. Python sends the model the alias stands for
/// (`resolve_model_latest_alias`), and the native app sends the name as written.
pub(in crate::admit) fn alias(catalog: &Catalog, provider: &str, model: &str) -> Option<Box<str>> {
    let name = model
        .strip_prefix(provider)
        .and_then(|rest| rest.strip_prefix('/'))
        .unwrap_or(model);
    [format!("{provider}/{name}"), name.to_owned()]
        .into_iter()
        .find(|key| catalog.aliases().contains_key(key))
        .map(String::into_boxed_str)
}

fn of_provider(provider: &str) -> Box<str> {
    format!("model_list.litellm_params.model of provider {provider}").into()
}

/// An `api_base` the native client and the proxy's both send to as written. The proxy sends any
/// other one through its own client.
fn is_sendable(api_base: &str) -> bool {
    api_base.parse::<Uri>().is_ok_and(|uri| {
        matches!(uri.scheme_str(), Some("http" | "https"))
            && uri.authority().is_some_and(|authority| {
                let host_only = authority.as_str() == authority.host();
                !authority.host().is_empty()
                    && !authority.as_str().contains('@')
                    && (host_only || authority.port().is_some())
            })
    })
}

fn is_not_positive(timeout: &Option<Spelled<f64>>) -> bool {
    matches!(timeout, Some(Spelled::Value(seconds)) if *seconds <= 0.0)
}

/// `rpm` and `tpm` are honored as the shuffle's weights. A deployment with no `api_key` gets its
/// credential from a flow the proxy can host and the gateway may not, as Entra and workload
/// identity are.
fn params(params: &LitellmParams) -> Vec<Box<str>> {
    let LitellmParams {
        model: _,
        api_version: _,
        aws: _,
        vertex: _,
        custom_llm_provider: _,
        tpm: _,
        rpm: _,
        drop_params: _,
        aws_bedrock_project_id: _,
        claude_platform: _,
        api_key,
        api_base,
        timeout,
        stream_timeout,
        max_retries,
        itpm,
        otpm,
        max_parallel_requests,
        organization,
        tags,
        tag_regex,
        max_budget,
        budget_duration,
        default_api_key_tpm_limit,
        default_api_key_rpm_limit,
        use_in_pass_through,
        use_chat_completions_api,
        litellm_credential_name,
        provider_affinity_header,
        extra,
    } = params;
    let unsent = [
        ("api_key unset", api_key.is_none()),
        (
            "api_base",
            api_base.as_deref().is_some_and(|base| !is_sendable(base)),
        ),
        ("timeout", is_not_positive(timeout)),
        ("stream_timeout", is_not_positive(stream_timeout)),
    ]
    .into_iter()
    .filter(|(_, found)| *found)
    .map(|(name, _)| format!("model_list.litellm_params.{name}").into());
    set!(
        "model_list.litellm_params.",
        [
            max_retries,
            itpm,
            otpm,
            max_parallel_requests,
            organization,
            tags,
            tag_regex,
            max_budget,
            budget_duration,
            default_api_key_tpm_limit,
            default_api_key_rpm_limit,
            litellm_credential_name,
            provider_affinity_header,
        ],
        on: [use_in_pass_through, use_chat_completions_api]
    )
    .chain(unsent)
    .chain(other(
        "model_list.litellm_params.",
        extra.keys(),
        &HONORED_EXTRA_PARAMS,
    ))
    .collect()
}

fn deployment(model: &Model) -> Vec<Box<str>> {
    let Model {
        model_name: _,
        written_params: _,
        litellm_params,
        model_info,
        blocked,
        additional_fields,
    } = model;
    params(litellm_params)
        .into_iter()
        .chain(other(
            "model_list.model_info.",
            model_info.keys(),
            &HONORED_MODEL_INFO,
        ))
        .chain((*blocked == Some(true)).then(|| "model_list.blocked".into()))
        .chain(unknown("model_list.", additional_fields))
        .collect()
}

#[cfg(test)]
mod tests {
    use litellm_config::Config;
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::admit::Deployed;

    const TAGS: &str = "model_list.litellm_params.tags";
    const GROUPS: [&str; 4] = ["claude", "backup", "last", "other"];
    const DEPLOYMENT: &str = "    litellm_params:\n      model: anthropic/claude-haiku-5-5\n      api_key: provider-key\n";

    fn asking(admit: &Admit, name: &str) -> Option<String> {
        let fields = json!({"model": name, "messages": []});
        model(admit, fields.as_object().unwrap()).map(|reason| reason.over.into())
    }

    fn anthropic(groups: &[&str]) -> Vec<Deployed> {
        groups
            .iter()
            .map(|group| ((*group).to_owned(), Ok(LlmProviders::Anthropic)))
            .collect()
    }

    /// Four groups of one deployment each, where only `backup` sets something the native app
    /// does not honor.
    fn backup_tagged(router_settings: &str) -> Admit {
        let models: String = GROUPS
            .iter()
            .map(|group| {
                let tags = if *group == "backup" {
                    "      tags: [paid]\n"
                } else {
                    ""
                };
                format!("  - model_name: {group}\n{DEPLOYMENT}{tags}")
            })
            .collect();
        let config = Config::from_yaml(&format!(
            "model_list:\n{models}router_settings:\n{router_settings}"
        ))
        .unwrap();
        Admit::over(&config, anthropic(&GROUPS))
    }

    fn one_deployment(written: &str, deployed: Vec<Deployed>) -> Option<String> {
        let config =
            Config::from_yaml(&format!("model_list:\n  - model_name: claude\n{written}")).unwrap();
        asking(&Admit::over(&config, deployed), "claude")
    }

    #[rstest]
    fn a_group_with_an_unhonored_deployment_is_named_and_no_other_group_is() {
        let admit = backup_tagged("  num_retries: 1\n");

        assert_eq!(
            GROUPS.map(|group| asking(&admit, group)),
            [None, Some(TAGS.to_owned()), None, None]
        );
    }

    #[rstest]
    #[case::a_fallback("  fallbacks:\n    - claude: [backup]\n", "claude")]
    #[case::a_fallback_of_a_fallback(
        "  fallbacks:\n    - claude: [last]\n    - last: [backup]\n",
        "claude"
    )]
    #[case::a_context_window_fallback(
        "  context_window_fallbacks:\n    - claude: [backup]\n",
        "claude"
    )]
    #[case::a_default_fallback("  default_fallbacks: [backup]\n", "other")]
    #[case::a_fallback_of_every_group("  fallbacks:\n    - \"*\": [backup]\n", "other")]
    #[case::a_fallback_to_an_alias(
        "  fallbacks:\n    - claude: [spare]\n  model_group_alias:\n    spare: backup\n",
        "claude"
    )]
    #[case::a_bare_name_which_is_a_fallback_of_every_group("  fallbacks: [backup]\n", "other")]
    #[case::a_fallback_of_the_alias_asked_for(
        "  fallbacks:\n    - sonnet: [backup]\n  model_group_alias:\n    sonnet: claude\n",
        "sonnet"
    )]
    fn a_group_that_can_fall_back_to_a_named_one_is_named_too(
        #[case] router_settings: &str,
        #[case] asked: &str,
    ) {
        assert_eq!(
            asking(&backup_tagged(router_settings), asked),
            Some(format!("router_settings fallback to backup: {TAGS}"))
        );
    }

    #[rstest]
    fn a_fallback_to_a_group_the_config_does_not_hold_is_named() {
        let admit = backup_tagged("  fallbacks:\n    - claude: [nowhere]\n");

        assert_eq!(
            asking(&admit, "claude").as_deref(),
            Some("router_settings fallback to nowhere")
        );
    }

    #[rstest]
    fn a_fallback_to_a_served_group_keeps_its_group() {
        let admit = backup_tagged("  fallbacks:\n    - claude: [other]\n");

        assert_eq!(asking(&admit, "claude"), None);
    }

    #[rstest]
    #[case::a_strategy_the_host_picks_by(
        "  routing_strategy: least-busy\n",
        "other",
        "router_settings.routing_strategy"
    )]
    #[case::health_check_routing(
        "  enable_health_check_routing: true\n",
        "other",
        "router_settings.enable_health_check_routing"
    )]
    #[case::a_context_window_fallback_written_as_a_dict(
        "  context_window_fallbacks:\n    - other: [{model: last}]\n",
        "other",
        "router_settings.context_window_fallbacks"
    )]
    #[case::behind_a_fallback(
        "  routing_groups:\n    - group_name: pool\n      models: [last]\n      routing_strategy: least-busy\n  fallbacks:\n    - other: [last]\n",
        "other",
        "router_settings fallback to last: router_settings.routing_groups"
    )]
    fn a_host_step_the_native_host_does_not_answer_names_each_group_that_reaches_it(
        #[case] router_settings: &str,
        #[case] asked: &str,
        #[case] over: &str,
    ) {
        assert_eq!(
            asking(&backup_tagged(router_settings), asked).as_deref(),
            Some(over)
        );
    }

    #[rstest]
    fn a_fallback_written_as_a_dict_leaves_every_other_group_native() {
        let admit = backup_tagged("  fallbacks:\n    - claude: [{model: other, temperature: 0}]\n");

        assert_eq!(
            ["claude", "other", "last"].map(|group| asking(&admit, group)),
            [Some("router_settings.fallbacks".to_owned()), None, None]
        );
    }

    #[rstest]
    #[case::a_name("  model_group_alias:\n    sonnet: claude\n", None)]
    #[case::a_hidden_alias(
        "  model_group_alias:\n    sonnet: {model: claude, hidden: true}\n",
        None
    )]
    #[case::of_a_named_group("  model_group_alias:\n    sonnet: backup\n", Some(TAGS))]
    fn an_alias_is_judged_as_the_group_it_names(
        #[case] router_settings: &str,
        #[case] over: Option<&str>,
    ) {
        assert_eq!(
            asking(&backup_tagged(router_settings), "sonnet").as_deref(),
            over
        );
    }

    #[rstest]
    #[case::another_name("gpt")]
    #[case::a_pattern("anthropic/claude-haiku-5-5")]
    #[case::an_alias_of_no_group("lost")]
    #[case::an_alias_without_a_model("unread")]
    fn a_model_the_config_does_not_name_is_unknown(#[case] asked: &str) {
        let aliases = "  model_group_alias:\n    lost: nowhere\n    unread: {hidden: true}\n";

        assert_eq!(
            asking(&backup_tagged(aliases), asked).as_deref(),
            Some("unknown_model")
        );
    }

    #[rstest]
    #[case::a_typed_param("      max_retries: 2\n", "model_list.litellm_params.max_retries")]
    #[case::a_param_the_router_does_not_read(
        "      cache_control_injection_points: [{location: message, role: system}]\n",
        "model_list.litellm_params.cache_control_injection_points"
    )]
    #[case::an_api_base_without_a_scheme(
        "      api_base: api.anthropic.com\n",
        "model_list.litellm_params.api_base"
    )]
    #[case::an_api_base_with_credentials(
        "      api_base: https://user:secret@api.anthropic.com\n",
        "model_list.litellm_params.api_base"
    )]
    #[case::an_api_base_with_an_unreadable_port(
        "      api_base: https://api.anthropic.com:port\n",
        "model_list.litellm_params.api_base"
    )]
    #[case::no_timeout_at_all("      timeout: 0\n", "model_list.litellm_params.timeout")]
    #[case::no_stream_timeout_at_all(
        "      stream_timeout: -1\n",
        "model_list.litellm_params.stream_timeout"
    )]
    #[case::model_info_beyond_the_id(
        "    model_info:\n      id: claude-a\n      base_model: claude-haiku-5-5\n",
        "model_list.model_info.base_model"
    )]
    #[case::an_unknown_model_key("    guardrails: [pii]\n", "model_list.guardrails")]
    fn a_deployment_the_native_app_does_not_honor_names_its_group(
        #[case] written: &str,
        #[case] over: &str,
    ) {
        assert_eq!(
            one_deployment(&format!("{DEPLOYMENT}{written}"), anthropic(&["claude"])).as_deref(),
            Some(over)
        );
    }

    #[rstest]
    fn a_deployment_honoring_every_param_keeps_its_group() {
        let written = "      api_base: https://api.anthropic.com\n      rpm: 300\n      weight: 2\n      timeout: 30\n    model_info:\n      id: claude-a\n      access_groups: [haiku]\n";

        assert_eq!(
            one_deployment(&format!("{DEPLOYMENT}{written}"), anthropic(&["claude"])),
            None
        );
    }

    #[rstest]
    fn a_deployment_without_a_key_names_its_group() {
        let written = "    litellm_params:\n      model: azure_ai/claude-haiku-5-5\n      tenant_id: tenant\n";

        assert_eq!(
            one_deployment(written, anthropic(&["claude"])).as_deref(),
            Some("model_list.litellm_params.api_key unset")
        );
    }

    #[rstest]
    #[case::off_the_list(
        Ok(LlmProviders::Openai),
        Some("model_list.litellm_params.model of provider openai")
    )]
    #[case::one_the_gateway_cannot_name(
        Err("sagemaker".to_owned()),
        Some("model_list.litellm_params.model of provider sagemaker")
    )]
    #[case::on_the_list(Ok(LlmProviders::AzureAi), None)]
    fn a_provider_the_native_route_was_not_compared_on_names_its_group(
        #[case] provider: Result<LlmProviders, String>,
        #[case] over: Option<&str>,
    ) {
        let deployed = vec![
            ("claude".to_owned(), Ok(LlmProviders::Anthropic)),
            ("claude".to_owned(), provider),
        ];

        assert_eq!(one_deployment(DEPLOYMENT, deployed).as_deref(), over);
    }
}
