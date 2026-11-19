//! Which settings the native app does not honor as the Python proxy does, and whose requests each
//! one leaves to Python.
//!
//! A setting is honored when the native app reads it as the proxy does, or when it only shapes
//! something the gateway never admits: another route, or a caller the master key does not
//! authenticate. Every other setting is a finding, never ignored. Each section is taken apart
//! field by field, so a setting added to the config does not compile here until it is either
//! honored or named. Porting a feature deletes its finding.

use std::collections::BTreeMap;

use axum::http::Uri;
use litellm_config::{
    AdditionalFields, Config, GeneralSettings, LiteLlmSettings, LitellmParams, Model, Object,
    RouterSettings, Spelled,
};
use strum::VariantArray;

use super::{
    plan::{Finding, Plan, Routes, Scope},
    request::Trait,
};

/// The providers the Python proxy itself serves `/v1/messages` through the Rust route for, as
/// `RULES` in `litellm/rust_bridge/catalog.py` lists them. Python answers any other provider
/// through an adapter the gateway has not been compared with.
const MESSAGES_PROVIDERS: [&str; 2] = ["anthropic", "azure_ai"];

/// The `litellm_params` beyond the typed ones that the router reads as Python's does.
const HONORED_EXTRA_PARAMS: [&str; 1] = ["weight"];

/// The `model_info` keys the router reads as Python's does.
const HONORED_MODEL_INFO: [&str; 1] = ["id"];

/// An environment variable that changes how the proxy answers `/v1/messages` and that the
/// gateway does not read. The list is what was found, since a variable nobody named cannot be.
#[derive(Clone, Copy, strum::IntoStaticStr, strum::VariantArray)]
#[strum(serialize_all = "SCREAMING_SNAKE_CASE")]
enum Switch {
    LitellmDropParams,
    LitellmModifyParams,
    LitellmEnableAnthropicPromptCaching,
    LitellmAnthropicPromptCachingTtl,
    LitellmReasoningAutoSummary,
    LitellmUseChatCompletionsUrlForAnthropicMessages,
    LegacyMultiInstanceRateLimiting,
    UseAwsKms,
    LitellmSslCiphers,
    LitellmWorkerStartupHooks,
    LitellmOtelV2,
    UseDdtrace,
    LitellmLensUrl,
    LitellmTelemetryEndpoint,
    LitellmMaxStreamingDurationSeconds,
    RouterMaxFallbacks,
    DefaultMaxRetries,
    DefaultAllowedFails,
    DefaultCooldownTimeSeconds,
}

/// The name of each environment variable the caller looks up for [`Native::switches`].
pub(super) fn switches() -> impl Iterator<Item = &'static str> {
    Switch::VARIANTS.iter().map(<&str>::from)
}

/// What the config does not say about the process the native app runs in.
pub(super) struct Native<'a> {
    /// Each deployment the router holds, as its model group and the provider it resolved to.
    pub deployed: &'a [(String, String)],
    /// Whether Python runs with a database, which it reads keys, spend and settings from.
    pub database: bool,
    /// The [`switches`] that are set.
    pub switches: &'a [&'static str],
}

trait IsSet {
    fn is_set(&self) -> bool;
}

impl IsSet for bool {
    fn is_set(&self) -> bool {
        *self
    }
}

impl<T> IsSet for Option<T> {
    fn is_set(&self) -> bool {
        self.is_some()
    }
}

impl<T> IsSet for Box<[T]> {
    fn is_set(&self) -> bool {
        !self.is_empty()
    }
}

impl<K, V> IsSet for BTreeMap<K, V> {
    fn is_set(&self) -> bool {
        !self.is_empty()
    }
}

impl IsSet for Object {
    fn is_set(&self) -> bool {
        !self.is_empty()
    }
}

/// Each of the fields that is set, and each of the `on` flags that is `true`, named as the config
/// spells it. A flag written `false` is the proxy's default.
macro_rules! set {
    ($section:literal, [$($field:ident),+ $(,)?] $(, on: [$($flag:ident),+ $(,)?])?) => {
        [
            $((concat!($section, stringify!($field)), IsSet::is_set($field)),)+
            $($((concat!($section, stringify!($flag)), *$flag == Some(true)),)+)?
        ]
        .into_iter()
        .filter(|(_, set)| *set)
        .map(|(name, _)| Box::<str>::from(name))
    };
}

/// Each of `keys` that is not among `honored`.
fn other<'a>(
    section: &'a str,
    keys: impl Iterator<Item = &'a String> + 'a,
    honored: &'a [&str],
) -> impl Iterator<Item = Box<str>> + 'a {
    keys.filter(|key| !honored.contains(&key.as_str()))
        .map(move |key| format!("{section}{key}").into())
}

fn unknown<'a>(
    section: &'a str,
    fields: &'a AdditionalFields,
) -> impl Iterator<Item = Box<str>> + 'a {
    other(section, fields.keys(), &[])
}

/// The database settings are left to [`Native::database`]. The JWT settings shape only a caller
/// whose token is a JWT, and the public routes only one who sends no key.
fn general(settings: &GeneralSettings) -> Vec<Box<str>> {
    let GeneralSettings {
        master_key: _,
        litellm_license: _,
        admission_queue_timeout_seconds: _,
        max_request_size_mb: _,
        database_url: _,
        database_connection_pool_limit: _,
        database_connection_timeout: _,
        database_connect_timeout: _,
        database_socket_timeout: _,
        database_max_idle_connection_lifetime: _,
        database_connection_pool_timeout: _,
        database_statement_timeout: _,
        database_lock_timeout: _,
        database_idle_in_transaction_session_timeout: _,
        database_disable_prepared_statements: _,
        proxy_config_reload_interval_seconds: _,
        health_check_interval: _,
        health_check_concurrency: _,
        enable_public_model_hub: _,
        mcp_allowed_hosts: _,
        mcp_allowed_origins: _,
        public_routes: _,
        enable_jwt_auth: _,
        litellm_jwtauth: _,
        allow_user_auth: _,
        completion_model,
        max_in_flight_requests_per_worker,
        max_queued_requests_per_worker,
        tracing,
        max_parallel_requests,
        global_max_parallel_requests,
        max_response_size_mb,
        background_health_checks,
        store_model_in_db,
        forward_client_headers_to_llm_api,
        cancel_on_disconnect,
        infer_model_from_keys,
        dangerously_permit_weak_or_unset_master_key,
        plugins,
        coordination_redis,
        pass_through_endpoints,
        enable_oauth2_auth,
        enable_oauth2_proxy_auth,
        allow_requests_on_db_unavailable,
        allow_client_side_credentials,
        allowed_ips,
        allowed_routes,
        custom_auth,
        litellm_key_header_name,
        role_permissions,
        enforce_rbac,
        user_header_name,
        user_header_mappings,
        additional_fields,
    } = settings;
    set!(
        "general_settings.",
        [
            completion_model,
            max_in_flight_requests_per_worker,
            max_queued_requests_per_worker,
            tracing,
            max_parallel_requests,
            global_max_parallel_requests,
            max_response_size_mb,
            plugins,
            coordination_redis,
            pass_through_endpoints,
            enable_oauth2_auth,
            enable_oauth2_proxy_auth,
            allow_requests_on_db_unavailable,
            allow_client_side_credentials,
            allowed_ips,
            allowed_routes,
            custom_auth,
            litellm_key_header_name,
            role_permissions,
            enforce_rbac,
            user_header_name,
            user_header_mappings,
        ],
        on: [
            background_health_checks,
            store_model_in_db,
            forward_client_headers_to_llm_api,
            cancel_on_disconnect,
            infer_model_from_keys,
            dangerously_permit_weak_or_unset_master_key,
        ]
    )
    .chain(unknown("general_settings.", additional_fields))
    .collect()
}

/// Every setting router-core reads is its own to honor or refuse when the router is built, so
/// only the ones it never reads are named here.
fn router(settings: &RouterSettings) -> Vec<Box<str>> {
    let RouterSettings {
        routing_strategy: _,
        routing_groups: _,
        retry_policy: _,
        model_group_retry_policy: _,
        allowed_fails_policy: _,
        model_group_affinity_config: _,
        allowed_fails: _,
        cooldown_time: _,
        disable_cooldowns: _,
        num_retries: _,
        timeout: _,
        request_timeout: _,
        retry_after: _,
        fallbacks: _,
        context_window_fallbacks: _,
        content_policy_fallbacks: _,
        default_fallbacks: _,
        max_fallbacks: _,
        model_group_alias: _,
        enable_pre_call_checks: _,
        enable_weighted_failover: _,
        enable_health_check_routing: _,
        fallback_on_unsupported_params: _,
        enable_tag_filtering: _,
        optional_pre_call_checks: _,
        routing_strategy_args,
        max_retries,
        weights,
        tag_routing_prefix,
        additional_fields,
    } = settings;
    set!(
        "router_settings.",
        [
            routing_strategy_args,
            max_retries,
            weights,
            tag_routing_prefix
        ]
    )
    .chain(other("router_settings.", additional_fields.keys(), &[]))
    .collect()
}

fn litellm(settings: &LiteLlmSettings) -> Vec<Box<str>> {
    let LiteLlmSettings {
        ssl_verify: _,
        ssl_certificate: _,
        ssl_security_level: _,
        ssl_ecdh_curve: _,
        force_ipv4: _,
        http2: _,
        aiohttp_trust_env: _,
        disable_aiohttp_trust_env: _,
        disable_aiohttp_transport: _,
        drop_params: _,
        json_logs: _,
        set_verbose: _,
        request_timeout,
        num_retries,
        cache,
        cache_params,
        callbacks,
        success_callback,
        failure_callback,
        additional_fields,
    } = settings;
    set!(
        "litellm_settings.",
        [
            request_timeout,
            num_retries,
            cache_params,
            callbacks,
            success_callback,
            failure_callback,
        ],
        on: [cache]
    )
    .chain(unknown("litellm_settings.", additional_fields))
    .collect()
}

fn sections(config: &Config) -> Vec<Box<str>> {
    let Config {
        model_list: _,
        general_settings: _,
        router_settings: _,
        litellm_settings: _,
        environment_variables: _,
        include: _,
        callback_settings,
        assistant_settings,
        default_vertex_config,
        mcp_servers,
        credential_list,
        guardrails,
        prompts,
        sandbox_tools,
        search_tools,
        files_settings,
        finetune_settings,
        mcp_tools,
        vector_store_registry,
        worker_registry,
        agents,
        agent_list,
        policies,
        policy_attachments,
        additional_fields,
    } = config;
    set!(
        "",
        [
            callback_settings,
            assistant_settings,
            default_vertex_config,
            mcp_servers,
            credential_list,
            guardrails,
            prompts,
            sandbox_tools,
            search_tools,
            files_settings,
            finetune_settings,
            mcp_tools,
            vector_store_registry,
            worker_registry,
            agents,
            agent_list,
            policies,
            policy_attachments,
        ]
    )
    .chain(unknown("", additional_fields))
    .collect()
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

/// The proxy turns `drop_params` on for a coding agent unless the setting reads as a flag, and the
/// native app only reads the setting.
fn defaults_differ_for_a_coding_agent(settings: &LiteLlmSettings) -> bool {
    !matches!(settings.drop_params, Some(Spelled::Value(_)))
}

fn findings(config: &Config, native: &Native<'_>) -> Vec<Finding> {
    let everything = native
        .database
        .then(|| Box::<str>::from("database"))
        .into_iter()
        .chain(native.switches.iter().map(|switch| Box::from(*switch)))
        .chain(general(&config.general_settings))
        .chain(router(&config.router_settings))
        .chain(litellm(&config.litellm_settings))
        .chain(sections(config))
        .map(|setting| Finding {
            setting,
            leaves: Scope::Everything,
        });
    let settings = config.model_list.iter().flat_map(|model| {
        deployment(model).into_iter().map(|setting| Finding {
            setting,
            leaves: Scope::Group(model.model_name.as_str().into()),
        })
    });
    let providers = native
        .deployed
        .iter()
        .filter(|(_, provider)| !MESSAGES_PROVIDERS.contains(&provider.as_str()))
        .map(|(group, provider)| Finding {
            setting: format!("model_list.litellm_params.model of provider {provider}").into(),
            leaves: Scope::Group(group.as_str().into()),
        });
    let coding_agent =
        defaults_differ_for_a_coding_agent(&config.litellm_settings).then(|| Finding {
            setting: "litellm_settings.drop_params unset".into(),
            leaves: Scope::Asking(Trait::CodingAgent),
        });
    everything
        .chain(settings)
        .chain(providers)
        .chain(coding_agent)
        .collect()
}

/// The plan for `config` as the native app `native` describes runs it. Fallbacks or aliases
/// written in a shape this does not read leave every request to Python.
pub(super) fn plan(config: &Config, native: &Native<'_>) -> Plan {
    let routes = Routes::read(
        config
            .model_list
            .iter()
            .map(|model| model.model_name.as_str()),
        &config.router_settings,
    );
    let unread = routes.is_none().then(|| Finding {
        setting: "router_settings fallbacks or model_group_alias".into(),
        leaves: Scope::Everything,
    });
    Plan::new(
        findings(config, native).into_iter().chain(unread).collect(),
        routes.unwrap_or_default(),
    )
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    const MODEL: &str = "  - model_name: claude\n    litellm_params:\n      model: anthropic/claude-haiku-5-5\n      api_key: provider-key\n";
    const DROP: &str = "litellm_settings:\n  drop_params: true\n";

    fn native<'a>(deployed: &'a [(String, String)]) -> Native<'a> {
        Native {
            deployed,
            database: false,
            switches: &[],
        }
    }

    fn found(yaml: &str, native: &Native<'_>) -> Vec<(String, Scope)> {
        findings(&Config::from_yaml(yaml).unwrap(), native)
            .into_iter()
            .map(|finding| (finding.setting.into(), finding.leaves))
            .collect()
    }

    fn everything(setting: &str) -> Vec<(String, Scope)> {
        vec![(setting.to_owned(), Scope::Everything)]
    }

    fn claude(setting: &str) -> Vec<(String, Scope)> {
        vec![(setting.to_owned(), Scope::Group("claude".into()))]
    }

    #[rstest]
    fn a_config_of_honored_settings_has_no_finding() {
        let yaml = format!(
            "model_list:\n{MODEL}      api_base: https://api.anthropic.com\n      rpm: 300\n      weight: 2\n      timeout: 30\n    model_info:\n      id: claude-a\nrouter_settings:\n  routing_strategy: simple-shuffle\n  num_retries: 1\n  fallbacks:\n    - claude-down: [claude]\nlitellm_settings:\n  drop_params: true\ngeneral_settings:\n  master_key: gateway-key\n  max_request_size_mb: 1\n  enable_jwt_auth: true\n  store_model_in_db: false\nenvironment_variables:\n  SOME_KEY: value\n"
        );
        let deployed = [("claude".to_owned(), "anthropic".to_owned())];

        assert_eq!(found(&yaml, &native(&deployed)), []);
    }

    #[rstest]
    #[case::a_general_setting(
        "general_settings:\n  custom_auth: my.auth\n",
        "general_settings.custom_auth"
    )]
    #[case::a_general_flag_that_is_on(
        "general_settings:\n  enable_oauth2_auth: true\n",
        "general_settings.enable_oauth2_auth"
    )]
    #[case::a_general_flag_the_proxy_defaults_off(
        "general_settings:\n  forward_client_headers_to_llm_api: true\n",
        "general_settings.forward_client_headers_to_llm_api"
    )]
    #[case::an_unknown_general_setting(
        "general_settings:\n  alerting: [slack]\n",
        "general_settings.alerting"
    )]
    #[case::a_router_setting_the_router_never_reads(
        "router_settings:\n  routing_strategy_args: {ttl: 10}\n",
        "router_settings.routing_strategy_args"
    )]
    #[case::an_unknown_router_setting(
        "router_settings:\n  redis_host: localhost\n",
        "router_settings.redis_host"
    )]
    #[case::a_litellm_setting(
        "litellm_settings:\n  drop_params: true\n  callbacks: [prometheus]\n",
        "litellm_settings.callbacks"
    )]
    #[case::the_response_cache(
        "litellm_settings:\n  drop_params: true\n  cache: true\n",
        "litellm_settings.cache"
    )]
    #[case::an_unknown_litellm_setting(
        "litellm_settings:\n  drop_params: true\n  turn_off_message_logging: true\n",
        "litellm_settings.turn_off_message_logging"
    )]
    #[case::a_section_the_gateway_does_not_run(
        "guardrails:\n  - guardrail_name: pii\n",
        "guardrails"
    )]
    #[case::an_unknown_section("future_section: {}\n", "future_section")]
    fn a_setting_the_native_app_does_not_honor_leaves_everything_to_python(
        #[case] yaml: &str,
        #[case] setting: &str,
    ) {
        let yaml = match yaml.starts_with("litellm_settings") {
            true => yaml.to_owned(),
            false => format!("{DROP}{yaml}"),
        };

        assert_eq!(found(&yaml, &native(&[])), everything(setting));
    }

    #[rstest]
    fn a_flag_written_off_is_no_finding() {
        let yaml = format!(
            "{DROP}  cache: false\ngeneral_settings:\n  enable_oauth2_auth: false\n  enforce_rbac: false\n  cancel_on_disconnect: false\n"
        );

        assert_eq!(found(&yaml, &native(&[])), []);
    }

    #[rstest]
    #[case::a_typed_param("      max_retries: 2\n", "model_list.litellm_params.max_retries")]
    #[case::a_limit_on_every_key(
        "      default_api_key_rpm_limit: 10\n",
        "model_list.litellm_params.default_api_key_rpm_limit"
    )]
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
    #[case::an_api_base_of_another_scheme(
        "      api_base: ftp://api.anthropic.com\n",
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
    #[case::a_blocked_model("    blocked: true\n", "model_list.blocked")]
    #[case::an_unknown_model_key("    guardrails: [pii]\n", "model_list.guardrails")]
    fn a_deployment_the_native_app_does_not_honor_leaves_its_group_to_python(
        #[case] written: &str,
        #[case] setting: &str,
    ) {
        let yaml = format!("{DROP}model_list:\n{MODEL}{written}");

        assert_eq!(found(&yaml, &native(&[])), claude(setting));
    }

    #[rstest]
    fn a_deployment_without_a_key_leaves_its_group_to_python() {
        let yaml = format!(
            "{DROP}model_list:\n  - model_name: claude\n    litellm_params:\n      model: azure_ai/claude-haiku-5-5\n      tenant_id: tenant\n"
        );

        assert_eq!(
            found(&yaml, &native(&[]))[0],
            claude("model_list.litellm_params.api_key unset")[0]
        );
    }

    #[rstest]
    fn a_provider_off_the_list_leaves_its_group_to_python() {
        let deployed = [
            ("claude".to_owned(), "anthropic".to_owned()),
            ("azure".to_owned(), "azure_ai".to_owned()),
            ("gpt".to_owned(), "openai".to_owned()),
        ];

        assert_eq!(
            found(DROP, &native(&deployed)),
            [(
                "model_list.litellm_params.model of provider openai".to_owned(),
                Scope::Group("gpt".into())
            )]
        );
    }

    #[rstest]
    #[case::unset("general_settings:\n  master_key: gateway-key\n", true)]
    #[case::spelled_as_text("litellm_settings:\n  drop_params: \"true\"\n", true)]
    #[case::on("litellm_settings:\n  drop_params: true\n", false)]
    #[case::off("litellm_settings:\n  drop_params: false\n", false)]
    fn a_coding_agent_is_left_to_python_unless_drop_params_is_a_flag(
        #[case] yaml: &str,
        #[case] left: bool,
    ) {
        let expected = left.then(|| {
            (
                "litellm_settings.drop_params unset".to_owned(),
                Scope::Asking(Trait::CodingAgent),
            )
        });

        assert_eq!(found(yaml, &native(&[])), Vec::from_iter(expected));
    }

    #[rstest]
    fn a_database_and_each_switch_that_is_set_leave_everything_to_python() {
        let native = Native {
            deployed: &[],
            database: true,
            switches: &["LITELLM_MODIFY_PARAMS", "USE_AWS_KMS"],
        };

        assert_eq!(
            found(DROP, &native),
            [
                everything("database"),
                everything("LITELLM_MODIFY_PARAMS"),
                everything("USE_AWS_KMS")
            ]
            .concat()
        );
    }

    #[rstest]
    fn each_switch_is_named_as_the_proxy_reads_it() {
        assert_eq!(
            Vec::from_iter(switches()),
            [
                "LITELLM_DROP_PARAMS",
                "LITELLM_MODIFY_PARAMS",
                "LITELLM_ENABLE_ANTHROPIC_PROMPT_CACHING",
                "LITELLM_ANTHROPIC_PROMPT_CACHING_TTL",
                "LITELLM_REASONING_AUTO_SUMMARY",
                "LITELLM_USE_CHAT_COMPLETIONS_URL_FOR_ANTHROPIC_MESSAGES",
                "LEGACY_MULTI_INSTANCE_RATE_LIMITING",
                "USE_AWS_KMS",
                "LITELLM_SSL_CIPHERS",
                "LITELLM_WORKER_STARTUP_HOOKS",
                "LITELLM_OTEL_V2",
                "USE_DDTRACE",
                "LITELLM_LENS_URL",
                "LITELLM_TELEMETRY_ENDPOINT",
                "LITELLM_MAX_STREAMING_DURATION_SECONDS",
                "ROUTER_MAX_FALLBACKS",
                "DEFAULT_MAX_RETRIES",
                "DEFAULT_ALLOWED_FAILS",
                "DEFAULT_COOLDOWN_TIME_SECONDS",
            ]
        );
    }

    #[rstest]
    fn fallbacks_written_in_a_shape_that_is_not_read_leave_everything_to_python() {
        let config = Config::from_yaml(&format!(
            "{DROP}model_list:\n{MODEL}router_settings:\n  fallbacks: [claude]\n"
        ))
        .unwrap();

        let plan = plan(&config, &native(&[]));

        assert_eq!(
            plan.everything().err().map(|left| left.over.to_owned()),
            Some("router_settings fallbacks or model_group_alias".to_owned())
        );
    }

    #[rstest]
    fn one_group_left_to_python_leaves_the_groups_beside_it_native() {
        let config = Config::from_yaml(&format!(
            "{DROP}model_list:\n{MODEL}  - model_name: gpt\n    litellm_params:\n      model: openai/gpt-5.5\n      api_key: provider-key\n  - model_name: tagged\n    litellm_params:\n      model: anthropic/claude-haiku-5-5\n      api_key: provider-key\n      tags: [paid]\n  - model_name: backed\n    litellm_params:\n      model: anthropic/claude-haiku-5-5\n      api_key: provider-key\nrouter_settings:\n  fallbacks:\n    - backed: [gpt]\n"
        ))
        .unwrap();
        let deployed = [
            ("claude".to_owned(), "anthropic".to_owned()),
            ("gpt".to_owned(), "openai".to_owned()),
            ("tagged".to_owned(), "anthropic".to_owned()),
            ("backed".to_owned(), "anthropic".to_owned()),
        ];

        let plan = plan(&config, &native(&deployed));

        assert_eq!(
            ["claude", "gpt", "tagged", "backed"].map(|model| plan.model(model).is_ok()),
            [true, false, false, false]
        );
        assert_eq!(plan.everything(), Ok(()));
    }
}
