//! The settings that reach every request. Each config section and `Environment` are taken apart
//! field by field, so a new field does not compile until it is honored (`field: _`) or named. A
//! key the config types do not name is always named.

use axum::http::request::Parts;
use litellm_callbacks_model_capacity::NAME as MODEL_CAPACITY;
use litellm_config::{
    Config, Environment, GeneralSettings, LiteLlmSettings, PriorityReservationSettings,
    RouterSettings, Value,
    db::switches::{DbObject, SupportedDbObjects},
};

use super::{
    Admit, Reason, Refused,
    fields::{IsSet, other, unknown},
};
use crate::{StoredConfig, stored::holds};

const STORE_MODEL_IN_DB: &str = "store_model_in_db";
/// The `LiteLLM_UISettings` row, named when it could not be read.
const UI_SETTINGS: &str = "ui_settings";
/// The `general_settings` keys the config types do not name that the native app honors.
const HONORED_GENERAL_SETTINGS: [&str; 2] = ["supported_db_objects", "encryption_algorithm"];

/// The `litellm_settings` keys a variable can fill, as `Config::with_environment` layers them.
/// `reasoning_auto_summary` and `request_timeout` read their variables as Python does, so neither
/// is here.
const LAYERED: [&str; 11] = [
    "litellm_settings.drop_params",
    "litellm_settings.modify_params",
    "litellm_settings.enable_anthropic_prompt_caching",
    "litellm_settings.use_chat_completions_url_for_anthropic_messages",
    "litellm_settings.route_all_chat_openai_to_responses",
    "litellm_settings.gemini_live_defer_setup",
    "litellm_settings.anthropic_prompt_caching_ttl",
    "litellm_settings.model_cost_map_url",
    "litellm_settings.blog_posts_url",
    "litellm_settings.anthropic_beta_headers_url",
    "litellm_settings.autorouter_presets_url",
];

pub(super) fn settings(admit: &Admit, _: &Parts) -> Option<Reason> {
    match admit.stored {
        StoredConfig::Unread => {
            return Some(Reason::new(Refused::DbConfigNotLoaded, STORE_MODEL_IN_DB));
        }
        StoredConfig::UiSettingsUnread => {
            return Some(Reason::new(Refused::DbConfigNotLoaded, UI_SETTINGS));
        }
        StoredConfig::Unused | StoredConfig::Loaded(_) => {}
    }
    unhonored(&admit.config, &admit.environment)
        .chain(stored_objects(admit))
        .next()
        .map(|over| Reason::new(Refused::Settings, over))
}

/// Each kind of stored object the database holds that the proxy loads beside the stored config and
/// the native app does not read, unless `general_settings.supported_db_objects` leaves it out.
fn stored_objects(admit: &Admit) -> impl Iterator<Item = Box<str>> + '_ {
    let supported = SupportedDbObjects::of(&admit.config.general_settings);
    let held = match admit.stored {
        StoredConfig::Loaded(objects) => Some(objects),
        StoredConfig::Unused | StoredConfig::Unread | StoredConfig::UiSettingsUnread => None,
    };
    <DbObject as strum::VariantArray>::VARIANTS
        .iter()
        .filter(move |object| {
            held.is_some_and(|objects| holds(objects, **object)) && supported.loads(**object)
        })
        .map(|object| {
            format!(
                "general_settings.supported_db_objects.{}",
                <&str>::from(*object)
            )
            .into()
        })
}

/// The setting Python reads only where it sends a Messages request on.
pub(super) fn messages_only(config: &Config) -> Option<Box<str>> {
    let use_chat_completions_url_for_anthropic_messages = &config
        .litellm_settings
        .use_chat_completions_url_for_anthropic_messages;
    set!(
        "litellm_settings.",
        [],
        on: [use_chat_completions_url_for_anthropic_messages]
    )
    .map(|key| spelled(config, key))
    .next()
}

fn unhonored<'a>(
    config: &'a Config,
    environment: &Environment,
) -> impl Iterator<Item = Box<str>> + 'a {
    unread(config)
        .chain(variables(environment))
        .chain(general(config))
        .chain(router(&config.router_settings))
        .chain(litellm(config))
        .chain(sections(config))
}

/// A variable holding a word the key is not read from leaves the key unset here, where the proxy
/// reads some of those words its own way.
fn unread(config: &Config) -> impl Iterator<Item = Box<str>> + '_ {
    LAYERED
        .iter()
        .filter_map(|key| config.origins.unread(key))
        .map(Box::from)
}

/// The key as the variable that set it, when the environment did, so the reason names what to
/// change.
fn spelled(config: &Config, key: Box<str>) -> Box<str> {
    config.origins.variable(&key).map_or(key, Box::from)
}

/// The variables the proxy reads that no config key sets. A variable config.yaml can also set is
/// judged once, as the key it fills.
fn variables(environment: &Environment) -> impl Iterator<Item = Box<str>> + use<> {
    let Environment {
        litellm_cors_origins: _,
        litellm_cors_allow_credentials: _,
        server_root_path: _,
        server_root_paths: _,
        max_decompressed_request_body_size_mb: _,
        max_request_body_decompression_ratio: _,
        max_decompressed_request_bodies_in_flight_mb: _,
        non_llm_connection_timeout: _,
        litellm_rate_limit_window_size: _,
        litellm_tpm_token_reservation_enabled: _,
        proxy_batch_write_at: _,
        spend_log_queue_max_bytes: _,
        spend_log_write_batch_max_bytes: _,
        spend_log_write_batch_max_rows: _,
        legacy_multi_instance_rate_limiting,
        use_aws_kms,
        litellm_ssl_ciphers,
        litellm_worker_startup_hooks,
        litellm_otel_v2,
        use_ddtrace,
        litellm_lens_url,
        litellm_telemetry_endpoint,
        litellm_max_streaming_duration_seconds,
        router_max_fallbacks,
        default_max_retries,
        default_allowed_fails,
        default_cooldown_time_seconds,
        default_cooldown_redis_read_interval_seconds: _,
        default_max_redis_batch_cache_size: _,
        litellm_local_model_cost_map: _,
        litellm_offline_mode: _,
        model_cost_map_min_model_count: _,
        model_cost_map_max_shrink_ratio: _,
        aws_region_name: _,
    } = environment;
    set!(
        "",
        [
            legacy_multi_instance_rate_limiting,
            use_aws_kms,
            litellm_ssl_ciphers,
            litellm_worker_startup_hooks,
            litellm_otel_v2,
            use_ddtrace,
            litellm_lens_url,
            litellm_telemetry_endpoint,
            litellm_max_streaming_duration_seconds,
            router_max_fallbacks,
            default_max_retries,
            default_allowed_fails,
            default_cooldown_time_seconds,
        ]
    )
    .map(|field| field.to_uppercase().into())
}

/// Python fetches the cost map at import from `LITELLM_MODEL_COST_MAP_URL`, as the native app
/// does at boot, and reads `litellm_settings.model_cost_map_url` from config.yaml only on a reload.
fn set_by_its_variable(config: &Config) -> bool {
    config
        .origins
        .variable("litellm_settings.model_cost_map_url")
        .is_some()
}

/// The database settings are read as the proxy reads them, and what the proxy holds in its
/// database beyond keys is not checked here. `store_model_in_db` and `supported_db_objects` are
/// honored, since boot layers the stored config in, and [`settings`] checks what it could not
/// load. `encryption_algorithm` only picks the format Python writes a secret in, and the at-rest
/// crate reads either format. The JWT settings shape only a caller whose token is a JWT, and the public routes only one
/// who sends no key. `allowed_ips`, `use_x_forwarded_for` and `allowed_routes` are
/// `middleware::request_allowlist`, and `head::forwarded_for` names the address the native app
/// cannot resolve as Python does.
fn general(config: &Config) -> impl Iterator<Item = Box<str>> + '_ {
    let GeneralSettings {
        master_key: _,
        litellm_license: _,
        admission_queue_timeout_seconds: _,
        max_request_size_mb: _,
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
        token_rate_limit_type: _,
        user_api_key_cache_ttl: _,
        user_api_key_cache_max_size: _,
        allowed_routes: _,
        use_x_forwarded_for: _,
        allowed_ips: _,
        completion_model,
        max_in_flight_requests_per_worker,
        max_queued_requests_per_worker,
        tracing,
        max_parallel_requests,
        global_max_parallel_requests,
        max_response_size_mb,
        background_health_checks,
        store_model_in_db: _,
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
        custom_auth,
        litellm_key_header_name,
        required_customer_headers,
        role_permissions,
        enforce_rbac,
        user_header_name,
        user_header_mappings,
        additional_fields,
        database_url: _,
    } = &config.general_settings;
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
            custom_auth,
            litellm_key_header_name,
            required_customer_headers,
            role_permissions,
            enforce_rbac,
            user_header_name,
            user_header_mappings,
        ],
        on: [
            background_health_checks,
            forward_client_headers_to_llm_api,
            cancel_on_disconnect,
            infer_model_from_keys,
            dangerously_permit_weak_or_unset_master_key,
        ]
    )
    .chain(other(
        "general_settings.",
        additional_fields.keys(),
        &HONORED_GENERAL_SETTINGS,
    ))
    .map(|key| spelled(config, key))
}

/// The settings router-core never reads. It reads every other one as Python's router does, and
/// `model` names the ones it serves through a host step the native host does not answer, for the
/// groups a request reaches.
/// `optional_pre_call_checks` is honored when it names only the checks it runs itself.
fn router(settings: &RouterSettings) -> impl Iterator<Item = Box<str>> + '_ {
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
        enable_cross_model_group_collision_check: _,
        enable_health_check_routing: _,
        fallback_on_unsupported_params: _,
        enable_tag_filtering,
        optional_pre_call_checks,
        routing_strategy_args,
        max_retries,
        weights,
        tag_routing_prefix,
        redis: _,
        additional_fields,
    } = settings;
    set!(
        "router_settings.",
        [
            routing_strategy_args,
            max_retries,
            weights,
            tag_routing_prefix,
        ],
        on: [enable_tag_filtering],
        by: [optional_pre_call_checks => super::router::checked_by_the_host]
    )
    .chain(other("router_settings.", additional_fields.keys(), &[]))
}

/// A `request_timeout` the native call reads otherwise than Python. Python fails every call on
/// text `float()` refuses and keeps zero as a timeout, where the native call keeps neither. And
/// the native router takes `router_settings.request_timeout` as a timeout of its own, where the
/// Python router derives that value from this setting instead.
fn unkept(config: &Config) -> bool {
    config.litellm_settings.request_timeout.is_some()
        && (crate::boot::request_timeout(&config.litellm_settings).is_none()
            || config.router_settings.request_timeout.is_some())
}

/// A callback other than `dynamic_rate_limiter_v3`, the one the native app runs.
fn other_callbacks(callbacks: &Option<Vec<Value>>) -> bool {
    callbacks
        .iter()
        .flatten()
        .any(|callback| callback.as_str() != Some(MODEL_CAPACITY))
}

/// A per-model limit that borrows a model's idle capacity, which the dynamic rate limiter holds
/// tokens for before the call. The native app serves no caller that holds one.
fn borrows(settings: &Option<PriorityReservationSettings>) -> bool {
    settings.as_ref().is_some_and(|settings| {
        !settings
            .model_limit_borrowing_config
            .borrow_idle_capacity
            .is_empty()
    })
}

/// `drop_params` and `reasoning_auto_summary` are honored whichever of the config and its
/// variable set them, `request_timeout` unless it is [`unkept`], and
/// `use_chat_completions_url_for_anthropic_messages` is [`messages_only`]. `callbacks` is honored
/// when it names only the dynamic rate limiter, which reads `priority_reservation` and its
/// settings. The others the native app does not read only shape another route or the UI.
fn litellm(config: &Config) -> impl Iterator<Item = Box<str>> + '_ {
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
        reasoning_auto_summary: _,
        json_logs: _,
        set_verbose: _,
        enable_redis_auth_cache: _,
        use_chat_completions_url_for_anthropic_messages: _,
        route_all_chat_openai_to_responses: _,
        gemini_live_defer_setup: _,
        blog_posts_url: _,
        autorouter_presets_url: _,
        priority_reservation: _,
        priority_reservation_settings,
        modify_params,
        enable_anthropic_prompt_caching,
        request_timeout,
        num_retries,
        cache,
        cache_params,
        callbacks,
        success_callback,
        failure_callback,
        anthropic_prompt_caching_ttl,
        model_cost_map_url,
        anthropic_beta_headers_url,
        additional_fields,
    } = &config.litellm_settings;
    set!(
        "litellm_settings.",
        [
            num_retries,
            cache_params,
            success_callback,
            failure_callback,
            anthropic_prompt_caching_ttl,
            anthropic_beta_headers_url,
        ],
        on: [cache, modify_params, enable_anthropic_prompt_caching],
        by: [
            callbacks => other_callbacks,
            priority_reservation_settings => borrows,
            request_timeout => |_| unkept(config),
            model_cost_map_url => |url: &Option<_>| url.is_some() && !set_by_its_variable(config)
        ]
    )
    .chain(unknown("litellm_settings.", additional_fields))
    .map(|key| spelled(config, key))
}

fn sections(config: &Config) -> impl Iterator<Item = Box<str>> + '_ {
    let Config {
        model_list: _,
        general_settings: _,
        router_settings: _,
        litellm_settings: _,
        environment_variables: _,
        include: _,
        origins: _,
        callback_settings,
        assistant_settings,
        default_vertex_config,
        mcp_servers,
        credential_list: _,
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
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn found(config: &Config) -> Vec<String> {
        unhonored(config, &Environment::default())
            .map(String::from)
            .collect()
    }

    fn written(yaml: &str) -> Vec<String> {
        found(&Config::from_yaml(yaml).unwrap())
    }

    #[rstest]
    #[case::a_typed_setting(
        "general_settings:\n  custom_auth: my.auth\n",
        "general_settings.custom_auth"
    )]
    #[case::required_customer_headers(
        "general_settings:\n  required_customer_headers: [X-Tenant-ID]\n",
        "general_settings.required_customer_headers"
    )]
    #[case::a_flag_the_proxy_defaults_off(
        "general_settings:\n  forward_client_headers_to_llm_api: true\n",
        "general_settings.forward_client_headers_to_llm_api"
    )]
    #[case::a_flag_the_config_spells_as_text(
        "litellm_settings:\n  modify_params: \"true\"\n",
        "litellm_settings.modify_params"
    )]
    #[case::a_key_the_config_types_do_not_name(
        "router_settings:\n  provider_budget_config: {}\n",
        "router_settings.provider_budget_config"
    )]
    #[case::an_unknown_general_setting_beside_the_encryption_algorithm(
        "general_settings:\n  encryption_algorithm: aes-256-gcm\n  future_setting: true\n",
        "general_settings.future_setting"
    )]
    #[case::a_pre_call_check_the_host_runs(
        "router_settings:\n  optional_pre_call_checks: [enforce_model_rate_limits, prompt_caching]\n",
        "router_settings.optional_pre_call_checks"
    )]
    #[case::a_section_the_config_types_do_not_name("future_section: {}\n", "future_section")]
    #[case::a_callback_beside_the_dynamic_rate_limiter(
        "litellm_settings:\n  callbacks: [dynamic_rate_limiter_v3, prometheus]\n",
        "litellm_settings.callbacks"
    )]
    #[case::a_request_timeout_float_refuses(
        "litellm_settings:\n  request_timeout: soon\n",
        "litellm_settings.request_timeout"
    )]
    #[case::no_request_timeout_at_all(
        "litellm_settings:\n  request_timeout: 0\n",
        "litellm_settings.request_timeout"
    )]
    #[case::a_request_timeout_beside_the_routers_request_timeout(
        "litellm_settings:\n  request_timeout: 30\nrouter_settings:\n  request_timeout: 10\n",
        "litellm_settings.request_timeout"
    )]
    fn a_setting_the_native_app_does_not_honor_is_named(#[case] yaml: &str, #[case] setting: &str) {
        assert_eq!(written(yaml), [setting]);
    }

    #[rstest]
    #[case::reasoning_auto_summary_as_text(
        "litellm_settings:\n  reasoning_auto_summary: \"false\"\n"
    )]
    #[case::a_request_timeout("litellm_settings:\n  request_timeout: 30\n")]
    #[case::a_request_timeout_as_text("litellm_settings:\n  request_timeout: \"2.5\"\n")]
    #[case::the_routers_timeout_alone("router_settings:\n  timeout: 10\n")]
    #[case::a_request_timeout_beside_the_routers_timeout(
        "litellm_settings:\n  request_timeout: 30\nrouter_settings:\n  timeout: 10\n"
    )]
    fn a_messages_setting_the_native_call_reads_as_python_does_is_not_named(#[case] yaml: &str) {
        assert_eq!(written(yaml), Vec::<String>::new());
    }

    #[rstest]
    fn a_flag_written_off_is_not_named() {
        let yaml = "litellm_settings:\n  cache: false\n  modify_params: false\n  use_chat_completions_url_for_anthropic_messages: false\ngeneral_settings:\n  enable_oauth2_auth: false\n  enforce_rbac: false\n  cancel_on_disconnect: false\n";

        assert_eq!(
            (
                written(yaml),
                messages_only(&Config::from_yaml(yaml).unwrap())
            ),
            (vec![], None)
        );
    }

    #[rstest]
    #[case::aes_256_gcm("aes-256-gcm")]
    #[case::xsalsa20_poly1305("xsalsa20-poly1305")]
    fn the_encryption_algorithm_python_writes_with_is_honored(#[case] algorithm: &str) {
        assert_eq!(
            written(&format!(
                "general_settings:\n  encryption_algorithm: {algorithm}\n"
            )),
            Vec::<String>::new()
        );
    }

    #[rstest]
    fn a_setting_the_proxy_reads_only_for_messages_is_named_only_for_that_route() {
        let config = Config::from_yaml(
            "litellm_settings:\n  use_chat_completions_url_for_anthropic_messages: true\n",
        )
        .unwrap();

        assert_eq!(
            (found(&config), messages_only(&config).as_deref()),
            (
                vec![],
                Some("litellm_settings.use_chat_completions_url_for_anthropic_messages")
            )
        );
    }

    fn set_by(variable: &'static str, value: &'static str) -> Config {
        let lookup = move |name: &str| (name == variable).then(|| value.to_owned());
        Config::default()
            .with_environment(&lookup)
            .with_process_environment(&lookup)
    }

    #[rstest]
    fn a_setting_the_environment_set_is_named_by_its_variable() {
        assert_eq!(
            found(&set_by("LITELLM_MODIFY_PARAMS", "true")),
            ["LITELLM_MODIFY_PARAMS"]
        );
    }

    #[rstest]
    fn a_setting_the_native_app_honors_is_honored_when_the_environment_set_it() {
        assert_eq!(
            found(&set_by("LITELLM_DROP_PARAMS", "true")),
            Vec::<String>::new()
        );
    }

    #[rstest]
    #[case::reasoning_auto_summary("LITELLM_REASONING_AUTO_SUMMARY", "yes", None)]
    #[case::a_request_timeout("REQUEST_TIMEOUT", "30", None)]
    #[case::a_request_timeout_float_refuses("REQUEST_TIMEOUT", "soon", None)]
    #[case::no_request_timeout_at_all("REQUEST_TIMEOUT", "0", Some("REQUEST_TIMEOUT"))]
    fn a_messages_setting_its_variable_set_is_named_by_it_only_where_the_native_call_differs(
        #[case] variable: &'static str,
        #[case] value: &'static str,
        #[case] named: Option<&str>,
    ) {
        assert_eq!(
            found(&set_by(variable, value)),
            named.into_iter().collect::<Vec<_>>()
        );
    }

    #[rstest]
    #[case::by_its_variable(
        set_by("LITELLM_MODEL_COST_MAP_URL", "https://cost-map.test/map.json"),
        &[]
    )]
    #[case::in_the_config(
        Config::from_yaml("litellm_settings:\n  model_cost_map_url: https://cost-map.test/map.json\n")
            .unwrap(),
        &["litellm_settings.model_cost_map_url"]
    )]
    fn the_cost_map_url_is_honored_only_where_the_proxy_fetches_at_boot_from(
        #[case] config: Config,
        #[case] named: &[&str],
    ) {
        assert_eq!(found(&config), named);
    }

    #[rstest]
    fn a_variable_holding_a_word_that_is_no_flag_is_named() {
        assert_eq!(
            found(&set_by("LITELLM_DROP_PARAMS", "maybe")),
            ["LITELLM_DROP_PARAMS"]
        );
    }

    #[rstest]
    fn a_variable_no_config_key_sets_is_named_as_the_variable() {
        let environment = Environment {
            use_aws_kms: Some("1".into()),
            ..Environment::default()
        };

        assert_eq!(
            variables(&environment)
                .map(String::from)
                .collect::<Vec<_>>(),
            ["USE_AWS_KMS"]
        );
    }
}
