//! Which configs the native app runs the way the Python proxy does.
//!
//! A setting is honored when the native app reads it as the proxy does, or when it only shapes
//! something the gateway never admits: another route, a caller the master key does not
//! authenticate, or a database, which `Admit` refuses on its own. Every other setting is
//! refused, never ignored, so a config is never served with a setting silently dropped.

use std::collections::BTreeMap;

use litellm_config::{
    AdditionalFields, Config, GeneralSettings, LiteLlmSettings, LitellmParams, Model, Object,
    RouterSettings,
};
use litellm_router_core::{Router, registry::DeploymentFilter};

/// The providers the Python proxy itself serves `/v1/messages` through the Rust route for, as
/// `RULES` in `litellm/rust_bridge/catalog.py` lists them. Python answers any other provider
/// through an adapter the gateway does not have.
const MESSAGES_PROVIDERS: [&str; 2] = ["anthropic", "azure_ai"];

/// The `litellm_params` beyond the typed ones that the router reads as Python's does.
const HONORED_EXTRA_PARAMS: [&str; 1] = ["weight"];

/// The `model_info` keys the router reads as Python's does. Every other one feeds cost
/// tracking, health checks or parameter gating the gateway has not been compared on.
const HONORED_MODEL_INFO: [&str; 1] = ["id"];

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

/// The first of `fields` that is set, named as the config spells it.
macro_rules! first_set {
    ($section:literal, [$($field:ident),+ $(,)?]) => {
        [$((concat!($section, stringify!($field)), IsSet::is_set($field))),+]
            .into_iter()
            .find_map(|(name, set)| set.then(|| Box::from(name)))
    };
}

/// The first of `keys` that is not among `honored`.
fn other<'a>(
    section: &str,
    mut keys: impl Iterator<Item = &'a String>,
    honored: &[&str],
) -> Option<Box<str>> {
    keys.find(|key| !honored.contains(&key.as_str()))
        .map(|key| format!("{section}{key}").into())
}

fn unknown(section: &str, fields: &AdditionalFields) -> Option<Box<str>> {
    other(section, fields.keys(), &[])
}

/// The first setting of `config` the native app does not honor, named as the config spells it.
/// Each section is taken apart field by field, so a setting added to the config does not compile
/// here until it is either honored or refused.
pub(super) fn unhonored(config: &Config) -> Option<Box<str>> {
    let Config {
        model_list,
        general_settings,
        router_settings,
        litellm_settings,
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
    general(general_settings)
        .or_else(|| router(router_settings))
        .or_else(|| litellm(litellm_settings))
        .or_else(|| model_list.iter().find_map(model))
        .or_else(|| {
            first_set!(
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
        })
        .or_else(|| unknown("", additional_fields))
}

/// The first deployment `router` holds whose provider the native `/v1/messages` does not serve
/// the way Python does. The request's model is not read before a request is admitted, so one
/// such deployment leaves every request to Python.
pub(super) fn unserved(router: &Router) -> Option<Box<str>> {
    router
        .registry()
        .deployments(&DeploymentFilter::default())
        .iter()
        .map(|deployment| deployment.provider())
        .find(|provider| !MESSAGES_PROVIDERS.contains(provider))
        .map(|provider| format!("model_list.litellm_params.model of provider {provider}").into())
}

/// The database settings are left to `Admit`, which asks whether there is a database at all. The
/// in-flight limits count one worker's requests in Python and the whole gateway's here.
fn general(settings: &GeneralSettings) -> Option<Box<str>> {
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
        enable_jwt_auth,
        enable_oauth2_auth,
        enable_oauth2_proxy_auth,
        allow_user_auth,
        allow_requests_on_db_unavailable,
        allow_client_side_credentials,
        allowed_ips,
        allowed_routes,
        custom_auth,
        litellm_key_header_name,
        litellm_jwtauth,
        role_permissions,
        enforce_rbac,
        user_header_name,
        user_header_mappings,
        additional_fields,
    } = settings;
    first_set!(
        "general_settings.",
        [
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
            enable_jwt_auth,
            enable_oauth2_auth,
            enable_oauth2_proxy_auth,
            allow_user_auth,
            allow_requests_on_db_unavailable,
            allow_client_side_credentials,
            allowed_ips,
            allowed_routes,
            custom_auth,
            litellm_key_header_name,
            litellm_jwtauth,
            role_permissions,
            enforce_rbac,
            user_header_name,
            user_header_mappings,
        ]
    )
    .or_else(|| unknown("general_settings.", additional_fields))
}

/// Every setting router-core reads is its own to honor or refuse when the router is built, so
/// only the ones it never reads are refused here.
fn router(settings: &RouterSettings) -> Option<Box<str>> {
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
    first_set!(
        "router_settings.",
        [
            routing_strategy_args,
            max_retries,
            weights,
            tag_routing_prefix,
        ]
    )
    .or_else(|| other("router_settings.", additional_fields.keys(), &[]))
}

fn litellm(settings: &LiteLlmSettings) -> Option<Box<str>> {
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
    first_set!(
        "litellm_settings.",
        [
            request_timeout,
            num_retries,
            cache,
            cache_params,
            callbacks,
            success_callback,
            failure_callback,
        ]
    )
    .or_else(|| unknown("litellm_settings.", additional_fields))
}

fn model(model: &Model) -> Option<Box<str>> {
    let Model {
        model_name: _,
        litellm_params,
        written_params: _,
        model_info,
        blocked,
        additional_fields,
    } = model;
    params(litellm_params)
        .or_else(|| {
            other(
                "model_list.model_info.",
                model_info.keys(),
                &HONORED_MODEL_INFO,
            )
        })
        .or_else(|| first_set!("model_list.", [blocked]))
        .or_else(|| unknown("model_list.", additional_fields))
}

/// `rpm` and `tpm` are honored as the shuffle's weights. Python also holds a deployment to that
/// many requests in flight, which the gateway does not yet.
fn params(params: &LitellmParams) -> Option<Box<str>> {
    let LitellmParams {
        model: _,
        api_key: _,
        api_base: _,
        api_version: _,
        aws: _,
        vertex: _,
        custom_llm_provider: _,
        timeout: _,
        stream_timeout: _,
        tpm: _,
        rpm: _,
        drop_params: _,
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
    first_set!(
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
            use_in_pass_through,
            use_chat_completions_api,
            litellm_credential_name,
            provider_affinity_header,
        ]
    )
    .or_else(|| {
        other(
            "model_list.litellm_params.",
            extra.keys(),
            &HONORED_EXTRA_PARAMS,
        )
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    const MODEL: &str = "  - model_name: claude\n    litellm_params:\n      model: anthropic/claude-haiku-5-5\n      api_key: provider-key\n";

    fn refused(yaml: &str) -> Option<String> {
        unhonored(&Config::from_yaml(yaml).unwrap()).map(String::from)
    }

    #[rstest]
    fn a_config_of_honored_settings_is_refused_nothing() {
        let yaml = format!(
            "model_list:\n{MODEL}      rpm: 300\n      weight: 2\n      timeout: 30\n    model_info:\n      id: claude-a\nrouter_settings:\n  routing_strategy: simple-shuffle\n  num_retries: 1\n  fallbacks:\n    - claude-down: [claude]\nlitellm_settings:\n  drop_params: true\ngeneral_settings:\n  master_key: gateway-key\n  max_request_size_mb: 1\nenvironment_variables:\n  SOME_KEY: value\n"
        );

        assert_eq!(refused(&yaml), None);
    }

    #[rstest]
    #[case::a_general_setting(
        "general_settings:\n  custom_auth: my.auth\n",
        "general_settings.custom_auth"
    )]
    #[case::a_flag_that_is_on(
        "general_settings:\n  enable_jwt_auth: true\n",
        "general_settings.enable_jwt_auth"
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
        "litellm_settings:\n  callbacks: [prometheus]\n",
        "litellm_settings.callbacks"
    )]
    #[case::an_unknown_litellm_setting(
        "litellm_settings:\n  turn_off_message_logging: true\n",
        "litellm_settings.turn_off_message_logging"
    )]
    #[case::a_section_the_gateway_does_not_run(
        "guardrails:\n  - guardrail_name: pii\n",
        "guardrails"
    )]
    #[case::an_unknown_section("future_section: {}\n", "future_section")]
    fn a_setting_the_native_app_does_not_honor_is_named(#[case] yaml: &str, #[case] named: &str) {
        assert_eq!(refused(yaml).as_deref(), Some(named));
    }

    #[rstest]
    #[case::a_typed_param("      max_retries: 2\n", "model_list.litellm_params.max_retries")]
    #[case::a_param_the_router_does_not_read(
        "      mock_response: hi\n",
        "model_list.litellm_params.mock_response"
    )]
    #[case::model_info_beyond_the_id(
        "    model_info:\n      id: claude-a\n      base_model: claude-haiku-5-5\n",
        "model_list.model_info.base_model"
    )]
    #[case::a_blocked_model("    blocked: true\n", "model_list.blocked")]
    #[case::an_unknown_model_key("    guardrails: [pii]\n", "model_list.guardrails")]
    fn a_deployment_the_native_app_does_not_honor_is_named(
        #[case] written: &str,
        #[case] named: &str,
    ) {
        assert_eq!(
            refused(&format!("model_list:\n{MODEL}{written}")).as_deref(),
            Some(named)
        );
    }

    #[rstest]
    fn a_flag_that_is_off_is_not_a_setting() {
        assert_eq!(
            refused("general_settings:\n  enable_jwt_auth: false\n  enforce_rbac: false\n"),
            None
        );
    }
}
