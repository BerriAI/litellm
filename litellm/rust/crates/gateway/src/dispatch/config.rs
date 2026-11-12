use litellm_config::{
    AdditionalFields, Config, GeneralSettings, LiteLlmSettings, Model, Object, RouterSettings,
};

/// The first thing that puts a deployment outside what the gateway is known to serve the way
/// Python does.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum Unsupported {
    #[error("`{0}` is set")]
    Setting(String),
    #[error("`{0}` is in the environment")]
    Variable(&'static str),
    #[error("`general_settings.master_key` is not set")]
    NoMasterKey,
    #[error("model `{0}` is a wildcard")]
    WildcardModel(String),
    #[error("model `{0}` has several deployments")]
    LoadBalancedModel(String),
}

type Supported = Result<(), Unsupported>;

/// An allowlist: the gateway serves its native routes only for a configuration made of settings
/// it is known to honor the way Python does. Every struct is destructured in full, so a new
/// setting does not compile until it is placed, and anything not placed as safe forwards.
pub(crate) fn supported(config: &Config, env: &dyn Fn(&str) -> Option<String>) -> Supported {
    let Config {
        model_list,
        general_settings,
        router_settings,
        litellm_settings,
        environment_variables,
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
    unset(
        "",
        [
            ("callback_settings", !callback_settings.is_empty()),
            ("assistant_settings", !assistant_settings.is_empty()),
            ("default_vertex_config", !default_vertex_config.is_empty()),
            ("mcp_servers", !mcp_servers.is_empty()),
            ("credential_list", !credential_list.is_empty()),
            ("guardrails", !guardrails.is_empty()),
            ("prompts", !prompts.is_empty()),
            ("sandbox_tools", !sandbox_tools.is_empty()),
            ("search_tools", !search_tools.is_empty()),
            ("files_settings", !files_settings.is_empty()),
            ("finetune_settings", !finetune_settings.is_empty()),
            ("mcp_tools", !mcp_tools.is_empty()),
            ("vector_store_registry", !vector_store_registry.is_empty()),
            ("worker_registry", !worker_registry.is_empty()),
            ("agents", !agents.is_empty()),
            ("agent_list", !agent_list.is_empty()),
            ("policies", !policies.is_empty()),
            ("policy_attachments", !policy_attachments.is_empty()),
        ],
    )?;
    only("", additional_fields, &[])?;
    environment(environment_variables, env)?;
    general(general_settings)?;
    router(router_settings)?;
    litellm(litellm_settings)?;
    models(model_list)
}

fn unset<const N: usize>(section: &str, settings: [(&str, bool); N]) -> Supported {
    match settings.into_iter().find(|(_, set)| *set) {
        Some((name, _)) => Err(Unsupported::Setting(format!("{section}{name}"))),
        None => Ok(()),
    }
}

fn only(section: &str, fields: &AdditionalFields, known: &[&str]) -> Supported {
    match fields.keys().find(|key| !known.contains(&key.as_str())) {
        Some(key) => Err(Unsupported::Setting(format!("{section}{key}"))),
        None => Ok(()),
    }
}

fn listed<T>(values: &Option<Vec<T>>) -> bool {
    values.as_ref().is_some_and(|values| !values.is_empty())
}

/// The environment can't be allowlisted, since Python reads far more variables than anything
/// lists. These are the ones known to change how a native route would have to answer: a database
/// means keys, spend logs and DB-stored settings the gateway has none of.
fn environment(variables: &Object, env: &dyn Fn(&str) -> Option<String>) -> Supported {
    let in_config = |name: &str| variables.contains_key(name);
    let database = ["DATABASE_URL", "DIRECT_URL", "DATABASE_HOST"]
        .into_iter()
        .find(|name| env(name).is_some() || in_config(name));
    let root_path = (env("SERVER_ROOT_PATH").is_some_and(|path| !path.is_empty())
        || in_config("SERVER_ROOT_PATH"))
    .then_some("SERVER_ROOT_PATH");
    match database.or(root_path) {
        Some(name) => Err(Unsupported::Variable(name)),
        None => Ok(()),
    }
}

fn general(settings: &GeneralSettings) -> Supported {
    let GeneralSettings {
        master_key,
        // The auth core plans with these, and the gate forwards every caller it does not
        // authenticate as the master key.
        enable_jwt_auth: _,
        enable_oauth2_auth: _,
        enable_oauth2_proxy_auth: _,
        allow_user_auth: _,
        litellm_jwtauth: _,
        role_permissions: _,
        enforce_rbac: _,
        // Read only with a database, by Python's background work, or by routes Python serves.
        allow_requests_on_db_unavailable: _,
        database_connection_pool_limit: _,
        database_connection_timeout: _,
        database_connect_timeout: _,
        database_socket_timeout: _,
        database_max_idle_connection_lifetime: _,
        admission_queue_timeout_seconds: _,
        proxy_config_reload_interval_seconds: _,
        background_health_checks: _,
        health_check_interval: _,
        health_check_concurrency: _,
        enable_public_model_hub: _,
        dangerously_permit_weak_or_unset_master_key: _,
        mcp_allowed_hosts: _,
        mcp_allowed_origins: _,
        completion_model,
        max_in_flight_requests_per_worker,
        max_queued_requests_per_worker,
        database_url,
        tracing,
        max_parallel_requests,
        global_max_parallel_requests,
        max_request_size_mb,
        max_response_size_mb,
        store_model_in_db,
        forward_client_headers_to_llm_api,
        cancel_on_disconnect,
        infer_model_from_keys,
        plugins,
        coordination_redis,
        public_routes,
        allowed_ips,
        allowed_routes,
        custom_auth,
        litellm_key_header_name,
        pass_through_endpoints,
        user_header_name,
        user_header_mappings,
        additional_fields,
    } = settings;
    if master_key.is_none() {
        return Err(Unsupported::NoMasterKey);
    }
    unset(
        "general_settings.",
        [
            ("completion_model", completion_model.is_some()),
            (
                "max_in_flight_requests_per_worker",
                max_in_flight_requests_per_worker.is_some(),
            ),
            (
                "max_queued_requests_per_worker",
                max_queued_requests_per_worker.is_some(),
            ),
            ("database_url", database_url.is_some()),
            ("tracing", tracing.is_some()),
            ("max_parallel_requests", max_parallel_requests.is_some()),
            (
                "global_max_parallel_requests",
                global_max_parallel_requests.is_some(),
            ),
            ("max_request_size_mb", max_request_size_mb.is_some()),
            ("max_response_size_mb", max_response_size_mb.is_some()),
            ("store_model_in_db", store_model_in_db.is_some()),
            (
                "forward_client_headers_to_llm_api",
                forward_client_headers_to_llm_api.is_some(),
            ),
            ("cancel_on_disconnect", cancel_on_disconnect.is_some()),
            ("infer_model_from_keys", infer_model_from_keys.is_some()),
            ("plugins", plugins.is_some()),
            ("coordination_redis", coordination_redis.is_some()),
            (
                "public_routes",
                public_routes
                    .as_deref()
                    .is_some_and(|routes| !routes.is_empty()),
            ),
            ("allowed_ips", allowed_ips.is_some()),
            ("allowed_routes", allowed_routes.is_some()),
            ("custom_auth", custom_auth.is_some()),
            ("litellm_key_header_name", litellm_key_header_name.is_some()),
            ("pass_through_endpoints", pass_through_endpoints.is_some()),
            ("user_header_name", user_header_name.is_some()),
            ("user_header_mappings", !user_header_mappings.is_empty()),
        ],
    )?;
    only("general_settings.", additional_fields, &[])
}

fn router(settings: &RouterSettings) -> Supported {
    let RouterSettings {
        routing_strategy,
        routing_strategy_args,
        routing_groups,
        retry_policy,
        model_group_retry_policy,
        model_group_affinity_config,
        allowed_fails,
        cooldown_time,
        num_retries,
        timeout,
        max_retries,
        retry_after,
        fallbacks,
        context_window_fallbacks,
        model_group_alias,
        enable_tag_filtering,
        weights,
        tag_routing_prefix,
        optional_pre_call_checks,
        additional_fields,
    } = settings;
    unset(
        "router_settings.",
        [
            ("routing_strategy", routing_strategy.is_some()),
            ("routing_strategy_args", routing_strategy_args.is_some()),
            ("routing_groups", routing_groups.is_some()),
            ("retry_policy", retry_policy.is_some()),
            (
                "model_group_retry_policy",
                model_group_retry_policy.is_some(),
            ),
            (
                "model_group_affinity_config",
                model_group_affinity_config.is_some(),
            ),
            ("allowed_fails", allowed_fails.is_some()),
            ("cooldown_time", cooldown_time.is_some()),
            ("num_retries", num_retries.is_some()),
            ("timeout", timeout.is_some()),
            ("max_retries", max_retries.is_some()),
            ("retry_after", retry_after.is_some()),
            ("fallbacks", fallbacks.is_some()),
            (
                "context_window_fallbacks",
                context_window_fallbacks.is_some(),
            ),
            ("model_group_alias", model_group_alias.is_some()),
            ("enable_tag_filtering", enable_tag_filtering.is_some()),
            ("weights", weights.is_some()),
            ("tag_routing_prefix", tag_routing_prefix.is_some()),
            (
                "optional_pre_call_checks",
                optional_pre_call_checks.is_some(),
            ),
        ],
    )?;
    only("router_settings.", additional_fields, &[])
}

fn litellm(settings: &LiteLlmSettings) -> Supported {
    let LiteLlmSettings {
        // `build_inference` and the model router read these.
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
        // Python reads `cache_params` only when `cache` is on, and the last two only shape logs.
        cache_params: _,
        json_logs: _,
        set_verbose: _,
        request_timeout,
        num_retries,
        cache,
        callbacks,
        success_callback,
        failure_callback,
        additional_fields,
    } = settings;
    unset(
        "litellm_settings.",
        [
            ("request_timeout", request_timeout.is_some()),
            ("num_retries", num_retries.is_some()),
            ("cache", *cache == Some(true)),
            ("callbacks", listed(callbacks)),
            ("success_callback", listed(success_callback)),
            ("failure_callback", listed(failure_callback)),
        ],
    )?;
    only(
        "litellm_settings.",
        additional_fields,
        &["require_auth_for_metrics_endpoint"],
    )
}

fn models(models: &[Model]) -> Supported {
    models.iter().enumerate().try_for_each(|(index, model)| {
        let Model {
            model_name,
            litellm_params: _,
            model_info,
            blocked,
            additional_fields,
        } = model;
        if model_name.contains('*') {
            return Err(Unsupported::WildcardModel(model_name.clone()));
        }
        if models[..index]
            .iter()
            .any(|earlier| earlier.model_name == *model_name)
        {
            return Err(Unsupported::LoadBalancedModel(model_name.clone()));
        }
        unset(
            "model_list.",
            [
                ("model_info", !model_info.is_empty()),
                ("blocked", blocked.is_some()),
            ],
        )?;
        only("model_list.", additional_fields, &[])
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    fn with_general(line: &str) -> Config {
        with_section(&format!("  {line}"))
    }

    fn with_section(yaml: &str) -> Config {
        Config::from_yaml(&format!(
            "model_list: []\ngeneral_settings:\n  master_key: sk-master\n{yaml}\n"
        ))
        .unwrap()
    }

    fn setting(name: &str) -> Supported {
        Err(Unsupported::Setting(name.to_owned()))
    }

    #[rstest]
    #[case::master_key_only("")]
    #[case::null_setting("allowed_ips: null")]
    #[case::empty_public_routes("public_routes: []")]
    #[case::jwt_auth("enable_jwt_auth: true")]
    #[case::health_checks("background_health_checks: true")]
    fn serves_general_settings_on_the_allowlist(#[case] line: &str) {
        assert_eq!(supported(&with_general(line), &no_env), Ok(()));
    }

    #[rstest]
    #[case::allowed_ips("allowed_ips: ['10.0.0.1']", "general_settings.allowed_ips")]
    #[case::allowed_routes(
        "allowed_routes: ['/v1/chat/completions']",
        "general_settings.allowed_routes"
    )]
    #[case::custom_auth("custom_auth: my_module.auth", "general_settings.custom_auth")]
    #[case::key_header_name(
        "litellm_key_header_name: x-my-key",
        "general_settings.litellm_key_header_name"
    )]
    #[case::pass_through_endpoints(
        "pass_through_endpoints: []",
        "general_settings.pass_through_endpoints"
    )]
    #[case::max_request_size("max_request_size_mb: 10", "general_settings.max_request_size_mb")]
    #[case::database_url("database_url: postgresql://db", "general_settings.database_url")]
    #[case::public_routes("public_routes: ['/v1/models']", "general_settings.public_routes")]
    #[case::parallel_requests("max_parallel_requests: 5", "general_settings.max_parallel_requests")]
    #[case::unknown("alerting: ['slack']", "general_settings.alerting")]
    fn forwards_a_general_setting_off_the_allowlist(#[case] line: &str, #[case] name: &str) {
        assert_eq!(supported(&with_general(line), &no_env), setting(name));
    }

    #[rstest]
    #[case::provider_transport("litellm_settings: {drop_params: true, ssl_verify: false}")]
    #[case::metrics_auth("litellm_settings: {require_auth_for_metrics_endpoint: false}")]
    #[case::cache_off("litellm_settings: {cache: false}")]
    #[case::no_callbacks("litellm_settings: {callbacks: []}")]
    #[case::variables("environment_variables: {OPENAI_API_KEY: sk-provider}")]
    fn serves_sections_on_the_allowlist(#[case] section: &str) {
        assert_eq!(supported(&with_section(section), &no_env), Ok(()));
    }

    #[rstest]
    #[case::guardrails("guardrails: [{guardrail_name: pii}]", "guardrails")]
    #[case::mcp_servers("mcp_servers: {docs: {url: 'https://mcp.example/mcp'}}", "mcp_servers")]
    #[case::callbacks(
        "litellm_settings: {callbacks: [langfuse]}",
        "litellm_settings.callbacks"
    )]
    #[case::one_success_callback(
        "litellm_settings: {success_callback: langfuse}",
        "litellm_settings.success_callback"
    )]
    #[case::cache("litellm_settings: {cache: true}", "litellm_settings.cache")]
    #[case::retries("litellm_settings: {num_retries: 3}", "litellm_settings.num_retries")]
    #[case::unknown_litellm_setting(
        "litellm_settings: {max_budget: 10}",
        "litellm_settings.max_budget"
    )]
    #[case::fallbacks(
        "router_settings: {fallbacks: [{a: [b]}]}",
        "router_settings.fallbacks"
    )]
    #[case::unknown_router_setting(
        "router_settings: {redis_host: localhost}",
        "router_settings.redis_host"
    )]
    #[case::unknown_section("some_future_section: {enabled: true}", "some_future_section")]
    fn forwards_a_section_off_the_allowlist(#[case] section: &str, #[case] name: &str) {
        assert_eq!(supported(&with_section(section), &no_env), setting(name));
    }

    #[rstest]
    #[case::one_deployment_each(
        "[{model_name: a, litellm_params: {model: openai/a}}, {model_name: b, litellm_params: {model: openai/b}}]",
        Ok(())
    )]
    #[case::load_balanced(
        "[{model_name: a, litellm_params: {model: openai/a}}, {model_name: a, litellm_params: {model: openai/b}}]",
        Err(Unsupported::LoadBalancedModel("a".into()))
    )]
    #[case::wildcard(
        "[{model_name: 'openai/*', litellm_params: {model: 'openai/*'}}]",
        Err(Unsupported::WildcardModel("openai/*".into()))
    )]
    #[case::model_info(
        "[{model_name: a, litellm_params: {model: openai/a}, model_info: {base_model: gpt}}]",
        setting("model_list.model_info")
    )]
    fn serves_only_models_with_one_plain_deployment(
        #[case] models: &str,
        #[case] expected: Supported,
    ) {
        let config = Config::from_yaml(&format!(
            "model_list: {models}\ngeneral_settings:\n  master_key: sk-master\n"
        ))
        .unwrap();
        assert_eq!(supported(&config, &no_env), expected);
    }

    #[rstest]
    #[case::database_url(
        "DATABASE_URL",
        "postgresql://db",
        Err(Unsupported::Variable("DATABASE_URL"))
    )]
    #[case::direct_url(
        "DIRECT_URL",
        "postgresql://db",
        Err(Unsupported::Variable("DIRECT_URL"))
    )]
    #[case::database_host("DATABASE_HOST", "db", Err(Unsupported::Variable("DATABASE_HOST")))]
    #[case::server_root_path(
        "SERVER_ROOT_PATH",
        "/api",
        Err(Unsupported::Variable("SERVER_ROOT_PATH"))
    )]
    #[case::empty_server_root_path("SERVER_ROOT_PATH", "", Ok(()))]
    #[case::unrelated("PORT", "4000", Ok(()))]
    fn forwards_when_the_environment_names_a_database_or_root_path(
        #[case] name: &str,
        #[case] value: &str,
        #[case] expected: Supported,
    ) {
        let env = |key: &str| (key == name).then(|| value.to_owned());
        assert_eq!(supported(&with_general(""), &env), expected);
    }

    #[rstest]
    fn a_database_named_in_the_config_environment_forwards() {
        let config = with_section("environment_variables: {DATABASE_URL: postgresql://db}");
        assert_eq!(
            supported(&config, &no_env),
            Err(Unsupported::Variable("DATABASE_URL"))
        );
    }

    #[rstest]
    fn no_master_key_forwards() {
        let config = Config::from_yaml("model_list: []\n").unwrap();
        assert_eq!(supported(&config, &no_env), Err(Unsupported::NoMasterKey));
    }

    // TODO: serve this natively. It is a production deployment's configuration cut down to one of
    // each feature it uses, and that deployment also runs with a database. Each feature the
    // gateway learns to honor leaves the allowlist's blockers, and when none is left this
    // becomes `Ok(())`.
    #[rstest]
    fn a_production_shaped_configuration_forwards() {
        let config = Config::from_yaml(PRODUCTION_SHAPED).unwrap();
        assert!(supported(&config, &no_env).is_err());
    }

    const PRODUCTION_SHAPED: &str = r"
model_list:
  - model_name: azure/anthropic/claude-sonnet-5-5
    litellm_params:
      model: azure_ai/claude-sonnet-5-5
      api_base: https://azure.example.invalid/anthropic
      api_key: os.environ/AZURE_API_KEY
      cache_control_injection_points:
        - {location: message, role: system, control: {type: ephemeral, ttl: 1h}}
        - {location: message, index: -1, control: {type: ephemeral, ttl: 5m}}
      tpm: 220000000
      rpm: 220000
      tags: [azure]
    model_info:
      id: azure/anthropic/claude-sonnet-5-5
      mode: chat
      base_model: azure_ai/claude-sonnet-5-5
      access_groups: [default-models]
  - model_name: azure/openai/gpt-5.5
    litellm_params:
      model: azure/gpt-5.5
      api_base: https://east.example.invalid/
      api_version: v1
      api_key: os.environ/AZURE_EAST_API_KEY
      weight: 1
    model_info:
      id: azure/openai/gpt-5.5-east
      mode: responses
      base_model: azure/gpt-5.5
      access_groups: [default-models]
  - model_name: azure/openai/gpt-5.5
    litellm_params:
      model: azure/gpt-5.5
      api_base: https://west.example.invalid/
      api_version: v1
      api_key: os.environ/AZURE_WEST_API_KEY
      weight: 1
    model_info:
      id: azure/openai/gpt-5.5-west
      mode: responses
      base_model: azure/gpt-5.5
      access_groups: [default-models]
  - model_name: aws/anthropic/claude-sonnet-5-5
    litellm_params:
      model: bedrock/anthropic.claude-sonnet-5-5
      aws_region_name: us-west-2
      tpm: 200000000
      rpm: 10500
    model_info:
      id: aws/anthropic/claude-sonnet-5-5
      mode: chat
      access_groups: [default-models]
  - model_name: gcp/google/gemini-3.1-pro-preview
    litellm_params:
      model: vertex_ai/gemini-3.1-pro-preview
      vertex_project: example-project
      vertex_location: global
      vertex_credentials: os.environ/VERTEX_CREDENTIALS
      use_in_pass_through: true
    model_info:
      id: gcp/google/gemini-3.1-pro-preview
      mode: chat
      input_cost_per_token: 0.000001
      output_cost_per_token: 0.000002
      access_groups: [default-models]
    metadata:
      catalog_id: example
  - model_name: hosted/open-model
    litellm_params:
      model: openai/open-model
      api_base: https://hosted.example.invalid/v1
      api_key: os.environ/HOSTED_API_KEY
      custom_llm_provider: openai
      additional_drop_params: [stream_options]
      extra_headers: {x-route: hosted}
    model_info:
      id: hosted/open-model
      mode: chat
      max_input_tokens: 128000
      supports_function_calling: true
      access_groups: [default-models]
search_tools:
  - search_tool_name: web-search
    litellm_params:
      search_provider: perplexity
      api_key: os.environ/SEARCH_API_KEY
general_settings:
  master_key: os.environ/PROXY_MASTER_KEY
  disable_budget_reservation: true
  store_prompts_in_spend_logs: false
  use_redis_transaction_buffer: true
  forward_client_headers_to_llm_api: true
  token_rate_limit_type: total
  allow_requests_on_db_unavailable: true
  max_request_size_mb: 500
  database_connection_pool_limit: 10
  user_api_key_cache_ttl: 600
  user_api_key_cache_max_size: 500
  disable_error_logs: false
  proxy_batch_write_at: 60
  store_model_in_db: false
  use_x_forwarded_for: true
  maximum_spend_logs_retention_period: 10d
  maximum_spend_logs_retention_interval: 30m
  enable_jwt_auth: true
  litellm_jwtauth:
    user_id_jwt_field: oid
    user_email_jwt_field: upn
    team_ids_jwt_field: roles
    user_id_upsert: true
    enforce_team_based_model_access: true
    team_allowed_routes: [openai_routes, info_routes, mcp_routes, anthropic_routes]
  pass_through_endpoints:
    - path: /v1/embeddings/hosted/open-embedding
      target: https://embeddings.example.invalid/v1/embeddings
      auth: true
      methods: [POST]
      headers:
        authorization: Bearer os.environ/HOSTED_API_KEY
        content-type: application/json
      forward_headers: true
    - path: /v1/solver/solution
      target: https://solver.example.invalid/solution
      auth: true
      include_subpath: true
      forward_headers: true
litellm_settings:
  return_response_headers: true
  cost_discount_config: {vertex_ai: 0.1, openai: 0.1, azure: 0.1, bedrock: 0.1}
  cache: true
  enable_redis_auth_cache: true
  cache_params:
    type: redis
    ssl: true
    ttl: 600
    password: os.environ/REDIS_PASSWORD
    max_connections: 20
    supported_call_types: []
    redis_startup_nodes:
      - {host: os.environ/REDIS_HOST, port: os.environ/REDIS_PORT}
  callbacks:
    - /etc/litellm-callbacks/request_metadata.proxy_handler_instance
    - dynamic_rate_limiter_v3
    - prometheus
    - s3_v2
  success_callback:
    - /etc/litellm-callbacks/spend_metadata.proxy_handler_instance
  service_callback: [prometheus_system]
  s3_callback_params:
    s3_bucket_name: example-logging-bucket
    s3_region_name: us-west-2
    s3_path: input
    s3_use_team_prefix: true
  require_auth_for_metrics_endpoint: false
  prometheus_emit_stream_label: true
  prometheus_initialize_budget_metrics: false
  prometheus_exclude_metrics: [litellm_llm_api_latency_metric]
  prometheus_exclude_labels: [client_ip, end_user, hashed_api_key]
  include_cost_in_streaming_usage: true
  drop_params: true
  json_logs: true
  priority_reservation: {dev: 0.2, prod: 0.8}
  priority_reservation_settings:
    default_priority: 0.2
    saturation_threshold: 0.6
    saturation_check_cache_ttl: 10
  request_timeout: 360
  set_verbose: false
  store_audit_logs: true
router_settings:
  optional_pre_call_checks: [enforce_model_rate_limits, encrypted_content_affinity]
  model_group_affinity_config:
    azure/openai/gpt-5.5: [responses_api_deployment_check, encrypted_content_affinity]
  enable_pre_call_checks: true
  redis_host: os.environ/REDIS_HOST
  redis_password: os.environ/REDIS_PASSWORD
  redis_port: os.environ/REDIS_PORT
  routing_strategy: simple-shuffle
  disable_cooldowns: true
  model_group_alias:
    gcp/google/gemini-pro: gcp/google/gemini-3.1-pro-preview
  num_retries: 0
  retry_policy:
    TimeoutErrorRetries: 0
    RateLimitErrorRetries: 0
    InternalServerErrorRetries: 0
";
}
