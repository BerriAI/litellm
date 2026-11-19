use litellm_config::{Config, Error, RouteList, Spelled, TracingStoreSettings};
use rstest::rstest;

#[rstest]
fn config_debug_redacts_api_keys() {
    let config = Config::from_yaml(
        "model_list: [{model_name: assistant, litellm_params: {model: anthropic/test-model, api_key: secret-value}}]",
    )
    .unwrap();
    assert_eq!(
        config.model_list[0]
            .litellm_params
            .api_key
            .as_ref()
            .unwrap()
            .expose(),
        "secret-value"
    );
    assert!(!format!("{config:?}").contains("secret-value"));
}

#[rstest]
fn rejects_malformed_yaml() {
    assert!(matches!(
        Config::from_yaml("model_list: ["),
        Err(Error::Load(_))
    ));
}

#[rstest]
#[case::missing_params("model_list: [{model_name: assistant}]")]
#[case::missing_model("model_list: [{model_name: assistant, litellm_params: {api_key: key}}]")]
#[case::provider_param_of_the_wrong_type(
    "model_list: [{model_name: assistant, litellm_params: {model: test, aws_region_name: [eu-central-1]}}]"
)]
fn rejects_malformed_and_incomplete_config(#[case] yaml: &str) {
    assert!(matches!(Config::from_yaml(yaml), Err(Error::Parse(_))));
}

#[rstest]
#[case::literal("secret-master-key")]
#[case::reference("os.environ/LITELLM_MASTER_KEY")]
fn loads_and_redacts_the_master_key(#[case] key: &str) {
    let config = Config::from_yaml(&format!(
        "model_list: []\ngeneral_settings:\n  master_key: {key}\n"
    ))
    .unwrap();
    assert_eq!(
        config
            .general_settings
            .master_key
            .as_ref()
            .unwrap()
            .expose(),
        key
    );
    assert!(!format!("{config:?}").contains(key));
}

#[rstest]
fn loads_and_redacts_the_litellm_license() {
    let config =
        Config::from_yaml("model_list: []\ngeneral_settings:\n  litellm_license: secret-license\n")
            .unwrap();
    assert_eq!(
        config
            .general_settings
            .litellm_license
            .as_ref()
            .unwrap()
            .expose(),
        "secret-license"
    );
    assert!(!format!("{config:?}").contains("secret-license"));
}

#[rstest]
fn missing_general_settings_has_no_master_key() {
    let config = Config::from_yaml("model_list: []").unwrap();
    assert!(config.general_settings.master_key.is_none());
}

#[test]
fn tracing_settings_are_typed_and_redact_the_url() {
    let config = Config::from_yaml(
        "general_settings:\n  tracing:\n    store:\n      type: clickhouse\n      url: https://writer:password@example.com\n      database: analytics\n      retention_days: 7\n",
    )
    .unwrap();
    let tracing = config.general_settings.tracing.as_ref().unwrap();
    let Some(TracingStoreSettings::ClickHouse(store)) = tracing.store.as_ref() else {
        panic!("expected ClickHouse tracing store")
    };
    assert_eq!(
        store.url.as_ref().unwrap().expose(),
        "https://writer:password@example.com"
    );
    assert_eq!(store.database.as_deref(), Some("analytics"));
    assert_eq!(store.retention_days, Some(Spelled::Value(7.0)));
    assert!(!format!("{config:?}").contains("password"));
}

#[test]
fn tracing_settings_accept_environment_references() {
    let config = Config::from_yaml(
        "general_settings:\n  tracing:\n    store:\n      type: clickhouse\n      url: os.environ/CLICKHOUSE_URL\n      retention_days: os.environ/RETENTION_DAYS\n",
    )
    .unwrap();
    let Some(TracingStoreSettings::ClickHouse(store)) =
        config.general_settings.tracing.unwrap().store
    else {
        panic!("expected ClickHouse tracing store")
    };
    assert_eq!(
        store.retention_days,
        Some(Spelled::Text("os.environ/RETENTION_DAYS".to_owned()))
    );
}

#[test]
fn tracing_settings_reject_string_store() {
    assert!(Config::from_yaml("general_settings:\n  tracing:\n    store: clickhouse\n").is_err());
}

#[test]
fn tracing_settings_reject_removed_reader_configuration() {
    assert!(Config::from_yaml("general_settings:\n  tracing:\n    store:\n      type: clickhouse\n      reader_url: http://localhost:8123\n").is_err());
}

#[rstest]
fn empty_config_matches_python_defaults() {
    let config = Config::from_yaml("{}").unwrap();
    assert!(config.model_list.is_empty());
    assert_eq!(config.general_settings.admission_queue_timeout_seconds, 1.0);
    assert_eq!(
        config.general_settings.database_connection_pool_limit,
        Some(10)
    );
    assert_eq!(
        config.general_settings.proxy_config_reload_interval_seconds,
        30
    );
    assert_eq!(config.general_settings.health_check_interval, 300);
}

#[rstest]
fn parses_typed_settings_and_preserves_extension_fields() {
    let config = Config::from_yaml(
        r#"
model_list:
  - model_name: assistant
    litellm_params:
      model: vertex_ai/test-model
      timeout: os.environ/REQUEST_TIMEOUT
      tpm: os.environ/TPM_LIMIT
      rpm: 5
      drop_params: "true"
      vertex_project: test-project
      vertex_credentials:
        type: service_account
      future_param: 1
    model_info:
      mode: chat
    access_groups: [internal]
general_settings:
  master_key: secret-master-key
  store_model_in_db: true
  custom_auth: auth.py
router_settings:
  routing_strategy: simple-shuffle
  allowed_fails: 2
  redis_host: cache.internal
litellm_settings:
  drop_params: true
  cache: true
  custom_callback_name: audit
future_section:
  enabled: true
"#,
    )
    .unwrap();

    let model = &config.model_list[0];
    assert_eq!(
        model.litellm_params.timeout,
        Some(Spelled::Text("os.environ/REQUEST_TIMEOUT".to_string()))
    );
    assert_eq!(
        model.litellm_params.tpm,
        Some(Spelled::Text("os.environ/TPM_LIMIT".to_string()))
    );
    assert_eq!(model.litellm_params.rpm, Some(Spelled::Value(5.0)));
    assert_eq!(
        model.litellm_params.drop_params,
        Some(Spelled::Text("true".to_string()))
    );
    assert_eq!(
        model.litellm_params.vertex.vertex_project.as_deref(),
        Some("test-project")
    );
    assert_eq!(
        model.litellm_params.vertex.vertex_credentials.as_deref(),
        Some(r#"{"type":"service_account"}"#)
    );
    assert!(model.litellm_params.extra.contains_key("future_param"));
    assert!(model.additional_fields.contains_key("access_groups"));
    assert_eq!(config.router_settings.allowed_fails, Some(2));
    assert!(
        config
            .router_settings
            .additional_fields
            .contains_key("redis_host")
    );
    assert!(
        config
            .litellm_settings
            .additional_fields
            .contains_key("custom_callback_name")
    );
    assert!(config.additional_fields.contains_key("future_section"));
}

#[rstest]
fn parses_python_config_sections() {
    let config = Config::from_yaml(
        r#"
environment_variables:
  REDIS_PORT: 6379
callback_settings:
  otel:
    message_logging: false
assistant_settings:
  custom_llm_provider: openai
credential_list:
  - credential_name: bedrock
    credential_values:
      aws_region_name: us-east-1
guardrails:
  - guardrail_name: pii
    litellm_params:
      guardrail: presidio
prompts:
  - prompt_id: support
sandbox_tools:
  - sandbox_tool_name: e2b
search_tools:
  - search_tool_name: web
files_settings:
  - custom_llm_provider: openai
finetune_settings:
  - custom_llm_provider: openai
mcp_tools:
  - name: lookup
mcp_servers:
  docs:
    url: https://example.test/mcp
vector_store_registry:
  - vector_store_name: docs
worker_registry:
  - worker_id: regional
agents:
  - agent_name: reviewer
agent_list:
  - agent_name: legacy-reviewer
policies:
  safe:
    guardrails:
      add: [pii]
policy_attachments:
  - policy_id: safe
include:
  - models.yaml
"#,
    )
    .unwrap();

    assert_eq!(config.environment_variables.len(), 1);
    assert_eq!(config.credential_list.len(), 1);
    assert_eq!(config.guardrails.len(), 1);
    assert_eq!(config.prompts.len(), 1);
    assert_eq!(config.sandbox_tools.len(), 1);
    assert_eq!(config.search_tools.len(), 1);
    assert_eq!(config.files_settings.len(), 1);
    assert_eq!(config.finetune_settings.len(), 1);
    assert_eq!(config.mcp_tools.len(), 1);
    assert_eq!(config.mcp_servers.len(), 1);
    assert_eq!(config.vector_store_registry.len(), 1);
    assert_eq!(config.worker_registry.len(), 1);
    assert_eq!(config.agents.len(), 1);
    assert_eq!(config.agent_list.len(), 1);
    assert!(config.policies.contains_key("safe"));
    assert_eq!(config.policy_attachments.len(), 1);
    assert_eq!(&*config.include, &["models.yaml"]);
}

#[rstest]
#[case::one("callbacks: custom_callbacks.logger", 1)]
#[case::many("callbacks: [prometheus, otel]", 2)]
fn accepts_python_callback_shorthand(#[case] setting: &str, #[case] expected_len: usize) {
    let config = Config::from_yaml(&format!("litellm_settings:\n  {setting}")).unwrap();
    assert_eq!(
        config.litellm_settings.callbacks.as_ref().unwrap().len(),
        expected_len
    );
}

#[rstest]
#[case::api_key("api_key", "provider-secret")]
#[case::aws_secret("aws_secret_access_key", "aws-secret")]
#[case::vertex_credentials("vertex_credentials", "vertex-secret")]
#[case::provider_extension("azure_ad_token", "azure-secret")]
#[case::general_extension("custom_auth_secret", "auth-secret")]
#[case::router_extension("redis_password", "redis-secret")]
#[case::litellm_extension("callback_token", "callback-secret")]
#[case::root_extension("private_token", "root-secret")]
fn debug_output_does_not_expose_config_values(#[case] field: &str, #[case] secret: &str) {
    let yaml = match field {
        "api_key" => format!(
            "model_list: [{{model_name: assistant, litellm_params: {{model: test, api_key: {secret}}}}}]"
        ),
        "aws_secret_access_key" => format!(
            "model_list: [{{model_name: assistant, litellm_params: {{model: test, aws_secret_access_key: {secret}}}}}]"
        ),
        "vertex_credentials" => format!(
            "model_list: [{{model_name: assistant, litellm_params: {{model: test, vertex_credentials: {secret}}}}}]"
        ),
        "azure_ad_token" => format!(
            "model_list: [{{model_name: assistant, litellm_params: {{model: test, azure_ad_token: {secret}}}}}]"
        ),
        "custom_auth_secret" => format!("general_settings: {{{field}: {secret}}}"),
        "redis_password" => format!("router_settings: {{{field}: {secret}}}"),
        "callback_token" => format!("litellm_settings: {{{field}: {secret}}}"),
        "private_token" => format!("{field}: {secret}"),
        _ => unreachable!(),
    };
    let config = Config::from_yaml(&yaml).unwrap();
    assert!(!format!("{config:?}").contains(secret));
}

#[rstest]
fn mcp_config_redacts_nested_credentials_and_preserves_policy_for_validation() {
    let config = Config::from_yaml("mcp_servers:\n  docs:\n    url: https://example.test/private-secret/mcp\n    authentication_token: upstream-secret\n    static_headers: {x-token: header-secret}\n    env: {TOKEN: env-secret}\n    args: [argument-secret]\n    client_secret: oauth-secret\n    allowed_tools: [search]\n").unwrap();
    let server = &config.mcp_servers["docs"];
    assert_eq!(server.allowed_tools.as_deref().unwrap(), ["search"]);
    assert!(server.unsupported.contains_key("client_secret"));
    let debug = format!("{config:?}");
    for secret in [
        "private-secret",
        "upstream-secret",
        "header-secret",
        "env-secret",
        "oauth-secret",
        "argument-secret",
    ] {
        assert!(!debug.contains(secret));
    }
}

#[rstest]
#[case::transport("transport: invalid")]
#[case::auth("auth_type: invalid")]
#[case::concurrency("max_concurrent_requests: -1")]
#[case::headers("static_headers: {x-token: [not, a, string]}")]
fn rejects_invalid_typed_mcp_settings(#[case] setting: &str) {
    assert!(Config::from_yaml(&format!("mcp_servers:\n  docs:\n    {setting}\n")).is_err());
}

#[rstest]
fn inactive_redis_cache_keeps_malformed_known_inputs_deferred() {
    let config = Config::from_yaml(
        "litellm_settings:\n  cache: false\n  cache_params:\n    type: redis\n    host: []\n    port: invalid\n    socket_timeout: invalid",
    )
    .unwrap();

    assert!(matches!(
        config.litellm_settings.cache_params,
        Some(litellm_config::CacheSettings::Redis(_))
    ));
}

#[rstest]
fn non_redis_cache_settings_remain_opaque() {
    let config = Config::from_yaml(
        "litellm_settings:\n  cache_params:\n    type: local\n    custom_option:\n      nested: true",
    )
    .unwrap();

    assert!(matches!(
        config.litellm_settings.cache_params,
        Some(litellm_config::CacheSettings::Other(_))
    ));
}

fn pass_through(entry: &str) -> litellm_config::PassThroughEndpoint {
    let config = Config::from_yaml(&format!(
        "general_settings: {{pass_through_endpoints: [{entry}]}}"
    ))
    .unwrap();
    config.general_settings.pass_through_endpoints.unwrap()[0].clone()
}

#[rstest]
fn a_pass_through_endpoint_reads_its_python_fields() {
    let endpoint = pass_through(
        "{path: '/bria/{item_id:int}', target: https://upstream.example/v1, include_subpath: true, \
         headers: {Authorization: os.environ/BRIA_KEY}, cost_per_request: '0.5', methods: [POST], \
         guardrails: {pii: {request_fields: [text]}, toxicity: null}, custom_auth_parser: langfuse, display_name: Bria}",
    );
    assert_eq!(
        endpoint.path,
        litellm_starlette::RouteTemplate::parse("/bria/{item_id:int}")
    );
    assert_eq!(
        endpoint.target.unwrap().expose(),
        "https://upstream.example/v1"
    );
    assert!(endpoint.include_subpath);
    assert_eq!(
        endpoint.headers.keys().collect::<Vec<_>>(),
        ["Authorization"]
    );
    assert_eq!(endpoint.methods.as_deref(), Some(&["POST".to_owned()][..]));
    let guardrails = endpoint.guardrails.unwrap();
    assert_eq!(
        guardrails["pii"]
            .as_ref()
            .unwrap()
            .request_fields
            .as_deref(),
        Some(&["text".to_owned()][..])
    );
    assert!(guardrails["toxicity"].is_none());
    assert_eq!(endpoint.custom_auth_parser.as_deref(), Some("langfuse"));
    assert!(endpoint.additional_fields.contains_key("display_name"));
}

/// `pass_through_auth_enforced(endpoint)`, `endpoint.get("include_subpath", False) is True` and
/// `bool(forward_headers)`, as the proxy reads them.
#[rstest]
#[case::nothing_set("", true, false, false)]
#[case::booleans(
    "auth: false, include_subpath: true, forward_headers: true",
    false,
    true,
    true
)]
#[case::strings(
    "auth: 'true', include_subpath: 'true', forward_headers: 'false'",
    true,
    false,
    true
)]
#[case::numbers("auth: 1, include_subpath: 1, forward_headers: 0", true, false, false)]
fn pass_through_flags_read_like_python(
    #[case] flags: &str,
    #[case] auth: bool,
    #[case] include_subpath: bool,
    #[case] forward_headers: bool,
) {
    let separator = if flags.is_empty() { "" } else { ", " };
    let endpoint = pass_through(&format!(
        "{{path: /x, target: https://t.example{separator}{flags}}}"
    ));
    assert_eq!(
        (
            endpoint.auth,
            endpoint.include_subpath,
            endpoint.forward_headers
        ),
        (auth, include_subpath, forward_headers)
    );
}

/// Expected values from pydantic 2.13.4 running `pass_through_auth_enforced` of BerriAI/litellm#51940:
/// `TypeAdapter(bool).validate_python(endpoint.get("auth", True))`, or `True` where it raises.
#[rstest]
#[case::text_false("'false'", false)]
#[case::no("'no'", false)]
#[case::off("'off'", false)]
#[case::text_zero("'0'", false)]
#[case::zero("0", false)]
#[case::padded_false("' false'", true)]
#[case::unknown_word("maybe", true)]
#[case::two("2", true)]
#[case::null("null", true)]
#[case::list("[]", true)]
fn pass_through_auth_stays_on_unless_pydantic_reads_false(
    #[case] auth: &str,
    #[case] expected: bool,
) {
    let endpoint = pass_through(&format!(
        "{{path: /x, target: https://t.example, auth: {auth}}}"
    ));
    assert_eq!(endpoint.auth, expected);
}

#[rstest]
#[case::bool_true("true", true)]
#[case::bool_false("false", false)]
#[case::string_true("'true'", false)]
#[case::environment_reference("os.environ/ENABLE_JWT_AUTH", false)]
#[case::number("1", false)]
fn auth_flags_are_on_only_for_boolean_true(#[case] value: &str, #[case] expected: bool) {
    let config = Config::from_yaml(&format!(
        "general_settings: {{enable_jwt_auth: {value}, enable_oauth2_auth: {value}, enable_oauth2_proxy_auth: {value}, allow_client_side_credentials: {value}}}"
    ))
    .unwrap();
    let general = config.general_settings;
    assert_eq!(
        [
            general.enable_jwt_auth,
            general.enable_oauth2_auth,
            general.enable_oauth2_proxy_auth,
            general.allow_client_side_credentials
        ],
        [expected; 4]
    );
}

#[rstest]
fn a_pass_through_endpoint_without_a_path_fails_the_load_like_registration() {
    assert!(
        Config::from_yaml(
            "general_settings: {pass_through_endpoints: [{target: https://t.example}]}"
        )
        .is_err()
    );
}

#[rstest]
fn pass_through_debug_hides_the_target_and_header_values() {
    let config = Config::from_yaml(
        "general_settings: {pass_through_endpoints: [{path: /x, target: https://user:target-secret@t.example, headers: {Authorization: header-secret}}]}",
    )
    .unwrap();
    let debug = format!("{config:?}");
    assert!(!debug.contains("target-secret") && !debug.contains("header-secret"));
}

#[rstest]
#[case::callbacks("callbacks")]
#[case::success_callback("success_callback")]
#[case::failure_callback("failure_callback")]
fn debug_hides_inline_callback_settings(#[case] field: &str) {
    let config = Config::from_yaml(&format!(
        "litellm_settings: {{{field}: [{{token: callback-list-secret}}]}}"
    ))
    .unwrap();
    assert!(!format!("{config:?}").contains("callback-list-secret"));
}

#[rstest]
#[case::list("[/a/*, /b]", Some(RouteList::Valid(["/a/*".to_owned(), "/b".to_owned()].into())))]
#[case::bare_string("/a/*", Some(RouteList::Malformed))]
#[case::non_string_entry("[/a, 5, /b/*]", Some(RouteList::Malformed))]
#[case::null("null", None)]
fn public_routes_never_fail_the_load(#[case] value: &str, #[case] expected: Option<RouteList>) {
    let config =
        Config::from_yaml(&format!("general_settings: {{public_routes: {value}}}")).unwrap();
    assert_eq!(config.general_settings.public_routes, expected);
}
