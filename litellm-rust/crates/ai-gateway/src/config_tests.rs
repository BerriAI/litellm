//! Tests for config schema parsing.

#[cfg(test)]
mod tests {
    use crate::config::{load_config_from_yaml, GeneralSettings, LiteLLMSettings, RouterSettings};
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_minimal_config() {
        let yaml = r#"
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY
"#;
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(yaml.as_bytes()).unwrap();
        
        let config = load_config_from_yaml(file.path().to_str().unwrap()).unwrap();
        
        assert_eq!(config.router.deployments().len(), 1);
        assert_eq!(config.router.deployments()[0].model_name, "gpt-4");
    }

    #[test]
    fn test_general_settings() {
        let yaml = r#"
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  database_url: os.environ/DATABASE_URL
  coordination_redis: os.environ/REDIS_URL
  max_parallel_requests: 100
  global_max_parallel_requests: 1000
  max_request_size_mb: 10
  max_response_size_mb: 100
  alerting:
    - slack
    - email
  alert_webhook_url: https://hooks.slack.com/services/xxx
  allowed_routes:
    - /v1/chat/completions
    - /v1/embeddings
"#;
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(yaml.as_bytes()).unwrap();
        
        let config = load_config_from_yaml(file.path().to_str().unwrap()).unwrap();
        
        assert!(config.general_settings.master_key.is_some());
        assert!(config.general_settings.database_url.is_some());
        assert!(config.general_settings.coordination_redis.is_some());
        assert_eq!(config.general_settings.max_parallel_requests, Some(100));
        assert_eq!(config.general_settings.global_max_parallel_requests, Some(1000));
        assert_eq!(config.general_settings.max_request_size_mb, Some(10));
        assert_eq!(config.general_settings.max_response_size_mb, Some(100));
        assert!(config.general_settings.alerting.is_some());
        assert_eq!(config.general_settings.alerting.as_ref().unwrap().len(), 2);
        assert!(config.general_settings.alert_webhook_url.is_some());
        assert!(config.general_settings.allowed_routes.is_some());
        assert_eq!(config.general_settings.allowed_routes.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_litellm_settings() {
        let yaml = r#"
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4

litellm_settings:
  callbacks:
    - type: langfuse
      public_key: pk_test
      secret_key: sk_test
      host: https://langfuse.example.com
    - type: datadog
      api_key: dd_api_key
      host: https://api.datadoghq.com
  guardrails:
    - guardrail_name: prompt_injection
      guardrail_type: lakera
      enabled: true
  cache: true
  cache_params:
    type: redis
    host: localhost
    port: 6379
    password: os.environ/REDIS_PASSWORD
    ttl: 300
  drop_params: true
  num_retries: 3
  timeout: 600
"#;
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(yaml.as_bytes()).unwrap();
        
        let config = load_config_from_yaml(file.path().to_str().unwrap()).unwrap();
        
        assert!(config.litellm_settings.callbacks.is_some());
        assert_eq!(config.litellm_settings.callbacks.as_ref().unwrap().len(), 2);
        assert!(config.litellm_settings.guardrails.is_some());
        assert_eq!(config.litellm_settings.guardrails.as_ref().unwrap().len(), 1);
        assert_eq!(config.litellm_settings.cache, Some(true));
        assert!(config.litellm_settings.cache_params.is_some());
        let cache_params = config.litellm_settings.cache_params.as_ref().unwrap();
        assert_eq!(cache_params.cache_type, Some("redis".to_string()));
        assert_eq!(cache_params.host, Some("localhost".to_string()));
        assert_eq!(cache_params.port, Some(6379));
        assert!(cache_params.password.is_some());
        assert_eq!(cache_params.ttl, Some(300));
        assert_eq!(config.litellm_settings.drop_params, Some(true));
        assert_eq!(config.litellm_settings.num_retries, Some(3));
        assert_eq!(config.litellm_settings.timeout, Some(600));
    }

    #[test]
    fn test_router_settings() {
        let yaml = r#"
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4

router_settings:
  routing_strategy: latency-based
  num_retries: 5
  timeout: 300
  cooldown_seconds: 60
  allowed_fails: 3
"#;
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(yaml.as_bytes()).unwrap();
        
        let config = load_config_from_yaml(file.path().to_str().unwrap()).unwrap();
        
        assert_eq!(config.router_settings.routing_strategy, Some("latency-based".to_string()));
        assert_eq!(config.router_settings.num_retries, Some(5));
        assert_eq!(config.router_settings.timeout, Some(300));
        assert_eq!(config.router_settings.cooldown_seconds, Some(60));
        assert_eq!(config.router_settings.allowed_fails, Some(3));
    }

    #[test]
    fn test_enhanced_model_list() {
        let yaml = r#"
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY
    model_info:
      input_cost_per_token: 0.00003
      output_cost_per_token: 0.00006
      mode: chat
    rpm: 1000
    tpm: 100000
    max_parallel_requests: 50
    mode: fallback
    healthy: true
    cooldown: 30
    weight: 10
"#;
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(yaml.as_bytes()).unwrap();
        
        let config = load_config_from_yaml(file.path().to_str().unwrap()).unwrap();
        
        assert_eq!(config.router.deployments().len(), 1);
        let deployment = &config.router.deployments()[0];
        assert_eq!(deployment.model_name, "gpt-4");
        // Note: The enhanced fields are parsed but not yet integrated into the Deployment struct
        // They will be used in future phases for routing decisions
    }

    #[test]
    fn test_pass_through_endpoints() {
        let yaml = r#"
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4

general_settings:
  pass_through_endpoints:
    - path: /custom/endpoint
      target: https://api.example.com/v1
      headers:
        X-Custom-Header: value
"#;
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(yaml.as_bytes()).unwrap();
        
        let config = load_config_from_yaml(file.path().to_str().unwrap()).unwrap();
        
        assert!(config.general_settings.pass_through_endpoints.is_some());
        let endpoints = config.general_settings.pass_through_endpoints.as_ref().unwrap();
        assert_eq!(endpoints.len(), 1);
        assert_eq!(endpoints[0].path, "/custom/endpoint");
        assert_eq!(endpoints[0].target, "https://api.example.com/v1");
        assert!(endpoints[0].headers.is_some());
    }

    #[test]
    fn test_env_var_resolution() {
        // SAFETY: This test is single-threaded and only modifies this specific env var
        unsafe {
            std::env::set_var("TEST_API_KEY", "test-key-123");
        }
        
        let yaml = r#"
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/TEST_API_KEY
"#;
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(yaml.as_bytes()).unwrap();
        
        let config = load_config_from_yaml(file.path().to_str().unwrap()).unwrap();
        
        assert_eq!(config.router.deployments().len(), 1);
        let deployment = &config.router.deployments()[0];
        assert_eq!(deployment.litellm_params.api_key, Some("test-key-123".to_string()));
        
        // SAFETY: This test is single-threaded and only modifies this specific env var
        unsafe {
            std::env::remove_var("TEST_API_KEY");
        }
    }

    #[test]
    fn test_empty_config() {
        let yaml = r#"
model_list: []
"#;
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(yaml.as_bytes()).unwrap();
        
        let config = load_config_from_yaml(file.path().to_str().unwrap()).unwrap();
        
        assert_eq!(config.router.deployments().len(), 0);
    }

    #[test]
    fn test_backward_compatibility() {
        // Test that old config format still works
        let yaml = r#"
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: sk-test
      api_base: https://api.openai.com/v1
"#;
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(yaml.as_bytes()).unwrap();
        
        let config = load_config_from_yaml(file.path().to_str().unwrap()).unwrap();
        
        assert_eq!(config.router.deployments().len(), 1);
        let deployment = &config.router.deployments()[0];
        assert_eq!(deployment.model_name, "gpt-4");
        assert_eq!(deployment.litellm_params.model, "openai/gpt-4");
        assert_eq!(deployment.litellm_params.api_key, Some("sk-test".to_string()));
        assert_eq!(deployment.litellm_params.api_base, Some("https://api.openai.com/v1".to_string()));
    }
}
