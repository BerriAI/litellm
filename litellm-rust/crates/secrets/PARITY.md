# Python secret-manager test parity

The inventory covers 132 tests in the secret-manager suites and the legacy secret-manager utility suites. It maps 108 tests to Rust coverage and identifies 24 tests owned by other layers or live environments. Rust tests exercise requests, returned values, caching, routing and failure behavior. Multiple Python tests can map to one parameterized Rust test

Tests of Python extension lifecycle, SDK credential selection in `auth-azure`, proxy hooks and example subclasses remain at their owning boundary. They are called out below rather than counted as Rust secret-manager coverage. Default AWS partition endpoints are owned by the AWS SDK

Azure callback absence remains `None`, while a native HTTP 404 still permits environment fallback. `azure_callback_absence_preserves_none_but_errors_fall_back` and `python_read_failures_preserve_provider_fallback_rules` cover these distinct results

Native APIs preserve typed errors and explicit absence. Python-compatible reads restore Python fallback, coercion, Google negative caching and missing-value behavior. Vault namespaces use the SDK namespace header instead of Python’s equivalent URL prefix. Rotation verifies fresh provider reads instead of trusting a just-written cache entry, preventing deletion after a failed replacement

Google payload corruption is intentionally rejected: malformed base64 and mismatched CRC32C values fail without populating the cache. `failed_or_missing_reads_are_not_cached` tests this correction against Python's permissive decoder and omitted checksum validation. See [RFC 4648 section 3.3](https://www.rfc-editor.org/rfc/rfc4648#section-3.3) and [Google's integrity guidance](https://docs.cloud.google.com/secret-manager/docs/data-integrity)

The Python API audit found that earlier AWS fallback tests encoded the wrong expectation. Differential calls to the existing handler show that missing secrets, denied reads, missing string payloads, and missing or empty primary secrets return `None`; they do not activate environment fallback or defaults. `test_aws_absence_and_failed_reads_match_python_without_environment_fallback` compares the public getter under both dispatch decisions, including HTTP 500 responses and their exact request counts. Python-compatible AWS reads disable SDK retries, matching Python's single attempt for service errors while retaining the native API's retry configuration. The bridge resolver test `aws_read_failure_preserves_absence_without_environment_fallback` checks the same rule when environment values exist. `test_aws_primary_values_match_python_handler` checks typed values, including arbitrary-size integers. `test_aws_primary_json_errors_preserve_python_exception_details` checks the exception class, arguments, document, and position against Python

Public AWS, Vault, CyberArk, and Google read methods now use catalog selection. AWS, Vault, and CyberArk keep their Python coroutine entrypoints, while Rust receives provider operation contexts. Full API replacement remains incomplete: AWS write/delete/rotation methods still need native bindings. Vault and CyberArk mutations now use native dispatch and share their native read caches. SDK-client configuration capture also needs to preserve explicit credentials, endpoints, and regions. Timeout phase handling, AWS optional-parameter side effects, per-call region selection without a base region, and environment lookup timing need further parity work. The private binding returns a Future, while the public methods retain ordinary Python coroutines as verified by lazy execution and `asyncio.create_task` tests. This follows the separation described in the PyO3 [signature](https://pyo3.rs/v0.29.2/function/signature.html) and [async](https://pyo3.rs/v0.29.2/async-await.html) guides. The catalog remains Python-only while these gaps are open

## Public API audit

The rollout decision comes from [catalog.py](../../../litellm/rust_bridge/catalog.py). All secret-manager rules remain `PYTHON_ONLY`; `LITELLM_RUST` does not make these incomplete routes production-ready. Differential tests pass explicit rules into the dispatch boundary

| Python entrypoint | Native bridge coverage | Remaining API work |
| --- | --- | --- |
| `litellm.get_secret`, `get_secret_str`, `get_secret_bool` | Existing Python entrypoints dispatch supported manager reads | SDK-client configuration, environment read timing and complete failure conversion |
| AWS `sync_read_secret`, `async_read_secret`, primary-secret helpers | Public signatures and coroutines retained; credentials, absence and typed JSON tested | Option consumption, timeout phases and region resolution |
| AWS `async_write_secret`, `async_delete_secret`, `async_rotate_secret`, `async_replicate_secret`, `async_put_secret_value` | Rust provider operations exist | Public native dispatch, original response fields and Python error contracts |
| Vault `sync_read_secret`, `async_read_secret` | Public signatures, nested overrides, namespace and data-key cache isolation tested | Complete timeout and initialization error parity |
| Vault `async_write_secret`, `async_delete_secret`, `async_rotate_secret` | Public native dispatch, complete response envelopes, HTTP error dictionaries, timeouts and fresh verification tested | Authentication and input-conversion edge cases, transport retries, timeout phases and mutable configuration read points |
| CyberArk reads, writes, deletes and rotations | Public native dispatch, shared cache, coroutine behavior, status errors and request counts tested | Other transport failures and client initialization timing |
| Google `get_secret_from_google_secret_manager` | Public native dispatch and distinct initial/cached missing results | Credential configuration and complete error parity |
| Azure Key Vault, AWS KMS, Google KMS SDK clients | Global secret-handler dispatch supports recognized clients | Explicit SDK credentials, endpoints, regions and caller-supplied credentials |
| Custom managers and subclasses | Preserve Python callbacks | Caller implementations must never be replaced by built-in native managers |

`_SecretManagerRuntime` is a private implementation detail, not a replacement SDK class. Its async methods return Futures; public `async def` methods retain lazy coroutine creation and `asyncio.create_task` support. Passing the same names and arguments is insufficient to claim parity until the remaining return-value, error, cache and configuration differences above are closed

## [tests/unit/secret_managers/test_aws_secret_manager_replication.py](../../../tests/unit/secret_managers/test_aws_secret_manager_replication.py)

| Python test | Rust coverage or boundary |
| --- | --- |
| `test_write_secret_replicates_when_configured` | [creation_replicates_only_to_configured_regions](../secrets-aws/tests/secret_manager/writes.rs) |
| `test_write_secret_no_replication_when_not_configured` | [creation_replicates_only_to_configured_regions](../secrets-aws/tests/secret_manager/writes.rs) |
| `test_replication_failure_does_not_fail_write` | [creation_passes_tags_and_kms_and_survives_replication_failure](../secrets-aws/tests/secret_manager/writes.rs) |
| `test_async_replicate_secret_empty_regions_returns_empty` | [creation_passes_tags_and_kms_and_survives_replication_failure](../secrets-aws/tests/secret_manager/writes.rs) |
| `test_async_replicate_secret_correct_payload` | [direct_replication_returns_response_or_service_error](../secrets-aws/tests/secret_manager/writes.rs) |
| `test_replication_fires_on_create` | [creation_replicates_only_to_configured_regions](../secrets-aws/tests/secret_manager/writes.rs) |
| `test_load_aws_secret_manager_passes_replica_regions` | [creation_replicates_only_to_configured_regions](../secrets-aws/tests/secret_manager/writes.rs) |
| `test_write_secret_http_error_raises` | [create_failure_does_not_overwrite_an_alias_without_a_deletion_date](../secrets-aws/tests/secret_manager/writes.rs) |
| `test_write_secret_timeout_raises` | [write_and_replication_timeouts_remain_errors](../secrets-aws/tests/secret_manager/writes.rs) |
| `test_replicate_secret_http_error_raises` | [direct_replication_returns_response_or_service_error](../secrets-aws/tests/secret_manager/writes.rs) |
| `test_replicate_secret_timeout_raises` | [write_and_replication_timeouts_remain_errors](../secrets-aws/tests/secret_manager/writes.rs) |

## [tests/unit/secret_managers/test_aws_secret_manager_rotation.py](../../../tests/unit/secret_managers/test_aws_secret_manager_rotation.py)

| Python test | Rust coverage or boundary |
| --- | --- |
| `test_rotate_secret_same_name_writes_requested_value_in_place` | [same_name_rotation_uses_put_and_returns_its_response](../secrets-aws/tests/secret_manager/writes.rs) |
| `test_rotate_secret_different_names_persists_requested_value_and_deletes_old_alias` | [renamed_rotation_reads_creates_verifies_then_deletes](../secrets-aws/tests/secret_manager/writes.rs) |
| `test_rotate_secret_back_to_name_inside_recovery_window_restores_and_stores_new_value` | [recovery_window_alias_is_restored_updated_and_tagged](../secrets-aws/tests/secret_manager/writes.rs) |
| `test_write_secret_to_name_inside_recovery_window_reschedules_deletion_when_update_fails` | [failed_update_reschedules_deletion_of_a_restored_alias](../secrets-aws/tests/secret_manager/writes.rs) |
| `test_write_secret_to_name_inside_recovery_window_restores_and_stores_new_value` | [recovery_window_alias_is_restored_updated_and_tagged](../secrets-aws/tests/secret_manager/writes.rs) |
| `test_write_secret_to_live_existing_name_still_fails_without_overwriting` | [create_failure_does_not_overwrite_an_alias_without_a_deletion_date](../secrets-aws/tests/secret_manager/writes.rs) |

## [tests/unit/secret_managers/test_aws_secret_manager_v2.py](../../../tests/unit/secret_managers/test_aws_secret_manager_v2.py)

| Python test | Rust coverage or boundary |
| --- | --- |
| `test_create_secret_uses_customer_managed_kms_key_from_settings` | [creation_replicates_only_to_configured_regions](../secrets-aws/tests/secret_manager/writes.rs) |
| `test_create_secret_omits_kms_key_id_when_not_configured` | [write_read_delete_preserves_the_complete_secret_string](../secrets-aws/tests/secret_manager/writes.rs) |
| `test_write_and_read_json_secret` | [write_read_delete_preserves_the_complete_secret_string](../secrets-aws/tests/secret_manager/writes.rs) |
| `test_prepare_request_builds_partition_endpoint` | AWS SDK owns partition endpoint construction. LiteLLM region selection is exercised by `trait_read_uses_the_aws_region_from_its_operation_context`; no vendor endpoint table is duplicated |
| `test_prepare_request_explicit_bedrock_runtime_endpoint_param_still_wins` | [endpoint_overrides_replace_the_service_and_override_the_region](../secrets-aws/tests/secret_manager/configuration.rs) |
| `test_prepare_request_env_bedrock_runtime_endpoint_still_wins` | [endpoint_overrides_replace_the_service_and_override_the_region](../secrets-aws/tests/secret_manager/configuration.rs) |

## [tests/unit/secret_managers/test_base_secret_manager.py](../../../tests/unit/secret_managers/test_base_secret_manager.py)

| Python test | Rust coverage or boundary |
| --- | --- |
| `test_raise_if_unsafe_secret_name_rejects_traversal_and_line_breaks` | [names_reject_path_traversal_and_control_characters](../secrets-types/tests/rotation.rs) |
| `test_raise_if_unsafe_secret_name_allows_legitimate_aliases` | [names_allow_safe_values](../secrets-types/tests/rotation.rs) |

## [tests/unit/secret_managers/test_custom_secret_manager.py](../../../tests/unit/secret_managers/test_custom_secret_manager.py)

| Python test | Rust coverage or boundary |
| --- | --- |
| `test_custom_secret_manager_initialization` | Exercises the Python example subclass itself or Python default methods. Caller-authored Python implementations remain Python callbacks; resolver integration is covered by `manager_strings_are_coerced_like_literal_eval` |
| `test_custom_secret_manager_sync_read` | Exercises the Python example subclass itself or Python default methods. Caller-authored Python implementations remain Python callbacks; resolver integration is covered by `manager_strings_are_coerced_like_literal_eval` |
| `test_custom_secret_manager_async_read` | Exercises the Python example subclass itself or Python default methods. Caller-authored Python implementations remain Python callbacks; resolver integration is covered by `manager_strings_are_coerced_like_literal_eval` |
| `test_custom_secret_manager_async_write` | Exercises the Python example subclass itself or Python default methods. Caller-authored Python implementations remain Python callbacks; resolver integration is covered by `manager_strings_are_coerced_like_literal_eval` |
| `test_custom_secret_manager_async_delete` | Exercises the Python example subclass itself or Python default methods. Caller-authored Python implementations remain Python callbacks; resolver integration is covered by `manager_strings_are_coerced_like_literal_eval` |
| `test_custom_secret_manager_integration_with_litellm` | [manager_strings_are_coerced_like_literal_eval](../secrets/tests/resolution.rs) |
| `test_minimal_custom_secret_manager` | Exercises the Python example subclass itself or Python default methods. Caller-authored Python implementations remain Python callbacks; resolver integration is covered by `manager_strings_are_coerced_like_literal_eval` |

## [tests/unit/secret_managers/test_cyberark_secret_manager.py](../../../tests/unit/secret_managers/test_cyberark_secret_manager.py)

| Python test | Rust coverage or boundary |
| --- | --- |
| `test_sync_read_matches_parity_fixture` | [secret_names_use_python_quote_encoding](../secrets-cyberark/tests/secret_manager/reads.rs) |
| `test_async_write_matches_parity_fixture` | [writes_match_python_parity_fixture](../secrets-cyberark/tests/secret_manager/writes.rs) |
| `test_missing_credentials_raise_value_error` | [new_validates_credentials_before_license_and_configuration](../secrets-cyberark/tests/secret_manager/configuration.rs) |

## [tests/unit/secret_managers/test_get_azure_ad_token_provider.py](../../../tests/unit/secret_managers/test_get_azure_ad_token_provider.py)

| Python test | Rust coverage or boundary |
| --- | --- |
| `test_deployment_identity_reaches_workload_and_managed_identity_only` | Credential-selection contract belongs to `litellm-auth-azure`, not a secrets crate |
| `test_deployment_identity_survives_a_developer_only_token_credentials_setting` | Credential-selection contract belongs to `litellm-auth-azure`, not a secrets crate |
| `test_default_azure_credential_keeps_its_full_chain` | Credential-selection contract belongs to `litellm-auth-azure`, not a secrets crate |
| `test_deployment_identity_refuses_to_mint_a_token_for_a_configured_service_principal` | Credential-selection contract belongs to `litellm-auth-azure`, not a secrets crate |
| `test_deployment_identity_still_reaches_a_system_assigned_managed_identity` | Credential-selection contract belongs to `litellm-auth-azure`, not a secrets crate |
| `test_deployment_identity_keeps_the_user_assigned_identity_under_a_dev_only_setting` | Credential-selection contract belongs to `litellm-auth-azure`, not a secrets crate |
| `test_get_azure_ad_token_provider_client_secret_credential` | Credential-selection contract belongs to `litellm-auth-azure`, not a secrets crate |
| `test_get_azure_ad_token_provider_managed_identity_credential` | Credential-selection contract belongs to `litellm-auth-azure`, not a secrets crate |
| `test_get_azure_ad_token_provider_certificate_credential` | Credential-selection contract belongs to `litellm-auth-azure`, not a secrets crate |
| `test_get_azure_ad_token_provider_password_protected_certificate_credential` | Credential-selection contract belongs to `litellm-auth-azure`, not a secrets crate |
| `test_get_azure_ad_token_provider_default_azure_credential` | Credential-selection contract belongs to `litellm-auth-azure`, not a secrets crate |
| `test_get_azure_ad_token_provider_prefers_workload_identity_over_managed_identity` | Credential-selection contract belongs to `litellm-auth-azure`, not a secrets crate |
| `test_get_azure_ad_token_provider_defaults_to_default_azure_credential` | Credential-selection contract belongs to `litellm-auth-azure`, not a secrets crate |

## [tests/unit/secret_managers/test_hashicorp_secret_manager.py](../../../tests/unit/secret_managers/test_hashicorp_secret_manager.py)

| Python test | Rust coverage or boundary |
| --- | --- |
| `test_sync_read_uses_login_namespace_for_approle_and_secret_namespace_for_url` | [login_and_secret_namespaces_follow_python_precedence](../secrets-hashicorp/tests/secret_manager/configuration.rs) |
| `test_login_header_is_omitted_when_no_namespace_is_configured` | [login_and_secret_namespaces_follow_python_precedence](../secrets-hashicorp/tests/secret_manager/configuration.rs) |
| `test_sync_read_per_secret_namespace_overrides_secret_namespace` | [operation_overrides_isolate_cached_targets_and_apply_to_writes_and_deletes](../secrets-hashicorp/tests/secret_manager/writes.rs) |
| `test_sync_read_caches_per_resolved_target` | [operation_overrides_isolate_cached_targets_and_apply_to_writes_and_deletes](../secrets-hashicorp/tests/secret_manager/writes.rs) |
| `test_sync_read_caches_per_data_key_for_the_same_secret_path` | [reads_cache_each_data_key_for_the_same_vault_path](../secrets-hashicorp/tests/secret_manager/reads.rs) |
| `test_async_delete_evicts_every_cached_field_of_the_secret_path` | [write_and_delete_invalidate_the_read_cache](../secrets-hashicorp/tests/secret_manager/writes.rs) |
| `test_async_read_uses_secret_namespace_and_login_namespace` | [login_and_secret_namespaces_follow_python_precedence](../secrets-hashicorp/tests/secret_manager/configuration.rs) |
| `test_async_write_and_read_share_the_secret_namespace_target` | [write_and_delete_invalidate_the_read_cache](../secrets-hashicorp/tests/secret_manager/writes.rs) |
| `test_tls_login_uses_login_namespace` | [tls_login_posts_the_role_and_uses_the_client_identity](../secrets-hashicorp/tests/secret_manager/configuration.rs) |
| `test_configuration_matches_native_parity_fixture` | [configuration_matches_python_parity_fixture](../secrets-hashicorp/tests/secret_manager/configuration.rs) |

## [tests/unit/secret_managers/test_secret_manager_handler.py](../../../tests/unit/secret_managers/test_secret_manager_handler.py)

| Python test | Rust coverage or boundary |
| --- | --- |
| `test_azure_key_vault_matches_rust_parity_fixture` | [parity_fixture_matches_python_backend_contract](../secrets-azure/tests/key_vault.rs) |

## [tests/unit/secret_managers/test_secret_managers_main.py](../../../tests/unit/secret_managers/test_secret_managers_main.py)

| Python test | Rust coverage or boundary |
| --- | --- |
| `test_oidc_google_success` | [google_expiry_caps_cache_and_preserves_audience](../secrets/tests/oidc.rs) |
| `test_oidc_google_cached` | [google_expiry_caps_cache_and_preserves_audience](../secrets/tests/oidc.rs) |
| `test_oidc_google_cache_ttl_capped_by_token_exp` | [google_tokens_expire_at_the_python_cache_deadline](../secrets/tests/oidc.rs) |
| `test_oidc_google_expired_token_not_cached` | [google_expiry_caps_cache_and_preserves_audience](../secrets/tests/oidc.rs) |
| `test_oidc_google_long_lived_token_still_capped_at_default_ttl` | [google_tokens_expire_at_the_python_cache_deadline](../secrets/tests/oidc.rs) |
| `test_oidc_google_non_jwt_token_keeps_default_ttl` | [google_tokens_expire_at_the_python_cache_deadline](../secrets/tests/oidc.rs) |
| `test_oidc_google_failure` | [google_oidc_failures_are_not_cached_or_hidden_by_defaults](../secrets/tests/oidc.rs) |
| `test_oidc_circleci_success` | [environment_sources_resolve_expected_value](../secrets/tests/oidc.rs) |
| `test_oidc_circleci_failure` | [missing_oidc_environment_is_an_error](../secrets/tests/oidc.rs) |
| `test_oidc_github_success` | [github_requests_are_authenticated_cached_and_revalidate_environment](../secrets/tests/oidc.rs) |
| `test_oidc_github_missing_env` | [github_requests_are_authenticated_cached_and_revalidate_environment](../secrets/tests/oidc.rs) |
| `test_oidc_azure_file_success` | [file_allowlist_resolves_symlinks_while_environment_paths_remain_explicit](../secrets/tests/oidc.rs) |
| `test_oidc_azure_ad_token_success` | [azure_oidc_acquires_the_requested_scope_and_preserves_failures](../secrets/tests/oidc.rs) |
| `test_oidc_file_success` | [file_allowlist_resolves_symlinks_while_environment_paths_remain_explicit](../secrets/tests/oidc.rs) |
| `test_oidc_file_rejects_path_outside_allowlist` | [file_allowlist_resolves_symlinks_while_environment_paths_remain_explicit](../secrets/tests/oidc.rs) |
| `test_oidc_file_rejects_relative_path` | [file_allowlist_resolves_symlinks_while_environment_paths_remain_explicit](../secrets/tests/oidc.rs) |
| `test_oidc_env_success` | [environment_sources_resolve_expected_value](../secrets/tests/oidc.rs) |
| `test_oidc_env_path_success` | [file_allowlist_resolves_symlinks_while_environment_paths_remain_explicit](../secrets/tests/oidc.rs) |
| `test_unsupported_oidc_provider` | [invalid_references_fail_before_environment_lookup](../secrets/tests/oidc.rs) |
| `test_normalize_nonempty_secret_str` | [normalization_matches_python_without_changing_embedded_whitespace](../secrets/tests/resolution.rs) |
| `test_secret_manager_would_be_consulted_matches_get_secret` | [gating_prediction_matches_actual_lookup](../secrets/tests/aws.rs) |
| `test_secret_manager_would_be_consulted_is_false_without_a_client` | [prefix_is_removed_once_and_resolved_from_environment](../secrets/tests/resolution.rs) |

## [tests/litellm_utils_tests/test_secret_manager.py](../../../tests/litellm_utils_tests/test_secret_manager.py)

| Python test | Rust coverage or boundary |
| --- | --- |
| `test_aws_secret_manager` | [write_read_delete_preserves_the_complete_secret_string](../secrets-aws/tests/secret_manager/writes.rs) |
| `test_oidc_google` | [google_expiry_caps_cache_and_preserves_audience](../secrets/tests/oidc.rs) |
| `test_oidc_github` | [github_requests_are_authenticated_cached_and_revalidate_environment](../secrets/tests/oidc.rs) |
| `test_oidc_circleci` | [environment_sources_resolve_expected_value](../secrets/tests/oidc.rs) |
| `test_oidc_circleci_v2` | [environment_sources_resolve_expected_value](../secrets/tests/oidc.rs) |
| `test_oidc_circleci_with_azure` | Quarantined live Azure token exchange, outside secrets crates. CircleCI token retrieval is covered by `environment_sources_resolve_expected_value` |
| `test_oidc_circle_v1_with_amazon` | Quarantined live AWS token exchange, outside secrets crates. Token retrieval and STS forwarding are covered independently |
| `test_oidc_env_variable` | [environment_sources_resolve_expected_value](../secrets/tests/oidc.rs) |
| `test_oidc_file` | [file_allowlist_resolves_symlinks_while_environment_paths_remain_explicit](../secrets/tests/oidc.rs) |
| `test_oidc_env_path` | [file_allowlist_resolves_symlinks_while_environment_paths_remain_explicit](../secrets/tests/oidc.rs) |
| `test_google_secret_manager` | [successful_reads_use_auth_latest_version_and_cache_including_empty_values](../secrets-google/tests/secret_manager.rs) |
| `test_google_secret_manager_read_in_memory` | [python_reads_reuse_cached_absence_until_expiry](../secrets-google/tests/secret_manager.rs) |
| `test_should_read_secret_from_secret_manager` | [gating_prediction_matches_actual_lookup](../secrets/tests/aws.rs) |
| `test_get_secret_with_access_mode` | [gating_prediction_matches_actual_lookup](../secrets/tests/aws.rs) |
| `test_key_management_settings_defaults` | [config_preserves_defaults_nulls_and_serialized_names](../secrets-types/tests/config.rs) |
| `test_key_management_settings_custom_values` | [config_preserves_defaults_nulls_and_serialized_names](../secrets-types/tests/config.rs) |
| `test_async_write_secret_receives_description_and_tags` | Proxy hook behavior stays in Python. Native write metadata is covered by `trait_write_uses_typed_write_context` |
| `test_key_management_settings_serialization_roundtrip` | [config_preserves_defaults_nulls_and_serialized_names](../secrets-types/tests/config.rs) |

## [tests/litellm_utils_tests/test_get_secret.py](../../../tests/litellm_utils_tests/test_get_secret.py)

| Python test | Rust coverage or boundary |
| --- | --- |
| `test_azure_kms` | [azure_handler_reads_missing_and_failed_secrets](../secrets/tests/azure.rs) |

## [tests/litellm_utils_tests/test_aws_secret_manager.py](../../../tests/litellm_utils_tests/test_aws_secret_manager.py)

| Python test | Rust coverage or boundary |
| --- | --- |
| `test_write_and_read_simple_secret` | [write_read_delete_preserves_the_complete_secret_string](../secrets-aws/tests/secret_manager/writes.rs) |
| `test_write_and_read_json_secret` | [write_read_delete_preserves_the_complete_secret_string](../secrets-aws/tests/secret_manager/writes.rs) |
| `test_read_nonexistent_secret` | [failed_read_returns_none_but_invalid_primary_json_is_an_error](../secrets-aws/tests/secret_manager/reads.rs) |
| `test_primary_secret_functionality` | [primary_lookup_preserves_read_semantics](../secrets-aws/tests/secret_manager/reads.rs) |
| `test_write_secret_with_description_and_tags` | [creation_passes_tags_and_kms_and_survives_replication_failure](../secrets-aws/tests/secret_manager/writes.rs) |
| `test_secret_manager_with_iam_role_settings` | [configured_sts_credentials_sign_the_secret_request](../secrets-aws/tests/secret_manager/configuration.rs) |
| `test_secret_manager_with_cross_account_settings` | [configured_sts_credentials_sign_the_secret_request](../secrets-aws/tests/secret_manager/configuration.rs) |
| `test_secret_manager_with_irsa_settings` | [configured_sts_credentials_sign_the_secret_request](../secrets-aws/tests/secret_manager/configuration.rs) |
| `test_secret_manager_with_custom_sts_endpoint` | [configured_sts_credentials_sign_the_secret_request](../secrets-aws/tests/secret_manager/configuration.rs) |
| `test_secret_manager_with_aws_profile` | [configured_profile_credentials_override_static_environment_credentials](../secrets-aws/tests/secret_manager/configuration.rs) |
| `test_load_aws_secret_manager_with_settings` | [creation_replicates_only_to_configured_regions](../secrets-aws/tests/secret_manager/writes.rs) |
| `test_end_to_end_iam_role_secret_write` | Live AWS account test, not a unit test. Offline STS signing and secret writes are covered without account assumptions |

## [tests/litellm_utils_tests/test_hashicorp.py](../../../tests/litellm_utils_tests/test_hashicorp.py)

| Python test | Rust coverage or boundary |
| --- | --- |
| `test_hashicorp_secret_manager_get_secret` | [token_reads_use_vault_headers_and_cache_values](../secrets-hashicorp/tests/secret_manager/reads.rs) |
| `test_hashicorp_secret_manager_write_secret` | [write_and_delete_invalidate_the_read_cache](../secrets-hashicorp/tests/secret_manager/writes.rs) |
| `test_hashicorp_secret_manager_write_secret_with_team_overrides` | [operation_overrides_isolate_cached_targets_and_apply_to_writes_and_deletes](../secrets-hashicorp/tests/secret_manager/writes.rs) |
| `test_hashicorp_secret_manager_delete_secret` | [write_and_delete_invalidate_the_read_cache](../secrets-hashicorp/tests/secret_manager/writes.rs) |
| `test_hashicorp_secret_manager_delete_secret_with_team_overrides` | [operation_overrides_isolate_cached_targets_and_apply_to_writes_and_deletes](../secrets-hashicorp/tests/secret_manager/writes.rs) |
| `test_hashicorp_secret_manager_tls_cert_auth` | [tls_login_posts_the_role_and_uses_the_client_identity](../secrets-hashicorp/tests/secret_manager/configuration.rs) |
| `test_hashicorp_secret_manager_approle_auth` | [approle_login_uses_namespace_and_reuses_the_token](../secrets-hashicorp/tests/secret_manager/configuration.rs) |
| `test_hashicorp_custom_mount_and_prefix` | [namespace_mount_and_prefix_are_sanitized_in_the_url](../secrets-hashicorp/tests/secret_manager/reads.rs) |
| `test_hashicorp_get_url_rejects_path_traversal` | [no_auth_and_invalid_names_fail_without_requests](../secrets-hashicorp/tests/secret_manager/configuration.rs) |
| `test_hashicorp_secret_manager_rotate_secret_different_names` | [rotation_applies_timeout_to_each_request](../secrets-hashicorp/tests/secret_manager/writes.rs) |
| `test_hashicorp_secret_manager_rotate_secret_same_name` | [same_name_rotation_keeps_the_replacement](../secrets-hashicorp/tests/secret_manager/writes.rs) |
| `test_hashicorp_secret_manager_rotate_secret_current_not_found` | [missing_old_or_new_value_stops_rotation_before_deletion](../secrets-types/tests/rotation.rs) |
| `test_hashicorp_secret_manager_rotate_secret_write_fails` | [provider_failures_stop_rotation_before_retirement](../secrets-types/tests/rotation.rs) |
| `test_hashicorp_secret_manager_rotate_secret_with_team_overrides` | [rotation_applies_timeout_to_each_request](../secrets-hashicorp/tests/secret_manager/writes.rs) |
| `test_hashicorp_secret_manager_rotate_secret_value_mismatch` | [a_different_replacement_never_deletes_the_current_secret](../secrets-types/tests/rotation.rs) |

## [tests/litellm_utils_tests/test_cyberark.py](../../../tests/litellm_utils_tests/test_cyberark.py)

| Python test | Rust coverage or boundary |
| --- | --- |
| `test_cyberark_write_secret_rejects_yaml_injection` | [unsafe_names_fail_before_http_calls](../secrets-cyberark/tests/secret_manager/reads.rs) |
| `test_cyberark_ensure_variable_exists_escapes_yaml_metacharacters` | [policy_writes_preserve_yaml_metacharacters_as_one_variable](../secrets-cyberark/tests/secret_manager/writes.rs) |
| `test_cyberark_write_and_read_secret` | [writes_tolerate_policy_status_and_cache_value](../secrets-cyberark/tests/secret_manager/writes.rs) |
| `test_cyberark_rotate_secret` | [rotation_stores_the_replacement_and_retains_other_aliases](../secrets-cyberark/tests/secret_manager/writes.rs) |
| `test_cyberark_rotate_secret_with_new_alias` | [rotation_stores_the_replacement_and_retains_other_aliases](../secrets-cyberark/tests/secret_manager/writes.rs) |

## Public provider read boundary

`test_public_aws_reads_preserve_coroutines_and_per_call_credentials` verifies lazy coroutine execution, `asyncio.create_task`, positional and keyword calls, request payloads, and per-call credentials, region, and endpoint overrides. `test_public_aws_primary_reads_ignore_operation_overrides_like_python` retains Python's ignored primary-read overrides. Bootstrap keys bypass only synchronous reads, including native backend initialization

Real HTTP timeouts are swallowed by AWS reads because LiteLLM's standard HTTP handler raises `litellm.Timeout`; tests that inject `httpx.TimeoutException` bypass that wrapping. `test_public_aws_read_timeouts_follow_the_python_http_handler` compares both implementations against delayed responses

Vault reads retain nested overrides, Python string conversion, and cache isolation. CyberArk reuses authentication and preserves raw secret text in its cache. Python's shared cache JSON-decodes CyberArk values on subsequent reads, corrupting quoted strings and changing types. The native behavior intentionally fixes this corruption, with the Python difference shown in `test_public_cyberark_reads_reuse_authentication_and_cached_values`

Google's first missing-secret read raises, while a cached miss returns `None`. `python_reads_reuse_cached_absence_until_expiry` and `python_cached_absence_expires_and_allows_recovery` retain both outcomes


## Public CyberArk mutation boundary

Public writes, deletes, and rotations retain the Python method signatures and coroutine entrypoints. Write and delete results preserve Python's status/message dictionaries. Unsupported deletion clears the shared native read cache without a provider request. Authentication and write HTTP failures preserve Python's messages and request counts, including the ignored initial authentication failure while ensuring a policy. Python-compatible reads and writes do not retry HTTP 401; native Rust retry policies remain unchanged

The bridge compares connection-refused errors against Python and builds HTTP status messages through HTTPX. Other transport failures and client-initialization timing still need a complete API audit; these checks do not establish full error parity

`test_public_cyberark_writes_and_deletes_share_the_read_cache` verifies real HTTP writes followed by sync and async cached reads and deletion invalidation. `test_public_cyberark_write_errors_match_python_without_http_retries`, `test_public_cyberark_connection_errors_match_python`, and `test_cyberark_handler_errors_match_python_after_cached_authentication_is_denied` compare error results against Python. Missing-extension selection remains covered for each mutation

Rotation preserves the documented fresh-read safeguard. Python's base rotation checks a cache populated by the write and does not compare the stored value, so a successful response can conceal a missing or incorrect replacement. `test_public_cyberark_rotation_requires_a_fresh_matching_replacement` rejects both cases and verifies the old cache remains intact. `test_public_cyberark_rotation_stops_after_a_failed_write` preserves the old value and returns the write error without further requests. Conjur retains the old provider alias because its deletion API is unsupported


## Public Vault mutation boundary

Public Vault writes, deletes and rotation now use catalog selection. Their Python signatures and coroutine entrypoints stay unchanged. Native operations use the Vault SDK request types and authenticated client while retaining the complete response bytes for Python JSON conversion. This preserves additional response fields, key order and arbitrary-size integers. Python-compatible writes make one provider attempt; the existing native API keeps its CAS recovery behavior

`test_public_vault_writes_preserve_complete_responses_and_request_fields` checks the response envelope, nested operation settings, payload and ignored tags. `test_public_vault_mutation_http_errors_match_python_without_retry` compares write/delete error dictionaries, including namespace URLs and exact request counts. Rotation tests compare current-secret failures, ordered request paths, write failures, failed verification, mismatched values, malformed verification shapes, same-name updates and best-effort old-alias deletion. A successful HTTP response containing `status: error` stops rotation and returns the original response. Verification fields remain raw JSON until Python error conversion, preserving large integers, nested values and the distinction between integer and floating-point type errors. Unsupported extension selection is checked separately for write, delete and rotate

`test_public_vault_mutation_timeouts_match_python` compares operation-specific timeout messages. The elapsed duration in a POST error is measured independently, so the test checks its structure and lower bound rather than requiring two independent requests to have identical elapsed time. Phase-specific connect/read/write/pool deadlines and cached Python transport configuration still need broader parity checks

Native writes retain two documented correctness safeguards. Python's `async_write_secret` does not invalidate the read cache, so a later read can return a value from before a successful write. `test_public_native_vault_write_invalidates_stale_cached_values` verifies that the native public API returns the updated provider value. Python also overwrites the secret when both the data key and description field are named `description`. `test_public_native_vault_write_rejects_description_overwriting_the_secret` rejects that collision before any request. These corrections do not change Python

The raw response path does not yet establish complete Vault API parity. SDK authentication payload parsing, argument conversion, connection retry behavior, non-HTTP transport errors, nonstandard JSON encodings during rotation and configuration changes during rotation remain under audit. The catalog stays Python-only


The Vault boundary update passed 317 provider and bridge tests, including 191 bridge cases. With the extension unavailable, 35 passed and 156 native-only cases skipped. Seven targeted mutations compiled and failed their regression tests: stripped response envelopes, ignored HTTP 400 failures, skipped replacement equality, skipped current-secret checks, stale write caches, fatal old-secret deletion failures and ignored write-error responses. The restored extension passed again. Five additional differential cases reproduced lossy large-integer error conversion before the raw-JSON correction and pass afterward. Live public native reads, writes, deletes and same-name/new-alias rotations passed against local Vault 1.20 with token and AppRole authentication, with Python HTTP construction forbidden
