mod auth;
mod billable_route;
mod budget_counters;
mod budgets;
mod cache;
mod callable;
mod callbacks_model_capacity;
mod coercion;
mod cost;
mod credentials;
mod db;
mod db_repository;
mod diagnostics;
mod errors;
mod execution;
mod http;
mod logger;
mod marshal;
mod model_cost;
mod model_registry;
mod parallel_request_limiter;
mod preflight;
mod python_settings;
mod redaction;
mod redis;
mod router;
mod routes;
mod s3_log;
mod secrets;
mod spend_log;
mod spend_log_rows;
mod spend_log_writer;
mod tokenizer;

#[cfg(feature = "fips")]
fn crypto_provider() -> rustls::crypto::CryptoProvider {
    rustls::crypto::default_fips_provider()
}

#[cfg(not(feature = "fips"))]
fn crypto_provider() -> rustls::crypto::CryptoProvider {
    rustls::crypto::aws_lc_rs::default_provider()
}

#[pymodule(gil_used = true)]
mod _native {
    #[pymodule_export]
    use litellm_host_python::{ForkedAfterNativeRuntimeStarted, ProcessReservedForForking};
    use pyo3::{prelude::*, types::PyModule};

    #[pymodule_export]
    use crate::auth::{
        AuthCall, AuthOutcome, AuthSettings, AuthStep, ClaimIdentityView, DenialView, IdentityView,
        KeyContextView, MappedClaimsView, MappingView, PlanView, TeamGrantView, jwt_claims,
        jwt_is_allowed_domain, jwt_is_jwt, jwt_presents_claims, jwt_unverified_claims,
    };
    #[pymodule_export]
    use crate::auth::{
        auth_allowed_routes_grant, auth_first_denied_route, auth_jwt_team_routes_grant,
        auth_managed_agent_route_allowed, auth_managed_route, auth_route_matches_denied,
        auth_route_matches_prefix, auth_route_matches_wildcard,
    };
    #[pymodule_export]
    use crate::billable_route::{billable_route_passthrough_prefixes, classify_billable_route};
    #[pymodule_export]
    use crate::budgets::NativeBudgets;
    #[pymodule_export]
    use crate::budgets::budget_check_decision;
    #[pymodule_export]
    use crate::callbacks_model_capacity::{NativeModelCapacity, NativeModelCapacityState};
    #[pymodule_export]
    use crate::cost::billable_service_tier;
    #[pymodule_export]
    use crate::cost::{
        NativeCallContext, NativePriced, NativePrices, NativeQuote, NativeRateCard,
        NativeTokenUsage,
    };
    #[pymodule_export]
    use crate::db::{DatabaseEndpoint, NativeDatabase, NativeDatabaseFailure};
    #[pymodule_export]
    use crate::db_repository::{
        RustAccessGroupRepository, RustAgentIdentityRepository, RustAuditLogRepository,
        RustBudgetRepository, RustEndUserRepository, RustJwtKeyMappingRepository,
        RustKeyOwnersRepository, RustKeyRepository, RustModelAccessGroupRepository,
        RustObjectPermissionRepository, RustOrganizationRepository, RustProjectRepository,
        RustSpendLogRepository, RustTagRepository, RustTeamMembershipRepository,
        RustTeamRepository, RustUserRepository,
    };
    #[cfg(feature = "panic-test")]
    #[pymodule_export]
    use crate::diagnostics::_panic_for_test;
    #[pymodule_export]
    use crate::diagnostics::{gil_stats, process_state_started, reserve_process_for_forking};
    #[pymodule_export]
    use crate::errors::{RustBridgeDeclined, RustUpstreamError};
    #[pymodule_export]
    use crate::http::unsupported_tls_settings;
    #[pymodule_export]
    use crate::model_cost::{
        CostMapRefreshStop, CostMapRefreshed, NativeCostMapSource, NativeFetchFailed,
        NativeFetchPolicy, NativeFetched, NativeInstalled, NativeModelCostMap,
        NativeModelCostMapState, NativeProvenance, NativeRegistered, NativeRejected,
        NativeTableSwapped,
    };
    #[pymodule_export]
    use crate::model_registry::bundled_model_cost_map;
    #[pymodule_export]
    use crate::parallel_request_limiter::{
        NativeParallelRequestLimiter, NativeParallelRequestLimiterState,
    };
    #[pymodule_export]
    use crate::redaction::{NativeDiagnosticProcessor, NativeTokenRedactor};
    #[pymodule_export]
    use crate::redis::breaker::{NativeAdmission, NativeRedisBreaker, native_redis_breaker_states};
    #[pymodule_export]
    use crate::redis::client::NativeRedisClient;
    #[pymodule_export]
    use crate::redis::store::active_teams::{NativeActiveTeams, PyActiveMember};
    #[pymodule_export]
    use crate::redis::store::auth_cache::{
        NativeAuthCache, NativeTeamRoster, PyAuthObject, PyLocalSizes,
    };
    #[pymodule_export]
    use crate::redis::store::cache_invalidation::NativeCacheInvalidation;
    #[pymodule_export]
    use crate::redis::store::deployment_health::NativeDeploymentHealth;
    #[pymodule_export]
    use crate::redis::store::rate_limits::{
        NativeRateLimits, PyAtomicApplied, PyAtomicOverLimit, PyGaugeOverLimit, PyGaugesAcquired,
        PyGuardedReply, PyWindowCount,
    };
    #[pymodule_export]
    use crate::redis::store::request::{
        NativeRequestRedis, NativeSend, drain_post_call_redis_batches,
    };
    #[pymodule_export]
    use crate::redis::store::router_session_pins::NativeRouterSessionPins;
    #[pymodule_export]
    use crate::redis::store::router_usage::{NativeRouterUsage, NativeUsageRead};
    #[pymodule_export]
    use crate::redis::store::sensitive_route_pins::NativeSensitiveRoutePins;
    #[pymodule_export]
    use crate::redis::store::spend_counters::{NativeSpendCounters, PySpendRead};
    #[pymodule_export]
    use crate::redis::store::{
        NativePending, NativeRedisStoreError, router_cooldowns::NativeRouterCooldowns,
    };
    #[pymodule_export]
    use crate::router::{AttemptShape, RouterCall, RouterCoreHandle};
    #[pymodule_export]
    use crate::routes::audio_transcription::{atranscription, transcription};
    #[pymodule_export]
    use crate::routes::chat_completions::{acompletion, completion};
    #[pymodule_export]
    use crate::routes::clickhouse_spend::{
        NativeClickHouseSpendConfig, NativeClickHouseSpendStorage,
    };
    #[pymodule_export]
    use crate::routes::embeddings::{aembedding, embedding};
    #[pymodule_export]
    use crate::routes::messages::{amessages, messages, messages_optional_params};
    #[pymodule_export]
    use crate::routes::ocr::{aocr, ocr, ocr_health_check_document, ocr_passthrough_response};
    #[pymodule_export]
    use crate::routes::responses::{ResponsesWebSocketConnection, aresponses, responses};
    #[pymodule_export]
    use crate::routes::token_counter::TokenCounter;
    #[pymodule_export]
    use crate::routes::traces::trace_encode_error;
    #[pymodule_export]
    use crate::s3_log::NativeS3Log;
    #[pymodule_export]
    use crate::spend_log::RustSpendLog;
    #[pymodule_export]
    use crate::spend_log_rows::RustSpendLogRowBuilder;
    #[pymodule_export]
    use crate::spend_log_writer::RustSpendLogWriter;
    #[cfg(feature = "huggingface")]
    #[pymodule_export]
    use crate::tokenizer::HuggingFaceEncoding;
    #[pymodule_export]
    use crate::tokenizer::Tokenizer;
    #[pymodule_export]
    use litellm_callbacks_legacy_python::{NativeBilledRates, NativeReceipt};

    #[pymodule_init]
    fn init(module: &Bound<'_, PyModule>) -> PyResult<()> {
        let _ = crate::crypto_provider().install_default();
        let _ = crate::logger::process().install_global();
        module.add(
            "JWT_ID_JAG_TYP",
            litellm_gateway_auth_core::authn::jwt::ID_JAG_TYP,
        )?;
        module.add(
            "BOUND_DEFAULT_MAX_OUTPUT_TOKENS",
            litellm_cost::bound::DEFAULT_MAX_OUTPUT_TOKENS,
        )?;
        let py = module.py();
        let dict = module.dict();
        dict.set_item(
            "_SecretManagerRuntime",
            py.get_type::<crate::secrets::runtime::NativeSecretManager>(),
        )
    }
}

use pyo3::prelude::*;

#[cfg(test)]
pub(crate) fn native_module(py: Python<'_>) -> Bound<'_, PyModule> {
    pyo3::wrap_pymodule!(_native)(py).into_bound(py)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rstest::rstest]
    fn module_registration_preserves_the_public_surface() {
        Python::initialize();
        Python::attach(|py| {
            let mut expected = vec![
                "RustBridgeDeclined",
                "RustUpstreamError",
                "ForkedAfterNativeRuntimeStarted",
                "ProcessReservedForForking",
                "ocr",
                "aocr",
                "ocr_health_check_document",
                "ocr_passthrough_response",
                "embedding",
                "aembedding",
                "transcription",
                "atranscription",
                "messages",
                "amessages",
                "messages_optional_params",
                "completion",
                "acompletion",
                "responses",
                "aresponses",
                "ResponsesWebSocketConnection",
                "NativeDiagnosticProcessor",
                "NativeTokenRedactor",
                "RustSpendLogRowBuilder",
                "NativeRedisClient",
                "NativeRedisBreaker",
                "NativeAdmission",
                "NativeRequestRedis",
                "NativeSend",
                "drain_post_call_redis_batches",
                "NativeAuthCache",
                "NativeTeamRoster",
                "NativeDeploymentHealth",
                "NativeRouterSessionPins",
                "AuthObject",
                "NativeCacheInvalidation",
                "NativeRateLimits",
                "WindowCount",
                "AtomicApplied",
                "AtomicOverLimit",
                "GuardedReply",
                "GaugesAcquired",
                "GaugeOverLimit",
                "NativeParallelRequestLimiter",
                "NativeParallelRequestLimiterState",
                "NativePending",
                "NativeRouterCooldowns",
                "NativeRouterUsage",
                "NativeUsageRead",
                "NativeSensitiveRoutePins",
                "NativeActiveTeams",
                "ActiveMember",
                "NativeRedisStoreError",
                "NativeSpendCounters",
                "SpendRead",
                "LocalSizes",
                "native_redis_breaker_states",
                "NativeClickHouseSpendConfig",
                "NativeClickHouseSpendStorage",
                "DatabaseEndpoint",
                "NativeDatabase",
                "NativeDatabaseFailure",
                "RustKeyRepository",
                "RustKeyOwnersRepository",
                "RustJwtKeyMappingRepository",
                "RustModelAccessGroupRepository",
                "RustBudgetRepository",
                "RustEndUserRepository",
                "RustUserRepository",
                "RustOrganizationRepository",
                "RustTeamMembershipRepository",
                "RustTeamRepository",
                "RustAccessGroupRepository",
                "RustAgentIdentityRepository",
                "RustAuditLogRepository",
                "RustObjectPermissionRepository",
                "RustProjectRepository",
                "RustTagRepository",
                "RustSpendLogRepository",
                "trace_encode_error",
                "bundled_model_cost_map",
                "NativeModelCostMap",
                "NativePrices",
                "NativeTokenUsage",
                "NativeCallContext",
                "NativePriced",
                "NativeQuote",
                "NativeReceipt",
                "NativeBilledRates",
                "NativeRateCard",
                "NativeModelCostMapState",
                "NativeInstalled",
                "NativeRejected",
                "NativeRegistered",
                "NativeTableSwapped",
                "NativeProvenance",
                "NativeCostMapSource",
                "NativeFetchPolicy",
                "NativeFetched",
                "NativeFetchFailed",
                "CostMapRefreshed",
                "CostMapRefreshStop",
                "TokenCounter",
                "Tokenizer",
                "RouterCoreHandle",
                "RouterCall",
                "AttemptShape",
                "gil_stats",
                "process_state_started",
                "reserve_process_for_forking",
                "unsupported_tls_settings",
                "NativeBudgets",
                "billable_route_passthrough_prefixes",
                "billable_service_tier",
                "classify_billable_route",
                "budget_check_decision",
                "NativeModelCapacity",
                "NativeModelCapacityState",
                "NativeS3Log",
                "RustSpendLog",
                "RustSpendLogWriter",
                "auth_allowed_routes_grant",
                "auth_first_denied_route",
                "auth_jwt_team_routes_grant",
                "auth_managed_agent_route_allowed",
                "auth_managed_route",
                "auth_route_matches_denied",
                "auth_route_matches_prefix",
                "auth_route_matches_wildcard",
                "JWT_ID_JAG_TYP",
                "BOUND_DEFAULT_MAX_OUTPUT_TOKENS",
                "RustAuth",
                "RustAuthOutcome",
                "RustAuthSettings",
                "RustAuthStep",
                "RustClaimIdentity",
                "RustDenial",
                "RustIdentity",
                "RustJwtMapping",
                "RustKeyContext",
                "RustMappedClaims",
                "RustPlan",
                "RustTeamGrant",
                "jwt_claims",
                "jwt_is_allowed_domain",
                "jwt_is_jwt",
                "jwt_presents_claims",
                "jwt_unverified_claims",
            ];
            #[cfg(feature = "huggingface")]
            expected.push("HuggingFaceEncoding");
            expected.sort_unstable();

            let mut public_names: Vec<String> = native_module(py)
                .dict()
                .keys()
                .extract::<Vec<String>>()
                .expect("module names should be strings")
                .into_iter()
                .filter(|name| !name.starts_with('_'))
                .collect();
            public_names.sort_unstable();
            assert_eq!(public_names, expected);
        });
    }
    #[rstest::rstest]
    fn module_import_installs_the_build_crypto_provider() {
        Python::initialize();
        Python::attach(|py| {
            native_module(py);
        });

        let installed = rustls::crypto::CryptoProvider::get_default()
            .expect("importing _native should install a process crypto provider");
        assert_eq!(
            format!("{installed:?}"),
            format!("{:?}", crate::crypto_provider())
        );
    }
}
