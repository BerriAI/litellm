mod cache;
mod callable;
mod coercion;
mod credentials;
mod db;
mod diagnostics;
mod errors;
mod execution;
mod http;
mod license;
mod logger;
mod marshal;
mod model_cost;
mod model_registry;
mod preflight;
mod python_settings;
mod redis;
mod router;
mod routes;
mod secrets;
mod tokenizer;

#[pymodule(gil_used = true)]
mod _native {
    #[pymodule_export]
    use litellm_host_python::{ForkedAfterNativeRuntimeStarted, ProcessReservedForForking};
    use pyo3::{prelude::*, types::PyModule};

    #[pymodule_export]
    use crate::db::NativeDatabase;
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
    use crate::license::{Entitlements, LicenseVerifier};
    #[pymodule_export]
    use crate::logger::NativeDiagnosticProcessor;
    #[pymodule_export]
    use crate::model_cost::{
        NativeCostMapSource, NativeFetchFailed, NativeFetchPolicy, NativeFetched, NativeInstalled,
        NativeModelCostMap, NativeModelCostMapState, NativeProvenance, NativeRegistered,
        NativeRejected, NativeTableSwapped,
    };
    #[pymodule_export]
    use crate::model_registry::bundled_model_cost_map;
    #[pymodule_export]
    use crate::redis::breaker::{NativeAdmission, NativeRedisBreaker, native_redis_breaker_states};
    #[pymodule_export]
    use crate::redis::client::NativeRedisClient;
    #[pymodule_export]
    use crate::redis::store::auth_cache::{NativeAuthCache, PyLocalSizes};
    #[pymodule_export]
    use crate::redis::store::deployment_health::NativeDeploymentHealth;
    #[pymodule_export]
    use crate::redis::store::rate_limits::{
        NativeRateLimits, PyAtomicApplied, PyAtomicOverLimit, PyGaugeOverLimit, PyGaugesAcquired,
        PyGuardedReply, PyWindowCount,
    };
    #[pymodule_export]
    use crate::redis::store::request::{NativeRequestRedis, NativeSend};
    #[pymodule_export]
    use crate::redis::store::router_session_pins::NativeRouterSessionPins;
    #[pymodule_export]
    use crate::redis::store::router_usage::{NativeRouterUsage, NativeUsageRead};
    #[pymodule_export]
    use crate::redis::store::sensitive_route_pins::NativeSensitiveRoutePins;
    #[pymodule_export]
    use crate::redis::store::spend_counters::{
        NativeSpendCounters, PyPostCallSettlement, PySpendRead,
    };
    #[pymodule_export]
    use crate::redis::store::{
        NativePending, NativeRedisStoreError, router_cooldowns::NativeRouterCooldowns,
    };
    #[pymodule_export]
    use crate::router::{RouterCall, RouterCoreHandle};
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
    use crate::routes::messages::{amessages, messages};
    #[pymodule_export]
    use crate::routes::ocr::{aocr, ocr, ocr_health_check_document, ocr_passthrough_response};
    #[pymodule_export]
    use crate::routes::responses::{ResponsesWebSocketConnection, aresponses, responses};
    #[pymodule_export]
    use crate::routes::token_counter::TokenCounter;
    #[pymodule_export]
    use crate::routes::traces::trace_encode_error;
    #[cfg(feature = "huggingface")]
    #[pymodule_export]
    use crate::tokenizer::HuggingFaceEncoding;
    #[pymodule_export]
    use crate::tokenizer::Tokenizer;

    #[pymodule_init]
    fn init(module: &Bound<'_, PyModule>) -> PyResult<()> {
        let _ = rustls::crypto::ring::default_provider().install_default();
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
                "completion",
                "acompletion",
                "responses",
                "aresponses",
                "ResponsesWebSocketConnection",
                "NativeDiagnosticProcessor",
                "Entitlements",
                "LicenseVerifier",
                "NativeRedisClient",
                "NativeRedisBreaker",
                "NativeAdmission",
                "NativeRequestRedis",
                "NativeSend",
                "NativeAuthCache",
                "NativeDeploymentHealth",
                "NativeRouterSessionPins",
                "NativeRateLimits",
                "WindowCount",
                "AtomicApplied",
                "AtomicOverLimit",
                "GuardedReply",
                "GaugesAcquired",
                "GaugeOverLimit",
                "NativePending",
                "NativeRouterCooldowns",
                "NativeRouterUsage",
                "NativeUsageRead",
                "NativeSensitiveRoutePins",
                "NativeRedisStoreError",
                "NativeSpendCounters",
                "PostCallSettlement",
                "SpendRead",
                "LocalSizes",
                "native_redis_breaker_states",
                "NativeClickHouseSpendConfig",
                "NativeClickHouseSpendStorage",
                "NativeDatabase",
                "trace_encode_error",
                "bundled_model_cost_map",
                "NativeModelCostMap",
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
                "TokenCounter",
                "Tokenizer",
                "RouterCoreHandle",
                "RouterCall",
                "gil_stats",
                "process_state_started",
                "reserve_process_for_forking",
                "unsupported_tls_settings",
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
    fn module_import_installs_ring_as_the_process_crypto_provider() {
        Python::initialize();
        Python::attach(|py| {
            native_module(py);
        });

        let installed = rustls::crypto::CryptoProvider::get_default()
            .expect("importing _native should install a process crypto provider");
        assert_eq!(
            format!("{installed:?}"),
            format!("{:?}", rustls::crypto::ring::default_provider())
        );
    }
}
