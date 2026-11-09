mod cache;
mod callable;
mod coercion;
mod credentials;
mod db;
mod diagnostics;
mod errors;
mod execution;
mod http;
mod logger;
mod marshal;
mod model_registry;
mod preflight;
mod python_settings;
mod request_auth;
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
    use crate::logger::NativeDiagnosticProcessor;
    #[pymodule_export]
    use crate::model_registry::bundled_model_cost_map;
    #[pymodule_export]
    use crate::request_auth::{
        NativeAuthContinuation, request_auth_access, request_auth_admission,
        request_auth_agent_access, request_auth_agent_ceiling, request_auth_agent_ceiling_source,
        request_auth_agent_grants, request_auth_agent_group_load, request_auth_agent_invocation,
        request_auth_budget_alert, request_auth_budget_alert_plan, request_auth_budget_exemption,
        request_auth_budget_price, request_auth_budget_route, request_auth_budget_window_start,
        request_auth_budgets, request_auth_common_budget, request_auth_common_budget_scopes,
        request_auth_common_flow, request_auth_common_read, request_auth_context,
        request_auth_context_end_user, request_auth_context_member, request_auth_context_models,
        request_auth_context_percentage, request_auth_context_temporary,
        request_auth_context_throttle, request_auth_context_token, request_auth_custom_flow,
        request_auth_end_user_apply, request_auth_end_user_assignments,
        request_auth_end_user_default_id, request_auth_end_user_lookup,
        request_auth_end_user_normalize_id, request_auth_end_user_registry,
        request_auth_end_user_validation, request_auth_expired, request_auth_fallback_access,
        request_auth_fallback_candidate, request_auth_key_model_access,
        request_auth_litellm_header_token, request_auth_managed, request_auth_managed_model_grants,
        request_auth_managed_models, request_auth_member_limit, request_auth_member_models,
        request_auth_model_access, request_auth_model_attribution, request_auth_model_budget_check,
        request_auth_model_budget_query, request_auth_model_budget_resolve,
        request_auth_model_groups, request_auth_model_helpers, request_auth_org_admin,
        request_auth_org_route, request_auth_owner_teams, request_auth_personal_budget,
        request_auth_personal_model_listing, request_auth_personal_model_request,
        request_auth_personal_models, request_auth_personal_owner, request_auth_reservation,
        request_auth_resolved_models, request_auth_route_access, request_auth_scheme_credential,
        request_auth_select_api_key, request_auth_select_websocket_api_key,
        request_auth_spend_counter_policy, request_auth_start, request_auth_team_aliases,
        request_auth_team_override, request_auth_unpriced_model,
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
                "NativeClickHouseSpendConfig",
                "NativeClickHouseSpendStorage",
                "NativeDatabase",
                "trace_encode_error",
                "bundled_model_cost_map",
                "TokenCounter",
                "Tokenizer",
                "RouterCoreHandle",
                "RouterCall",
                "gil_stats",
                "process_state_started",
                "reserve_process_for_forking",
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
}
