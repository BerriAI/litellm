//! Which Python logger a Rust target's lines go to. Python proxy code logs on `LiteLLM Proxy`, the router on
//! `LiteLLM Router` and the SDK on `LiteLLM`, so a Rust port keeps its lines on the logger its Python version used.
//! This table is the only copy of that rule: Python is told the logger's name and never sees a Rust target.

use strum::{EnumCount, IntoStaticStr, VariantArray};

#[derive(Clone, Copy, Debug, PartialEq, Eq, EnumCount, IntoStaticStr, VariantArray)]
pub(super) enum PythonLogger {
    #[strum(serialize = "LiteLLM")]
    Sdk,
    #[strum(serialize = "LiteLLM Proxy")]
    Proxy,
    #[strum(serialize = "LiteLLM Router")]
    Router,
}

const TARGETS: [(&str, PythonLogger); 3] = [
    (
        "litellm_callbacks_parallel_request_limiter",
        PythonLogger::Proxy,
    ),
    ("litellm_callbacks_model_capacity", PythonLogger::Proxy),
    ("litellm_router_core::model_info", PythonLogger::Router),
];

impl PythonLogger {
    pub(super) fn of(target: &str) -> Self {
        TARGETS
            .iter()
            .find(|(prefix, _)| target.starts_with(prefix))
            .map_or(Self::Sdk, |(_, logger)| *logger)
    }
}

#[cfg(test)]
mod tests {
    use super::PythonLogger;

    #[rstest::rstest]
    #[case::limiter(
        "litellm_callbacks_parallel_request_limiter::hosted",
        PythonLogger::Proxy
    )]
    #[case::limiter_crate_root("litellm_callbacks_parallel_request_limiter", PythonLogger::Proxy)]
    #[case::capacity("litellm_callbacks_model_capacity::capacity", PythonLogger::Proxy)]
    #[case::router_model_info("litellm_router_core::model_info", PythonLogger::Router)]
    #[case::router_call("litellm_router_core::call", PythonLogger::Sdk)]
    #[case::another_callback("litellm_callbacks_budgets::admission", PythonLogger::Sdk)]
    #[case::store("litellm_redis_store::cache_invalidation", PythonLogger::Sdk)]
    #[case::bridge("_native::redis::events", PythonLogger::Sdk)]
    fn a_target_logs_on_the_logger_its_python_version_used(
        #[case] target: &str,
        #[case] logger: PythonLogger,
    ) {
        assert_eq!(PythonLogger::of(target), logger);
    }
}
