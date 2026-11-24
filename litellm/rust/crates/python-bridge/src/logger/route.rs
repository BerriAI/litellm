//! Which Python logger a Rust target's lines go to. Python proxy code logs on `LiteLLM Proxy` and the SDK on
//! `LiteLLM`, so a Rust port keeps its lines on the logger its Python version used. This table is the only copy of
//! that rule: Python is told the logger's name and never sees a Rust target.

use strum::{EnumCount, IntoStaticStr, VariantArray};

#[derive(Clone, Copy, Debug, PartialEq, Eq, EnumCount, IntoStaticStr, VariantArray)]
pub(super) enum PythonLogger {
    #[strum(serialize = "LiteLLM")]
    Sdk,
    #[strum(serialize = "LiteLLM Proxy")]
    Proxy,
}

const PROXY_TARGETS: [&str; 1] = ["litellm_callbacks_parallel_request_limiter"];

impl PythonLogger {
    pub(super) fn of(target: &str) -> Self {
        if PROXY_TARGETS
            .iter()
            .any(|prefix| target.starts_with(prefix))
        {
            return Self::Proxy;
        }
        Self::Sdk
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
