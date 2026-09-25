use pyo3::prelude::*;
use strum::{IntoStaticStr, VariantArray};

const MODULE: &str = "litellm.rust_bridge.callbacks_legacy_python";

/// Every litellm Python internal the native call still borrows, grouped by the subsystem it
/// belongs to. Rust drives the call; these exist only so behaviour that Python owns today
/// (span tracking, the standard logging payload, spend, callback fan-out) keeps working.
/// A group is deleted once Rust owns that subsystem, so this enum only shrinks. Calling a
/// user's own callback is not borrowing and does not belong here.
///
/// `litellm/rust_bridge/callbacks_legacy_python.py` is the only Python module behind it, and
/// `python_contract.json` pins each function's parameters on both sides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LegacyPython {
    Wrapper(Wrapper),
    Logging(Logging),
    DeploymentHooks(DeploymentHooks),
    Streaming(Streaming),
}

/// The `@client` wrapper around the call: `function_setup`, response metadata and the
/// correlation context.
#[derive(Clone, Copy, Debug, IntoStaticStr, PartialEq, Eq, VariantArray)]
pub(crate) enum Wrapper {
    #[strum(serialize = "setup")]
    Setup,
    #[strum(serialize = "is_internal_call")]
    IsInternalCall,
    #[strum(serialize = "finalize")]
    Finalize,
    #[strum(serialize = "restore_context")]
    RestoreContext,
}

/// litellm's `Logging` object and the sync and async callback fan-out behind it.
#[derive(Clone, Copy, Debug, IntoStaticStr, PartialEq, Eq, VariantArray)]
pub(crate) enum Logging {
    #[strum(serialize = "custom_pricing_fields")]
    CustomPricingFields,
    #[strum(serialize = "update_logging")]
    Update,
    #[strum(serialize = "pre_call")]
    PreCall,
    #[strum(serialize = "post_call")]
    PostCall,
    #[strum(serialize = "defers_async_logging")]
    DefersAsync,
    #[strum(serialize = "defer_success")]
    DeferSuccess,
    #[strum(serialize = "sync_success_for_async_call")]
    SyncSuccessForAsyncCall,
    #[strum(serialize = "submit_success")]
    SubmitSuccess,
    #[strum(serialize = "async_success_handler")]
    AsyncSuccessHandler,
    #[strum(serialize = "enqueue_logging")]
    Enqueue,
    #[strum(serialize = "failure_handler")]
    FailureHandler,
}

/// The fan-outs that run one hook of every registered callback.
#[derive(Clone, Copy, Debug, IntoStaticStr, PartialEq, Eq, VariantArray)]
pub(crate) enum DeploymentHooks {
    #[strum(serialize = "pre_request_hooks")]
    PreRequest,
    #[strum(serialize = "before_deployment_call")]
    BeforeDeploymentCall,
    #[strum(serialize = "after_deployment_success")]
    AfterDeploymentSuccess,
    #[strum(serialize = "after_deployment_failure")]
    AfterDeploymentFailure,
}

/// The Messages stream iterator's logging: the stream flag, the end-of-stream billing
/// from the delivered chunks, and the partial-usage failure path.
#[derive(Clone, Copy, Debug, IntoStaticStr, PartialEq, Eq, VariantArray)]
pub(crate) enum Streaming {
    #[strum(serialize = "stream_opened")]
    Opened,
    #[strum(serialize = "stream_success")]
    Success,
    #[strum(serialize = "stream_failure")]
    Failure,
}

impl LegacyPython {
    fn name(self) -> &'static str {
        match self {
            Self::Wrapper(function) => function.into(),
            Self::Logging(function) => function.into(),
            Self::DeploymentHooks(function) => function.into(),
            Self::Streaming(function) => function.into(),
        }
    }

    pub(crate) fn call<'py, A>(self, py: Python<'py>, args: A) -> PyResult<Bound<'py, PyAny>>
    where
        A: pyo3::call::PyCallArgs<'py>,
    {
        py.import(MODULE)?.getattr(self.name())?.call1(args)
    }
}

impl Wrapper {
    pub(crate) fn call<'py, A>(self, py: Python<'py>, args: A) -> PyResult<Bound<'py, PyAny>>
    where
        A: pyo3::call::PyCallArgs<'py>,
    {
        LegacyPython::Wrapper(self).call(py, args)
    }
}

impl Logging {
    pub(crate) fn call<'py, A>(self, py: Python<'py>, args: A) -> PyResult<Bound<'py, PyAny>>
    where
        A: pyo3::call::PyCallArgs<'py>,
    {
        LegacyPython::Logging(self).call(py, args)
    }
}

impl Streaming {
    pub(crate) fn call<'py, A>(self, py: Python<'py>, args: A) -> PyResult<Bound<'py, PyAny>>
    where
        A: pyo3::call::PyCallArgs<'py>,
    {
        LegacyPython::Streaming(self).call(py, args)
    }
}

impl DeploymentHooks {
    pub(crate) fn call<'py, A>(self, py: Python<'py>, args: A) -> PyResult<Bound<'py, PyAny>>
    where
        A: pyo3::call::PyCallArgs<'py>,
    {
        LegacyPython::DeploymentHooks(self).call(py, args)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use strum::VariantArray;

    use super::{DeploymentHooks, LegacyPython, Logging, Streaming, Wrapper};
    use crate::test_support::PYTHON_CONTRACT;

    #[test]
    fn every_borrowed_function_is_in_the_python_contract() {
        let contract: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(PYTHON_CONTRACT).unwrap();
        let declared: BTreeSet<&str> = contract.keys().map(String::as_str).collect();
        let called: Vec<&str> = Wrapper::VARIANTS
            .iter()
            .map(|&function| LegacyPython::Wrapper(function))
            .chain(
                Logging::VARIANTS
                    .iter()
                    .map(|&function| LegacyPython::Logging(function)),
            )
            .chain(
                DeploymentHooks::VARIANTS
                    .iter()
                    .map(|&function| LegacyPython::DeploymentHooks(function)),
            )
            .chain(
                Streaming::VARIANTS
                    .iter()
                    .map(|&function| LegacyPython::Streaming(function)),
            )
            .map(LegacyPython::name)
            .collect();
        assert_eq!(called.len(), declared.len(), "a function is borrowed twice");
        assert_eq!(called.into_iter().collect::<BTreeSet<_>>(), declared);
    }
}
