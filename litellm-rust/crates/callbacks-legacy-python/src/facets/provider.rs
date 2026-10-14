use litellm_host::{
    hooks::ProviderHooks,
    interceptors::{ExecutionFacts, RawResponse, RequestContext, WireRequest},
};
use litellm_host_python::{Awaiter, PythonSurface};
use pyo3::prelude::*;

use crate::{CallBoundary, LegacyHooks};

pub(crate) const CALLBACKS: &[(CallBoundary, &[&str])] = &[
    (
        CallBoundary::BeforeProviderRequest,
        &["log_pre_api_call", "log_input_event"],
    ),
    (CallBoundary::AfterProviderResponse, &["log_post_api_call"]),
];

impl ProviderHooks<PythonSurface> for LegacyHooks {
    /// `Logging.pre_call` over the wire request, which the callbacks may edit in place.
    async fn before_provider_request<'a>(
        &'a self,
        _: &'a dyn Awaiter,
        wire: Box<WireRequest>,
        context: &RequestContext,
    ) -> PyResult<Box<WireRequest>> {
        self.with(|logging, py| logging.pre_call(py, wire, context))
    }

    /// `Logging.post_call` with the raw provider response.
    async fn after_provider_response<'a>(
        &'a self,
        _: &'a dyn Awaiter,
        raw: &RawResponse,
    ) -> PyResult<()> {
        self.with(|logging, py| logging.post_call(py, raw))
    }

    /// The provider, cost and cache facts the spend record is built from.
    async fn result_ready<'a>(
        &'a self,
        _: &'a dyn Awaiter,
        facts: &ExecutionFacts,
    ) -> PyResult<()> {
        self.with(|logging, py| logging.result_ready(py, facts))
    }
}
