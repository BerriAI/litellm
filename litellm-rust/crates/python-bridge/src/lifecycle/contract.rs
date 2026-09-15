use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyString;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CallMode {
    Sync,
    Async,
}

impl CallMode {
    pub(crate) fn from_async(asynchronous: bool) -> Self {
        if asynchronous {
            Self::Async
        } else {
            Self::Sync
        }
    }

    pub(crate) fn is_async(self) -> bool {
        matches!(self, Self::Async)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PythonCallType {
    Ocr,
    AsyncOcr,
    Completion,
    AsyncCompletion,
    AnthropicMessages,
    Transcription,
    AsyncTranscription,
    #[cfg(test)]
    Synthetic,
    #[cfg(test)]
    Test,
}

impl PythonCallType {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Ocr => "ocr",
            Self::AsyncOcr => "aocr",
            Self::Completion => "completion",
            Self::AsyncCompletion => "acompletion",
            Self::AnthropicMessages => "anthropic_messages",
            Self::Transcription => "transcription",
            Self::AsyncTranscription => "atranscription",
            #[cfg(test)]
            Self::Synthetic => "synthetic",
            #[cfg(test)]
            Self::Test => "test",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CallbackPhase {
    Input,
    Payload,
    SyncSuccess,
    SyncSuccessForAsyncCall,
    AsyncSuccess,
    SyncFailure,
    AsyncFailure,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AdapterOperation {
    PostProcess,
    CacheStore,
    CachedResponse,
    ProjectRequest,
    BeforeRequest,
    AfterResponse,
    ConstructResponse,
    MapFailure,
}

impl AdapterOperation {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::PostProcess => "post_process",
            Self::CacheStore => "cache_response",
            Self::CachedResponse => "cached_response",
            Self::ProjectRequest => "project",
            Self::BeforeRequest => "before_request",
            Self::AfterResponse => "after_response",
            Self::ConstructResponse => "response",
            Self::MapFailure => "map_failure",
        }
    }
}

impl CallbackPhase {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Input => "input",
            Self::Payload => "payload",
            Self::SyncSuccess => "sync_success",
            Self::SyncSuccessForAsyncCall => "sync_success_async",
            Self::AsyncSuccess => "async_success",
            Self::SyncFailure => "sync_failure",
            Self::AsyncFailure => "async_failure",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RequestField {
    ApiBase,
    ApiKey,
    Audio,
    Body,
    CustomLlmProvider,
    ExtraHeaders,
    HasAgenticHook,
    HostFacts,
    LitellmCallId,
    Messages,
    Model,
    OptionalParams,
    Timeout,
}

impl RequestField {
    pub(crate) fn key(self, py: Python<'_>) -> &Bound<'_, PyString> {
        match self {
            Self::ApiBase => intern!(py, "api_base"),
            Self::ApiKey => intern!(py, "api_key"),
            Self::Audio => intern!(py, "audio"),
            Self::Body => intern!(py, "body"),
            Self::CustomLlmProvider => intern!(py, "custom_llm_provider"),
            Self::ExtraHeaders => intern!(py, "extra_headers"),
            Self::HasAgenticHook => intern!(py, "has_agentic_hook"),
            Self::HostFacts => intern!(py, "host_facts"),
            Self::LitellmCallId => intern!(py, "litellm_call_id"),
            Self::Messages => intern!(py, "messages"),
            Self::Model => intern!(py, "model"),
            Self::OptionalParams => intern!(py, "optional_params"),
            Self::Timeout => intern!(py, "timeout"),
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::ApiBase => "api_base",
            Self::ApiKey => "api_key",
            Self::Audio => "audio",
            Self::Body => "body",
            Self::CustomLlmProvider => "custom_llm_provider",
            Self::ExtraHeaders => "extra_headers",
            Self::HasAgenticHook => "has_agentic_hook",
            Self::HostFacts => "host_facts",
            Self::LitellmCallId => "litellm_call_id",
            Self::Messages => "messages",
            Self::Model => "model",
            Self::OptionalParams => "optional_params",
            Self::Timeout => "timeout",
        }
    }
}
