pub mod console;

pub use litellm_core::logging::{
    BodySnapshot, ErrorEventInput, LogSink, ProviderDebugEvent, ProviderErrorEvent,
    ProviderRequestEvent, ProviderResponseEvent, ProviderStreamCompletedEvent,
    ProviderStreamStartedEvent, RequestEventInput, ResponseBody, ResponseEventInput, error_event,
    request_event, response_event, stream_completed, stream_started,
};
