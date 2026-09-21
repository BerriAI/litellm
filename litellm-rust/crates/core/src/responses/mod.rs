mod error;
pub use error::Error;
pub mod litellm_completion_transformation;
pub mod route;
pub mod streaming_iterator;
pub mod websocket;
pub mod websocket_session;

use litellm_types::responses::streaming::ResponsesResponse;

pub async fn responses(_call: route::ResponsesCall) -> Result<ResponsesResponse, Error> {
    todo!(
        "Resolve native or bridged provider, prepare/send HTTP request, and normalize the response through route host operations"
    )
}
