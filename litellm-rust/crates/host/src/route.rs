/// One public call surface: what a completed call produces, how it fails, and the
/// route-specific operations only its host can perform (request projection, file reads,
/// token acquisition).
pub trait Route: Send + Sync + 'static {
    type Response: Send + 'static;
    type Error: Clone + Send + Sync + 'static;
    type Op: Send + 'static;
    type OpResult: Send + 'static;
    /// One piece of a streamed response, handed to the caller as it arrives. A route
    /// that never streams uses `Infallible`.
    type Chunk: Send + 'static;
    /// What the route knows once a streamed response starts, before its first chunk.
    type StreamHead: Send + 'static;
}
