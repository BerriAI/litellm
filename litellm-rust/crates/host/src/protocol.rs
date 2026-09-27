/// One public call surface: what a completed call produces, how it fails, what the host
/// projects the caller's request into, and the protocol-specific operations only its host
/// can perform mid-call (token acquisition, for one).
pub trait Protocol: Send + Sync + 'static {
    type Response: Send + 'static;
    type Error: Clone + Send + Sync + 'static;
    /// The caller's request as the host projects it, answered once before anything else.
    type Projection: Send + 'static;
    /// Each operation carries the [`Reply`](crate::host::Reply) its answer goes through.
    /// A protocol with no operations of its own uses `Infallible`.
    type Op: Send + 'static;
    /// One piece of a streamed response, handed to the caller as it arrives. A protocol
    /// that never streams uses `Infallible`.
    type Chunk: Send + 'static;
    /// What the call knows once a streamed response starts, before its first chunk.
    type StreamHead: Send + 'static;
}
