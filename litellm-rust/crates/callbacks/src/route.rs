/// One public call surface: what a completed call produces, how it fails, and the
/// route-specific operations only its host can perform (request projection, file reads,
/// token acquisition).
pub trait Route: Send + Sync + 'static {
    type Response: Send + 'static;
    type Error: Clone + Send + Sync + 'static;
    type Op: Send + 'static;
    type OpResult: Send + 'static;
}
