/// Rust core cannot serve this request, and nothing was sent to the provider yet.
///
/// Every error on the call path carries it unchanged, so the host can hand the same request
/// to another implementation: the SDK runs legacy Python, the gateway forwards to the proxy.
/// Raise it only before the provider call. Anything after the send is terminal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("declined by the rust path: {reason}")]
pub struct Decline {
    pub reason: &'static str,
}

impl Decline {
    pub const fn new(reason: &'static str) -> Self {
        Self { reason }
    }
}
