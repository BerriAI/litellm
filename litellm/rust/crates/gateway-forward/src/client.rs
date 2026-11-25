use std::net::SocketAddr;

use axum::extract::Request;

use crate::NotForwarded;

/// The client's TCP connection to the public listener.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ClientSocket {
    pub peer: SocketAddr,
    pub local: SocketAddr,
}

impl ClientSocket {
    /// The socket the serving loop put in `request`'s extensions, as axum's `ConnectInfo` does.
    pub fn of(request: &Request) -> Result<Self, NotForwarded> {
        request
            .extensions()
            .get::<Self>()
            .copied()
            .ok_or(NotForwarded::Internal)
    }
}
