use std::ops::Deref;

/// Only `HttpClientPool` builds one, so every client carries the resolved TLS, proxy and timeout settings.
#[derive(Clone, Debug)]
pub struct Client(reqwest::Client);

impl Client {
    pub(crate) fn new(client: reqwest::Client) -> Self {
        Self(client)
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn for_test() -> Self {
        Self(
            reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .expect("a client without TLS or proxy settings builds"),
        )
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn wrap_for_test(client: reqwest::Client) -> Self {
        Self(client)
    }
}

impl Deref for Client {
    type Target = reqwest::Client;

    fn deref(&self) -> &reqwest::Client {
        &self.0
    }
}
