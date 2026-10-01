use std::ops::Deref;

#[derive(Clone, Debug)]
pub struct Client(reqwest::Client);

impl Client {
    pub(crate) fn new(client: reqwest::Client) -> Self {
        Self(client)
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn for_test(client: reqwest::Client) -> Self {
        Self(client)
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn plain_for_test() -> Self {
        Self(reqwest::Client::new())
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn no_redirect_for_test() -> Self {
        Self(
            reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .expect("a client without TLS or proxy settings builds"),
        )
    }
}

impl Deref for Client {
    type Target = reqwest::Client;

    fn deref(&self) -> &reqwest::Client {
        &self.0
    }
}
