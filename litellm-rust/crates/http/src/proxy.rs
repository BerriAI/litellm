use hyper_util::client::proxy::matcher::Matcher;

pub struct EnvironmentProxies(Matcher);

impl EnvironmentProxies {
    pub fn from_environment() -> Self {
        Self(Matcher::from_system())
    }

    pub fn apply_to(&self, url: &reqwest::Url) -> bool {
        url.as_str()
            .parse::<http::Uri>()
            .is_ok_and(|uri| self.0.intercept(&uri).is_some())
    }
}
