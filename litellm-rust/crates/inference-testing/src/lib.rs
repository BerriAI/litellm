use std::sync::{Arc, Mutex};

use futures_util::future::BoxFuture;
use litellm_http::{
    ClientVariant, HttpClientConfig, HttpClientPool, HttpSettings, Resolution,
    media::PublicDnsResolver,
};
use litellm_secrets::{SecretValue, source::SecretSource};

use litellm_inference::resources::CoreResources;

pub mod live;

pub fn http_pool() -> HttpClientPool {
    HttpClientPool::new(Arc::new(PublicDnsResolver))
}

pub fn resources() -> CoreResources {
    CoreResources::new(Arc::new(http_pool()))
}

pub fn no_secrets() -> Arc<dyn SecretSource> {
    Arc::new(RecordingSecrets::empty())
}

pub fn provider_http(resources: &CoreResources, config: &HttpClientConfig) -> litellm_http::Client {
    resources
        .pool
        .client(config, ClientVariant::Provider)
        .unwrap()
}

pub fn http_config() -> HttpClientConfig {
    Resolution::from(&HttpSettings::default()).config
}

pub struct RecordingSecrets {
    values: Vec<(String, String)>,
    fails: bool,
    requested: Mutex<Vec<String>>,
}

impl RecordingSecrets {
    pub fn new<'a>(values: impl IntoIterator<Item = (&'a str, &'a str)>) -> Self {
        Self {
            values: values
                .into_iter()
                .map(|(name, value)| (name.to_string(), value.to_string()))
                .collect(),
            fails: false,
            requested: Mutex::new(Vec::new()),
        }
    }

    pub fn empty() -> Self {
        Self::new([])
    }

    pub fn failing() -> Self {
        Self {
            fails: true,
            ..Self::empty()
        }
    }

    pub fn requested(&self) -> Vec<String> {
        self.requested.lock().unwrap().clone()
    }
}

impl SecretSource for RecordingSecrets {
    fn get_secret_str<'a>(
        &'a self,
        name: &'a str,
    ) -> BoxFuture<'a, Result<Option<SecretValue>, litellm_secrets::Error>> {
        Box::pin(async move {
            self.requested.lock().unwrap().push(name.to_string());
            if self.fails {
                return Err(litellm_secrets::Error::ManagedSecretMissing);
            }
            Ok(self
                .values
                .iter()
                .find(|(key, _)| key == name)
                .map(|(_, value)| SecretValue::new(value.clone())))
        })
    }
}
