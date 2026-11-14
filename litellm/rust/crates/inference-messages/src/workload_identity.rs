use futures_util::future::BoxFuture;
use litellm_router_types::LitellmParams;
use litellm_secret_value::SecretValue;

pub trait WorkloadIdentity: Send + Sync {
    fn token<'a>(
        &'a self,
        litellm_params: &'a LitellmParams,
        api_base: Option<&'a str>,
    ) -> BoxFuture<'a, Result<Option<SecretValue>, litellm_auth::Error>>;
}
