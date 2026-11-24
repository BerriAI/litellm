use futures_util::future::BoxFuture;
use litellm_router_types::LitellmParams;
use litellm_secret_value::SecretValue;

pub trait WorkloadIdentity: Send + Sync {
    fn client_credentials_token<'a>(
        &'a self,
        _litellm_params: &'a LitellmParams,
        _api_base: Option<&'a str>,
        _model: &'a str,
    ) -> BoxFuture<'a, Result<Option<SecretValue>, litellm_auth::Error>> {
        Box::pin(async { Ok(None) })
    }

    fn token<'a>(
        &'a self,
        litellm_params: &'a LitellmParams,
        api_base: Option<&'a str>,
    ) -> BoxFuture<'a, Result<Option<SecretValue>, litellm_auth::Error>>;
}
