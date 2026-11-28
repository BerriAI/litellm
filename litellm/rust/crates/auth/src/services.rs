#[derive(Default)]
pub struct AuthServices {
    #[cfg(feature = "aws")]
    pub aws: litellm_auth_aws::AwsAuthService,
    #[cfg(feature = "azure")]
    pub azure: litellm_auth_azure::AzureAuthService,
    #[cfg(feature = "gcp")]
    pub gcp: litellm_auth_gcp::VertexAuth,
    #[cfg(feature = "gcp")]
    pub vertex_login: litellm_auth_gcp::vertex::VertexLoginService,
}

#[cfg(all(feature = "aws", feature = "gcp"))]
impl AuthServices {
    /// Provider auth whose AWS credentials resolve an `oidc/` web identity token through
    /// `tokens`, for Bedrock and for Claude on Vertex AI's AWS federation alike.
    pub fn with_web_identity_tokens(
        tokens: std::sync::Arc<dyn litellm_auth_aws::WebIdentityTokenSource>,
    ) -> Self {
        let aws = litellm_auth_aws::AwsAuthService::with_web_identity_tokens(tokens);
        Self {
            vertex_login: litellm_auth_gcp::vertex::VertexLoginService::default()
                .with_aws(aws.clone()),
            aws,
            ..Self::default()
        }
    }
}
