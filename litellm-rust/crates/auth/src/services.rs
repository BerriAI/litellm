#[derive(Default)]
pub struct AuthServices {
    #[cfg(feature = "aws")]
    pub aws: litellm_auth_aws::AwsAuthService,
    #[cfg(feature = "azure")]
    pub azure: litellm_auth_azure::AzureAuthService,
    #[cfg(feature = "gcp")]
    pub gcp: litellm_auth_gcp::VertexAuth,
    #[cfg(feature = "copilot")]
    pub copilot: litellm_auth_copilot::CopilotAuthService,
}
