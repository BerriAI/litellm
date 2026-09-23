use super::*;

impl AwsSecretsManagerV2 {
    pub(super) fn with_context_client_factory(
        client: Client,
        write_settings: AwsSecretWriteSettings,
        context_client_factory: ContextClientFactory,
    ) -> Self {
        Self {
            client,
            context_client_factory: Some(Box::new(context_client_factory)),
            write_settings,
        }
    }

    pub fn load_aws_secret_manager(
        use_aws_secret_manager: Option<bool>,
        settings: KeyManagementSettings,
        environment: Arc<dyn Lookup + Send + Sync>,
    ) -> Result<Option<Self>, Error> {
        if use_aws_secret_manager != Some(true) {
            return Ok(None);
        }
        let context_client_factory = ContextClientFactory {
            settings: settings.clone(),
            environment: environment.clone(),
            endpoint_url: environment
                .get(AWS_BEDROCK_RUNTIME_ENDPOINT)
                .map(|url| url.replace("bedrock-runtime", "secretsmanager")),
        };
        let client = context_client_factory.client(&AwsOperationContext::default())?;
        Ok(Some(Self::with_context_client_factory(
            client,
            (&settings).into(),
            context_client_factory,
        )))
    }

    pub(super) fn client_for_context(
        &self,
        context: &AwsOperationContext,
    ) -> Result<Client, Error> {
        if context == &AwsOperationContext::default() {
            return Ok(self.client.clone());
        }
        self.context_client_factory
            .as_ref()
            .ok_or(Error::OperationContextUnavailable)?
            .client(context)
    }
}

impl ContextClientFactory {
    fn client(&self, context: &AwsOperationContext) -> Result<Client, Error> {
        let settings = KeyManagementSettings {
            aws_region_name: context
                .region_name
                .clone()
                .or_else(|| self.settings.aws_region_name.clone()),
            aws_role_name: context
                .role_name
                .clone()
                .or_else(|| self.settings.aws_role_name.clone()),
            aws_session_name: context
                .session_name
                .clone()
                .or_else(|| self.settings.aws_session_name.clone()),
            aws_external_id: context
                .external_id
                .clone()
                .or_else(|| self.settings.aws_external_id.clone()),
            aws_profile_name: context
                .profile_name
                .clone()
                .or_else(|| self.settings.aws_profile_name.clone()),
            aws_web_identity_token: context
                .web_identity_token
                .clone()
                .or_else(|| self.settings.aws_web_identity_token.clone()),
            aws_sts_endpoint: context
                .sts_endpoint
                .clone()
                .or_else(|| self.settings.aws_sts_endpoint.clone()),
            ..self.settings.clone()
        };
        let builder = aws_sdk_secretsmanager::Config::builder()
            .behavior_version(BehaviorVersion::latest())
            .region(Region::new(auth::region(
                &settings,
                self.environment.as_ref(),
            )?))
            .credentials_provider(auth::Credentials::with_context(
                &settings,
                self.environment.clone(),
                context,
            ));
        let builder = match context.timeout {
            Some(timeout) => builder.timeout_config(
                aws_sdk_secretsmanager::config::timeout::TimeoutConfig::builder()
                    .operation_timeout(timeout)
                    .build(),
            ),
            None => builder,
        };
        let endpoint_url = context
            .bedrock_runtime_endpoint
            .as_ref()
            .map(|url| url.replace("bedrock-runtime", "secretsmanager"))
            .or_else(|| self.endpoint_url.clone());
        let config = match endpoint_url {
            Some(endpoint_url) => builder.endpoint_url(endpoint_url).build(),
            None => builder.build(),
        };
        Ok(Client::from_conf(config))
    }
}
