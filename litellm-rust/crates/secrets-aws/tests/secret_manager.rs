use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use aws_sdk_secretsmanager::{
    Client,
    config::{BehaviorVersion, Credentials, Region, retry::RetryConfig},
};
use litellm_secrets_aws::{AwsSecretsManagerV2, Error, RotationResponse};
use litellm_secrets_types::{
    AwsOperationContext, BaseSecretManager, KeyManagementSettings, Secret, SecretDeleter,
    SecretValue, SecretWriteContext, SecretWriter,
};
use rstest::{fixture, rstest};
use serde_json::json;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_partial_json, header},
};

#[path = "secret_manager/support.rs"]
mod support;
use support::*;

#[path = "secret_manager/configuration.rs"]
mod configuration;
#[path = "secret_manager/reads.rs"]
mod reads;
#[path = "secret_manager/writes.rs"]
mod writes;
