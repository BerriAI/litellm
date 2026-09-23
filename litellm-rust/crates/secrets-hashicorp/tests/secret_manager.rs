use std::{collections::HashMap, sync::Arc, time::Duration};

use litellm_core_utils::settings::Lookup;
use litellm_secrets_hashicorp::{Error, HashicorpVault, HashicorpVaultConfig};
use litellm_secrets_types::{
    BaseSecretManager, HashicorpOperationContext, RotationError, SecretDeleter, SecretValue,
    SecretWriteContext, SecretWriter,
};
use rstest::{fixture, rstest};
use serde::Deserialize;
use serde_json::json;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{body_json, header, method, path},
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
