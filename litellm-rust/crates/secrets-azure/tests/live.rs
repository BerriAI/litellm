use std::sync::Arc;

use litellm_core_utils::settings::ProcessEnvironment;
use litellm_secrets_azure::AzureKeyVault;
use litellm_secrets_types::Secret;
use rstest::rstest;

#[rstest]
#[tokio::test]
#[ignore]
async fn reads_a_real_secret() {
    let environment = Arc::new(ProcessEnvironment);
    let manager = AzureKeyVault::new(litellm_http::Client::plain_for_test(), environment).unwrap();
    let name = std::env::var("AZURE_KEY_VAULT_LIVE_SECRET_NAME").unwrap();
    let secret = manager.get_secret(&name).await.unwrap().unwrap();
    assert!(matches!(&secret, Secret::String(_)));
    let host = std::env::var("AZURE_KEY_VAULT_URI")
        .unwrap()
        .parse::<reqwest::Url>()
        .unwrap()
        .host_str()
        .unwrap()
        .to_owned();
    let value_len = secret.as_str().unwrap().len();
    println!(
        "native provider=litellm-secrets-azure vault_host={host} secret={name} value_len={value_len}"
    );
}
