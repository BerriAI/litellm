use std::sync::Arc;

#[derive(Clone, Debug, thiserror::Error)]
pub(crate) enum LocalTokenizerRegistryError {
    #[error("local tokenizer registry capacity is full")]
    Capacity,
    #[error("local tokenizer worker failed")]
    Worker,
    #[error("local tokenizer configuration failed: {0}")]
    Tokenizer(#[source] Arc<litellm_token_counter::Error>),
}
