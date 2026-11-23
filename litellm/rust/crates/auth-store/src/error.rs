/// Why an auth object was not read from the database.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("the auth object was not read from the database")]
    Database(#[from] litellm_db::Error),
    #[error("the row's `{column}` column is not what the auth cache holds for it")]
    Row { column: &'static str },
}
