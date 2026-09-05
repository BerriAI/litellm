use thiserror::Error as ThisError;

#[derive(Debug, ThisError)]
pub enum Error {
    #[error("read_model_list failed: {0}")]
    PythonLoading(String),
    #[error("serializing model_list failed: {0}")]
    Serialization(String),
    #[error("parsing model_list failed: {0}")]
    ModelListParsing(#[source] serde_json::Error),
}
