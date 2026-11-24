//! Config the proxy stores in its database, layered over the config file the way the Python proxy
//! layers it once `store_model_in_db` is on. Nothing here reads the database or decrypts: the
//! caller fetches the rows, decrypts them and hands them in.

pub mod models;
pub mod settings;
pub mod switches;
