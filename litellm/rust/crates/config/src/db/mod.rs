//! Config the proxy keeps in its database, merged into the config file's document. Callers
//! read and decrypt the rows; nothing here does I/O or crypto.

pub mod models;
