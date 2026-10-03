use std::io;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("could not read migrations directory `{path}`")]
    ReadDirectory {
        path: String,
        #[source]
        source: io::Error,
    },
    #[error(
        "migration name `{name}` must be `<digits>_<description>.sql` with a `[a-z0-9_]` description"
    )]
    InvalidName { name: String },
    #[error("migration version `{version}` is declared more than once")]
    DuplicateVersion { version: u64 },
    #[error("migrations directory `{path}` contains no migrations")]
    Empty { path: String },
    #[error("migration path `{path}` is not valid UTF-8")]
    NonUtf8Path { path: String },
}
