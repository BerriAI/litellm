use std::{io, path::PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid Postgres connection URL")]
    Url(#[from] url::ParseError),
    #[error("reading {path}")]
    Read {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("starting the Postgres container")]
    Container(#[from] testcontainers_modules::testcontainers::TestcontainersError),
    #[error("connecting to the Postgres container")]
    Connect(#[from] sqlx::Error),
    #[error("applying the Prisma migrations")]
    Migrate(#[from] sqlx::migrate::MigrateError),
    #[error("running cargo sqlx prepare")]
    Prepare(#[source] io::Error),
}
