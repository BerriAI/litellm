use std::{
    env,
    process::{Command, ExitCode},
};

use litellm_db_testing::{Error, MigratedPostgres};

#[tokio::main]
async fn main() -> Result<ExitCode, Error> {
    let postgres = MigratedPostgres::start().await?;
    let status = Command::new(env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned()))
        .args(["sqlx", "prepare"])
        .args(env::args().skip(1))
        .args(["--", "--all-targets"])
        .current_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../db"))
        .env("DATABASE_URL", postgres.url())
        .env("SQLX_OFFLINE", "false")
        .status()
        .map_err(Error::Prepare)?;

    Ok(if status.success() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}
