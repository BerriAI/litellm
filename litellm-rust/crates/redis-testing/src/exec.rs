use testcontainers_modules::testcontainers::{
    ContainerAsync, Image,
    core::{CmdWaitFor, ExecCommand},
};

use crate::Error;

pub(crate) async fn run<I: Image>(
    container: &ContainerAsync<I>,
    command: Vec<String>,
) -> Result<String, Error> {
    let mut result = container
        .exec(ExecCommand::new(command.clone()).with_cmd_ready_condition(CmdWaitFor::exit()))
        .await?;
    let stdout = String::from_utf8_lossy(&result.stdout_to_vec().await?).into_owned();
    match result.exit_code().await? {
        Some(0) | None => Ok(stdout),
        Some(code) => Err(Error::Command {
            command: command.join(" "),
            code,
        }),
    }
}

pub(crate) fn cli(port: u16, args: &[&str]) -> Vec<String> {
    ["redis-cli", "-e", "-p", &port.to_string()]
        .into_iter()
        .chain(args.iter().copied())
        .map(str::to_owned)
        .collect()
}
