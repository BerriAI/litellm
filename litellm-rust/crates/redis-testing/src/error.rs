use std::io;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("reserving a host port for the Redis container")]
    Port(#[source] io::Error),
    #[error("Redis on host port {port} did not answer PING")]
    Unreachable { port: u16 },
    #[error("controlling the Redis container")]
    Container(#[from] testcontainers_modules::testcontainers::TestcontainersError),
    #[error("{command} exited with {code}")]
    Command { command: String, code: i64 },
}
