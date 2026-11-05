use std::{env, fs, path::PathBuf, process::ExitCode};

use litellm_model_catalog::Catalog;

fn main() -> ExitCode {
    let Some(path) = env::args_os().nth(1).map(PathBuf::from) else {
        eprintln!("usage: check-registry <model_prices_and_context_window.json>");
        return ExitCode::from(2);
    };
    let body = match fs::read(&path) {
        Ok(body) => body,
        Err(error) => {
            eprintln!("{}: {error}", path.display());
            return ExitCode::FAILURE;
        }
    };
    let problems = Catalog::problems(&body);
    for problem in &problems {
        eprintln!("{}", litellm_error::Report::new(problem));
    }
    if problems.is_empty() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
