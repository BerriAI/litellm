use std::{
    env, fs,
    io::{self, Write},
    process::ExitCode,
};

use litellm_db::{
    CATALOG,
    codegen::{catalog_module, schema_document},
};

fn main() -> ExitCode {
    let Some(catalog_path) = env::args().nth(1) else {
        eprintln!("usage: gen-python <catalog.py>; the JSON Schema goes to stdout");
        return ExitCode::FAILURE;
    };
    let queries: Vec<_> = CATALOG.iter().flat_map(|domain| domain.iter()).collect();
    let catalog = match catalog_module(&queries) {
        Ok(catalog) => catalog,
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(error) = fs::write(&catalog_path, catalog) {
        eprintln!("writing {catalog_path}: {error}");
        return ExitCode::FAILURE;
    }
    let schema = schema_document(&queries);
    match writeln!(io::stdout().lock(), "{schema:#}") {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("writing the JSON Schema: {error}");
            ExitCode::FAILURE
        }
    }
}
