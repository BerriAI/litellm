use std::path::PathBuf;

use litellm_config_load::load;
use rstest::rstest;

#[rstest]
fn loads_every_valid_fixture(#[files("tests/fixtures/valid/*.yaml")] path: PathBuf) {
    if let Err(error) = load(&path) {
        panic!("{}: {error:?}", path.display());
    }
}

#[rstest]
fn rejects_every_invalid_fixture(#[files("tests/fixtures/invalid/*.yaml")] path: PathBuf) {
    assert!(load(&path).is_err(), "{} loads", path.display());
}
