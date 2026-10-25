use std::fs;

use litellm_config_load::{Error, load};
use rstest::{fixture, rstest};
use tempfile::TempDir;

#[fixture]
fn directory() -> TempDir {
    tempfile::tempdir().unwrap()
}

#[rstest]
fn loads_model_list_from_file(directory: TempDir) {
    let path = directory.path().join("config.yaml");
    fs::write(
        &path,
        r#"
model_list:
  - model_name: assistant
    litellm_params:
      model: anthropic/test-model
  - model_name: local
    litellm_params:
      model: test-model
"#,
    )
    .unwrap();

    let config = load(path).unwrap();
    assert_eq!(
        config
            .model_list
            .iter()
            .map(|model| model.model_name.as_str())
            .collect::<Vec<_>>(),
        ["assistant", "local"]
    );
}

#[rstest]
#[case::missing_file(None, |error: &Error| matches!(error, Error::Read(error) if error.kind() == std::io::ErrorKind::NotFound))]
#[case::malformed_yaml(Some("model_list: ["), |error: &Error| matches!(error, Error::Parse(_)))]
fn distinguishes_read_errors_from_parse_errors(
    directory: TempDir,
    #[case] contents: Option<&str>,
    #[case] expected: fn(&Error) -> bool,
) {
    let path = directory.path().join("config.yaml");
    if let Some(contents) = contents {
        fs::write(&path, contents).unwrap();
    }

    let error = load(&path).unwrap_err();
    let Error::Document { source, .. } = &error else {
        panic!("{error:?}");
    };
    assert!(expected(source), "{error:?}");
}

#[rstest]
fn resolves_nested_includes_once_in_breadth_first_order(directory: TempDir) {
    let root = directory.path();
    fs::create_dir(root.join("nested")).unwrap();
    fs::write(
        root.join("config.yaml"),
        r#"
include: [nested/first.yaml, second.yaml]
model_list: [{model_name: root, litellm_params: {model: root}}]
"#,
    )
    .unwrap();
    fs::write(
        root.join("nested/first.yaml"),
        r#"
include: [third.yaml]
model_list: [{model_name: first, litellm_params: {model: first}}]
"#,
    )
    .unwrap();
    fs::write(
        root.join("second.yaml"),
        r#"
general_settings: {master_key: second}
model_list: [{model_name: second, litellm_params: {model: second}}]
"#,
    )
    .unwrap();
    fs::write(
        root.join("nested/third.yaml"),
        r#"
include: [../config.yaml]
general_settings: {master_key: third}
model_list: [{model_name: third, litellm_params: {model: third}}]
"#,
    )
    .unwrap();

    let config = load(root.join("config.yaml")).unwrap();
    assert_eq!(
        config
            .model_list
            .iter()
            .map(|model| model.model_name.as_str())
            .collect::<Vec<_>>(),
        ["root", "first", "second", "third"]
    );
    assert_eq!(
        config.general_settings.master_key.unwrap().expose(),
        "third"
    );
}
