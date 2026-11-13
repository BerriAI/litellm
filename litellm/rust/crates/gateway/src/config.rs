use std::path::Path;

use litellm_config::{Config, Value};
use litellm_config_load::{HostCall, Includes, Location, Step};

use crate::{ConfigSourceError, Error};

pub fn load_config(path: impl AsRef<Path>) -> Result<Config, Error> {
    let path = path.as_ref();
    let root = std::path::absolute(path).map_err(|source| ConfigSourceError::Read {
        path: path.to_owned(),
        source,
    })?;
    let mut includes = Includes::new(Location::File(root.clone()), read(&root)?);
    loop {
        match includes.step() {
            Step::Call(HostCall::Exists(path, reply)) => reply.send(path.exists()),
            Step::Call(HostCall::Read(Location::File(path), reply)) => reply.send(read(&path)),
            Step::Call(HostCall::Read(Location::Object(key), reply)) => {
                reply.send(Err(ConfigSourceError::Object { key }));
            }
            Step::Call(HostCall::Warn(warning, reply)) => {
                tracing::warn!(%warning, "config include");
                reply.send(());
            }
            Step::Done(merged) => return Ok(Config::from_value(Value::Mapping(merged?))?),
        }
    }
}

fn read(path: &Path) -> Result<Value, ConfigSourceError> {
    let text = std::fs::read_to_string(path).map_err(|source| ConfigSourceError::Read {
        path: path.to_owned(),
        source,
    })?;
    serde_yaml_ng::from_str(&text).map_err(|source| ConfigSourceError::Parse {
        path: path.to_owned(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};
    use tempfile::TempDir;

    use super::*;

    #[fixture]
    fn directory() -> TempDir {
        tempfile::tempdir().unwrap()
    }

    fn model(name: &str) -> String {
        format!("model_list: [{{model_name: {name}, litellm_params: {{model: {name}}}}}]\n")
    }

    #[rstest]
    fn nested_includes_on_disk_load_into_one_typed_config(directory: TempDir) {
        let root = directory.path();
        std::fs::create_dir(root.join("nested")).unwrap();
        std::fs::write(
            root.join("config.yaml"),
            format!(
                "include: [nested/first.yaml, second.yaml]\n{}",
                model("root")
            ),
        )
        .unwrap();
        std::fs::write(
            root.join("nested/first.yaml"),
            format!("include: [third.yaml]\n{}", model("first")),
        )
        .unwrap();
        std::fs::write(
            root.join("second.yaml"),
            format!(
                "general_settings: {{master_key: second}}\n{}",
                model("second")
            ),
        )
        .unwrap();
        std::fs::write(
            root.join("nested/third.yaml"),
            format!(
                "include: [../config.yaml]\ngeneral_settings: {{master_key: third}}\n{}",
                model("third")
            ),
        )
        .unwrap();

        let config = load_config(root.join("config.yaml")).unwrap();

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
        assert!(config.include.is_empty());
    }

    #[rstest]
    fn a_missing_config_is_a_read_error_naming_the_file(directory: TempDir) {
        let error = load_config(directory.path().join("missing.yaml")).unwrap_err();

        assert!(
            litellm_error::Report::new(&error)
                .to_string()
                .contains("missing.yaml")
        );
        assert!(matches!(
            error,
            Error::ConfigSource(ConfigSourceError::Read { source, .. })
                if source.kind() == std::io::ErrorKind::NotFound
        ));
    }

    #[rstest]
    fn a_missing_include_fails_the_load_naming_the_file(directory: TempDir) {
        std::fs::write(
            directory.path().join("config.yaml"),
            "include: [models.yaml]\n",
        )
        .unwrap();

        let error = load_config(directory.path().join("config.yaml")).unwrap_err();

        assert!(
            litellm_error::Report::new(&error)
                .to_string()
                .contains("models.yaml")
        );
        assert!(matches!(
            error,
            Error::ConfigLoad(litellm_config_load::Error::Host(
                ConfigSourceError::Read { .. }
            ))
        ));
    }
}
