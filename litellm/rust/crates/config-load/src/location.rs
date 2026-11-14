use std::{
    fmt,
    path::{Component, Path, PathBuf},
};

use crate::{
    Error,
    host::{Host, Warning},
};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Location {
    File(PathBuf),
    Object(String),
}

impl fmt::Display for Location {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::File(path) => write!(formatter, "{}", path.display()),
            Self::Object(key) => formatter.write_str(key),
        }
    }
}

impl Location {
    pub(crate) fn normalized(self) -> Self {
        match self {
            Self::File(path) => Self::File(normalized(&path)),
            object @ Self::Object(_) => object,
        }
    }

    pub(crate) async fn included<E>(
        &self,
        entry: &str,
        root: &Self,
        host: &Host<E>,
    ) -> Result<Self, Error<E>> {
        match (self, root) {
            (Self::Object(declared_in), _) => Ok(Self::Object(object_beside(declared_in, entry))),
            (Self::File(declared_in), Self::Object(_)) => {
                Ok(Self::File(file_beside(declared_in, entry)))
            }
            (Self::File(declared_in), Self::File(root)) => {
                file_beside_or_root(entry, declared_in, root, host)
                    .await
                    .map(Self::File)
            }
        }
    }
}

async fn file_beside_or_root<E>(
    entry: &str,
    declared_in: &Path,
    root: &Path,
    host: &Host<E>,
) -> Result<PathBuf, Error<E>> {
    let beside_declaring = file_beside(declared_in, entry);
    let beside_root = file_beside(root, entry);
    if beside_root == beside_declaring || !host.exists(&beside_root).await? {
        return Ok(beside_declaring);
    }
    if !host.exists(&beside_declaring).await? {
        host.warn(Warning::ReadBesideRoot {
            entry: entry.to_owned(),
            declared_in: declared_in.to_owned(),
            read: beside_root.clone(),
        })
        .await?;
        return Ok(beside_root);
    }
    host.warn(Warning::MatchesTwoFiles {
        entry: entry.to_owned(),
        declared_in: declared_in.to_owned(),
        read: beside_declaring.clone(),
        skipped: beside_root,
    })
    .await?;
    Ok(beside_declaring)
}

fn file_beside(config: &Path, entry: &str) -> PathBuf {
    normalized(&config.parent().unwrap_or(config).join(entry))
}

fn normalized(path: &Path) -> PathBuf {
    path.components()
        .fold(PathBuf::new(), |mut clean, component| {
            match component {
                Component::CurDir => {}
                Component::ParentDir => match clean.components().next_back() {
                    Some(Component::Normal(_)) => {
                        clean.pop();
                    }
                    Some(Component::RootDir | Component::Prefix(_)) => {}
                    Some(Component::ParentDir | Component::CurDir) | None => clean.push(".."),
                },
                component => clean.push(component),
            }
            clean
        })
}

fn object_beside(config_key: &str, entry: &str) -> String {
    if entry.starts_with('/') {
        return posix_normpath(entry).trim_start_matches('/').to_owned();
    }
    posix_normpath(&posix_join(posix_dirname(config_key), entry))
}

fn posix_dirname(path: &str) -> &str {
    let head = path.rfind('/').map_or("", |slash| &path[..=slash]);
    match head.trim_end_matches('/') {
        "" => head,
        trimmed => trimmed,
    }
}

fn posix_join(directory: &str, relative: &str) -> String {
    if directory.is_empty() || directory.ends_with('/') {
        return format!("{directory}{relative}");
    }
    format!("{directory}/{relative}")
}

fn posix_normpath(path: &str) -> String {
    let root = match path.len() - path.trim_start_matches('/').len() {
        0 => "",
        2 => "//",
        _ => "/",
    };
    let parts = path.split('/').fold(Vec::<&str>::new(), |mut parts, part| {
        match part {
            "" | "." => {}
            ".." if parts.last().is_some_and(|last| *last != "..") => {
                parts.pop();
            }
            ".." if parts.is_empty() && !root.is_empty() => {}
            part => parts.push(part),
        }
        parts
    });
    match format!("{root}{}", parts.join("/")) {
        normalized if normalized.is_empty() => ".".to_owned(),
        normalized => normalized,
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::empty("", ".")]
    #[case::dot(".", ".")]
    #[case::plain("a/b", "a/b")]
    #[case::doubled_separator("a//b", "a/b")]
    #[case::inner_dot("a/./b", "a/b")]
    #[case::leading_dot("./a", "a")]
    #[case::trailing_separator("a/", "a")]
    #[case::parent("a/b/..", "a")]
    #[case::parent_past_the_start("a/../..", "..")]
    #[case::leading_parent("../a", "../a")]
    #[case::leading_parents("../../a/..", "../..")]
    #[case::parents_past_the_start("a/b/../../../c", "../c")]
    #[case::root("/", "/")]
    #[case::two_slashes("//", "//")]
    #[case::three_slashes("///", "/")]
    #[case::parent_of_root("/..", "/")]
    #[case::parent_of_root_then_name("/../a", "/a")]
    #[case::two_slashes_then_name("//a", "//a")]
    #[case::three_slashes_then_name("///a", "/a")]
    fn normpath_matches_posixpath(#[case] path: &str, #[case] normalized: &str) {
        assert_eq!(posix_normpath(path), normalized);
    }

    #[rstest]
    #[case::beside_the_config("configs/config.yaml", "models.yaml", "configs/models.yaml")]
    #[case::leading_slash_is_the_bucket_root(
        "configs/config.yaml",
        "/shared/models.yaml",
        "shared/models.yaml"
    )]
    #[case::doubled_slashes_from_the_bucket_root(
        "configs/config.yaml",
        "//shared//models.yaml",
        "shared/models.yaml"
    )]
    #[case::dot_dot_leaves_the_prefix(
        "configs/config.yaml",
        "../shared/models.yaml",
        "shared/models.yaml"
    )]
    #[case::dot_names_the_same_object(
        "configs/config.yaml",
        "./shared.yaml",
        "configs/shared.yaml"
    )]
    #[case::config_at_the_bucket_root("config.yaml", "models.yaml", "models.yaml")]
    #[case::dot_dot_above_the_bucket_root("config.yaml", "../models.yaml", "../models.yaml")]
    #[case::dot_dot_past_a_deep_prefix("a/b/config.yaml", "../../../x.yaml", "../x.yaml")]
    #[case::config_key_with_a_leading_slash("/config.yaml", "models.yaml", "/models.yaml")]
    #[case::config_key_with_a_doubled_separator("configs//config.yaml", "m.yaml", "configs/m.yaml")]
    #[case::empty_entry("configs/config.yaml", "", "configs")]
    fn an_object_key_resolves_against_the_declaring_objects_prefix(
        #[case] config_key: &str,
        #[case] entry: &str,
        #[case] key: &str,
    ) {
        assert_eq!(object_beside(config_key, entry), key);
    }

    #[cfg(unix)]
    #[rstest]
    #[case::beside_the_config(
        "/etc/litellm/config.yaml",
        "models.yaml",
        "/etc/litellm/models.yaml"
    )]
    #[case::in_a_subdirectory(
        "/etc/litellm/config.yaml",
        "shared/models.yaml",
        "/etc/litellm/shared/models.yaml"
    )]
    #[case::dot(
        "/etc/litellm/config.yaml",
        "./models.yaml",
        "/etc/litellm/models.yaml"
    )]
    #[case::dot_dot("/etc/litellm/config.yaml", "../models.yaml", "/etc/models.yaml")]
    #[case::dot_dot_past_the_root("/config.yaml", "../../models.yaml", "/models.yaml")]
    #[case::absolute_entry("/etc/litellm/config.yaml", "/srv/models.yaml", "/srv/models.yaml")]
    fn a_file_resolves_next_to_its_declaring_config(
        #[case] config: &str,
        #[case] entry: &str,
        #[case] path: &str,
    ) {
        assert_eq!(file_beside(Path::new(config), entry), Path::new(path));
    }
}
