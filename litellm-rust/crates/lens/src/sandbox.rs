use crate::{Error, evidence::Workspace, wire};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    process::Stdio,
    sync::OnceLock,
    time::{Duration, Instant},
};
use tokio::{
    io::{AsyncRead, AsyncReadExt},
    process::Command,
    sync::Semaphore,
};

const READY: &[u8] = b"\x1eLENS_PYTHON_READY\x1e\n";
const BOOTSTRAP: &str = r#"
import resource
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
resource.setrlimit(resource.RLIMIT_AS, (536870912, 536870912))
resource.setrlimit(resource.RLIMIT_FSIZE, (16777216, 16777216))
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
import json, sys
sys.stderr.write("\x1eLENS_PYTHON_READY\x1e\n")
request = json.load(sys.stdin)
exec(compile(request["code"], "<lens-python>", "exec"), {"__name__": "__main__", "data": request["data"]})
"#;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Runtime {
    executable: PathBuf,
    directories: Vec<PathBuf>,
    read: Vec<PathBuf>,
    execute: Vec<PathBuf>,
}

fn command(directory: &Path, runtime_dir: &Path) -> Result<Command, Error> {
    if !cfg!(target_os = "linux") {
        return Err(Error::PythonUnsupportedPlatform);
    }
    let runtime: Runtime =
        serde_json::from_slice(&std::fs::read(runtime_dir.join("python-runtime.json"))?)?;
    let policy = runtime_dir.join("python.seccomp");
    if !policy.is_file() {
        return Err(Error::PythonPolicyMissing);
    }
    let mut command = Command::new("/usr/bin/setpriv");
    command.args(["--no-new-privs", "--landlock-access", "fs:execute,write-file,read-file,read-dir,remove-dir,remove-file,make-char,make-dir,make-reg,make-sock,make-fifo,make-block,make-sym,refer,truncate"]);
    for path in runtime.read {
        let access = if path.is_dir() {
            "read-file,read-dir"
        } else {
            "read-file"
        };
        command.args([
            "--landlock-rule",
            &format!("path-beneath:{access}:{}", path.display()),
        ]);
    }
    for path in runtime.execute {
        command.args([
            "--landlock-rule",
            &format!("path-beneath:read-file,execute:{}", path.display()),
        ]);
    }
    for path in runtime.directories {
        command.args([
            "--landlock-rule",
            &format!("path-beneath:read-dir:{}", path.display()),
        ]);
    }
    command.args(["--landlock-rule", &format!("path-beneath:read-file,read-dir,write-file,remove-file,remove-dir,make-dir,make-reg,make-sym,refer,truncate:{}", directory.display()), "--seccomp-filter"])
        .arg(policy).arg(runtime.executable).args(["-I", "-S", "-B", "-X", "utf8", "-u", "-c", BOOTSTRAP]);
    command
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("LANG", "C.UTF-8")
        .env("TMPDIR", directory)
        .current_dir(directory)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    Ok(command)
}

async fn output(mut pipe: impl AsyncRead + Unpin) -> Result<Vec<u8>, Error> {
    let mut output = Vec::new();
    let mut buffer = [0; 65536];
    loop {
        let count = pipe.read(&mut buffer).await?;
        if count == 0 {
            return Ok(output);
        }
        if output.len() + count > 4 * 1024 * 1024 {
            return Err(Error::PythonOutputTooLarge);
        }
        output.extend_from_slice(&buffer[..count]);
    }
}

#[cfg(target_os = "linux")]
fn scratch_usage(directory: &Path, pid: u32) -> Result<(), Error> {
    use std::{
        collections::BTreeSet,
        os::{
            fd::AsRawFd,
            unix::fs::{MetadataExt, OpenOptionsExt},
        },
    };
    let mut seen = BTreeSet::new();
    let mut bytes = 0;
    let mut entries = 0;
    let open_directory = |path: &Path| {
        std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW)
            .open(path)
    };
    let mut directories = vec![(open_directory(directory)?, 0)];
    let mut record = |metadata: std::fs::Metadata| -> Result<(), Error> {
        entries += 1;
        if seen.insert((metadata.dev(), metadata.ino())) {
            bytes += metadata.len().max(metadata.blocks().saturating_mul(512));
        }
        if entries > 2048 || bytes > 64 * 1024 * 1024 {
            return Err(Error::PythonScratchTooLarge);
        }
        Ok(())
    };
    while let Some((descriptor, depth)) = directories.pop() {
        if depth > 128 {
            return Err(Error::PythonScratchTooDeep);
        }
        for entry in std::fs::read_dir(format!("/proc/self/fd/{}", descriptor.as_raw_fd()))? {
            let entry = entry?;
            match std::fs::symlink_metadata(entry.path()) {
                Ok(metadata) => {
                    if metadata.is_dir() {
                        match open_directory(&entry.path()) {
                            Ok(child) => directories.push((child, depth + 1)),
                            Err(error)
                                if matches!(
                                    error.raw_os_error(),
                                    Some(libc::ENOENT | libc::ELOOP | libc::ENOTDIR)
                                ) => {}
                            Err(error) => return Err(error.into()),
                        }
                    }
                    record(metadata)?;
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
        }
    }
    match std::fs::read_dir(format!("/proc/{pid}/fd")) {
        Ok(descriptors) => {
            for descriptor in descriptors {
                let path = descriptor?.path();
                match std::fs::read_link(&path) {
                    Ok(target) if target.starts_with(directory) => match std::fs::metadata(path) {
                        Ok(metadata) => record(metadata)?,
                        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                        Err(error) => return Err(error.into()),
                    },
                    Ok(_) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => return Err(error.into()),
                }
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    }
    let mappings = match std::fs::read_to_string(format!("/proc/{pid}/maps")) {
        Ok(mappings) => mappings,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    for line in mappings.lines() {
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields.len() < 6 || fields[4] == "0" || !Path::new(fields[5]).starts_with(directory) {
            continue;
        }
        let (major, minor) = fields[3].split_once(':').ok_or(Error::InvalidRequest)?;
        let device = libc::makedev(
            u32::from_str_radix(major, 16).map_err(|_| Error::InvalidRequest)?,
            u32::from_str_radix(minor, 16).map_err(|_| Error::InvalidRequest)?,
        );
        let inode = fields[4]
            .parse::<u64>()
            .map_err(|_| Error::InvalidRequest)?;
        if seen.insert((device, inode)) {
            bytes += 16 * 1024 * 1024;
            entries += 1;
        }
        if entries > 2048 || bytes > 64 * 1024 * 1024 {
            return Err(Error::PythonScratchTooLarge);
        }
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn scratch_usage(_directory: &Path, _pid: u32) -> Result<(), Error> {
    Err(Error::PythonUnsupportedPlatform)
}

async fn monitor(directory: PathBuf, pid: u32) -> Result<(), Error> {
    loop {
        let path = directory.clone();
        tokio::task::spawn_blocking(move || scratch_usage(&path, pid))
            .await
            .map_err(|_| Error::Unavailable)??;
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

pub async fn execute(workspace: &Workspace, request: &wire::PythonRequest) -> Result<Value, Error> {
    static SLOTS: OnceLock<Semaphore> = OnceLock::new();
    let permit = SLOTS
        .get_or_init(|| Semaphore::new(2))
        .acquire()
        .await
        .map_err(|_| Error::Unavailable)?;
    let input = tempfile::NamedTempFile::new()?;
    let mut file = tokio::fs::File::create(input.path()).await?;
    use tokio::io::AsyncWriteExt;
    file.write_all(b"{\"code\":").await?;
    file.write_all(&serde_json::to_vec(&request.code)?).await?;
    file.write_all(b",\"data\":").await?;
    workspace.python_input(request, &mut file).await?;
    file.write_all(b"}").await?;
    file.flush().await?;
    drop(file);
    let directory = tempfile::Builder::new().prefix("lens-python-").tempdir()?;
    let runtime_dir = std::env::var_os("LENS_PYTHON_RUNTIME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/app/lens"));
    let (_cancel, cancelled) = tokio::sync::oneshot::channel();
    tokio::spawn(supervise(input, directory, runtime_dir, permit, cancelled))
        .await
        .map_err(|_| Error::Unavailable)?
}

async fn supervise(
    input: tempfile::NamedTempFile,
    directory: tempfile::TempDir,
    runtime_dir: PathBuf,
    _permit: tokio::sync::SemaphorePermit<'static>,
    mut cancelled: tokio::sync::oneshot::Receiver<()>,
) -> Result<Value, Error> {
    let directory_path = directory.path().canonicalize()?;
    let started = Instant::now();
    let mut child = command(&directory_path, &runtime_dir)?.spawn()?;
    let pid = child.id().ok_or(Error::Unavailable)?;
    let mut stdin = child.stdin.take().ok_or(Error::Unavailable)?;
    let stdout = child.stdout.take().ok_or(Error::Unavailable)?;
    let stderr = child.stderr.take().ok_or(Error::Unavailable)?;
    let computation = async {
        let feed = async {
            let mut file = tokio::fs::File::open(input.path()).await?;
            match tokio::io::copy(&mut file, &mut stdin).await {
                Ok(_) => {}
                Err(error) if error.kind() == std::io::ErrorKind::BrokenPipe => {}
                Err(error) => return Err(Error::Io(error)),
            }
            drop(stdin);
            Ok::<_, Error>(())
        };
        let wait = async { child.wait().await.map_err(Error::from) };
        tokio::try_join!(feed, output(stdout), output(stderr), wait)
    };
    let result = tokio::select! {
        result = tokio::time::timeout(Duration::from_secs(60), computation) => result.map_err(|_| Error::PythonTimedOut).and_then(|r| r),
        result = monitor(directory_path.clone(), pid) => Err(result.err().unwrap_or(Error::Unavailable)),
        _ = &mut cancelled => Err(Error::PythonCancelled),
    };
    let result = result.and_then(|output| {
        scratch_usage(&directory_path, pid)?;
        Ok(output)
    });
    let (stdout, stderr, exit_code, error) = match result {
        Ok(((), stdout, stderr, status)) => {
            let ready = stderr.starts_with(READY);
            let stderr = if ready {
                stderr[READY.len()..].to_vec()
            } else {
                stderr
            };
            let error = if !ready {
                "Python confinement failed before execution. Check worker image and kernel support."
            } else if !status.success() {
                "Python computation failed or reached a resource limit. Inspect stderr."
            } else {
                ""
            };
            (stdout, stderr, status.code(), error.to_owned())
        }
        Err(error) => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            (Vec::new(), Vec::new(), None, error.to_string())
        }
    };
    Ok(
        json!({"stdout": String::from_utf8_lossy(&stdout), "stderr": String::from_utf8_lossy(&stderr), "exit_code": exit_code, "elapsed_seconds": started.elapsed().as_secs_f64(), "output_complete": error.is_empty(), "error": error}),
    )
}
