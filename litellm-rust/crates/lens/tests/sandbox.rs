#![cfg(target_os = "linux")]

use litellm_lens::{
    config::http_client,
    control::{Control, JobClient},
    evidence::Workspace,
    sandbox, wire,
};
use rstest::{fixture, rstest};
use serde_json::{Value, json};
use std::{path::Path, time::Duration};

#[fixture]
fn workspace() -> Workspace {
    Workspace::new(
        Vec::new(),
        JobClient::new(
            Control::new(
                http_client().unwrap(),
                "http://127.0.0.1:1".parse().unwrap(),
                "unused".into(),
            ),
            "test",
            "test",
            1,
        )
        .unwrap(),
    )
}

fn request(code: &str) -> wire::PythonRequest {
    serde_json::from_value(json!({"action": "python", "code": code})).unwrap()
}

fn succeeded(reply: &Value) {
    assert_eq!(reply["exit_code"], 0, "{reply}");
    assert_eq!(reply["error"], "", "{reply}");
    assert_eq!(reply["output_complete"], true, "{reply}");
}

#[rstest]
#[tokio::test]
#[ignore = "requires the native Lens Linux image"]
async fn confined_python_can_analyze_evidence_with_the_standard_library(workspace: Workspace) {
    let reply = sandbox::execute(
        &workspace,
        &request(
            r#"
import collections, json, math, sqlite3, tempfile
assert data['sessions'] == []
with tempfile.TemporaryFile() as f:
    f.write(b'analysis'); f.seek(0); assert f.read() == b'analysis'
c = sqlite3.connect('evidence.db')
c.execute('create table evidence(value text)')
c.execute("insert into evidence values ('failed')")
assert c.execute('select value from evidence').fetchone()[0] == 'failed'
assert math.sqrt(81) == 9
print(json.dumps(dict(collections.Counter(['failed', 'failed', 'success'])), sort_keys=True))
"#,
        ),
    )
    .await
    .unwrap();
    succeeded(&reply);
    assert_eq!(reply["stdout"], "{\"failed\": 2, \"success\": 1}\n");
}

#[rstest]
#[tokio::test]
#[ignore = "requires the native Lens Linux image"]
async fn code_cannot_read_worker_files_escape_scratch_or_open_network(workspace: Workspace) {
    let sentinel = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(sentinel.path(), "worker private data").unwrap();
    let code = format!(
        r#"
import ctypes, errno, os, socket, sys
assert sys.flags.isolated and sys.flags.no_site
assert not any(k.startswith(('LENS_', 'LITELLM_', 'CLICKHOUSE_')) for k in os.environ)
def denied(action):
    try:
        action()
    except OSError as e:
        assert e.errno in (errno.EACCES, errno.EPERM, errno.EXDEV), e
        return
    raise AssertionError('escaped confinement')
secret = {sentinel:?}
for path in (secret, '/proc/self/environ', '/usr/local/bin/litellm-lens'):
    denied(lambda: open(path).read())
denied(lambda: open(secret, 'w'))
denied(lambda: os.chmod(secret, 0o777))
denied(lambda: os.utime(secret))
os.symlink(secret, 'escape')
denied(lambda: open('escape').read())
denied(lambda: open('escape', 'w'))
denied(lambda: os.link(secret, 'hardlink'))
denied(lambda: os.rename(secret, 'renamed'))
for family in (socket.AF_INET, socket.AF_INET6, socket.AF_UNIX):
    denied(lambda: socket.socket(family, socket.SOCK_STREAM))
denied(socket.socketpair)
denied(os.fork)
denied(lambda: os.kill(os.getppid(), 0))
denied(lambda: os.execv('/bin/sh', ['sh', '-c', 'exit 0']))
lib = ctypes.CDLL(None, use_errno=True)
for name, args in (('ptrace', (16, os.getppid(), 0, 0)), ('process_vm_readv', (os.getppid(), 0, 0, 0, 0, 0)), ('shmget', (0, 4096, 0o1600)), ('syscall', (425, 0, 0))):
    ctypes.set_errno(0)
    assert getattr(lib, name)(*args) == -1, name
    assert ctypes.get_errno() == errno.EPERM, name
print('confined')
"#,
        sentinel = sentinel.path().display().to_string()
    );
    let reply = sandbox::execute(&workspace, &request(&code)).await.unwrap();
    succeeded(&reply);
    assert_eq!(reply["stdout"], "confined\n");
    assert_eq!(
        std::fs::read_to_string(sentinel.path()).unwrap(),
        "worker private data"
    );
}

#[rstest]
#[case::memory("x = bytearray(1024 * 1024 * 1024)", "MemoryError")]
#[case::file(
    "open('large', 'wb').write(b'x' * (17 * 1024 * 1024))",
    "File too large"
)]
#[case::output("print('x' * (5 * 1024 * 1024))", "output exceeded")]
#[case::scratch(
    "import pathlib\nfor i in range(3000): pathlib.Path(str(i)).touch()",
    "scratch storage"
)]
#[case::hidden(
    "import ctypes,sys,time\nprint('before hiding', file=sys.stderr)\nassert ctypes.CDLL(None).prctl(4,0,0,0,0) == 0\ntime.sleep(2)",
    "resource monitoring failed"
)]
#[tokio::test]
#[ignore = "requires the native Lens Linux image"]
async fn resource_limits_fail_the_tool_and_clean_up(
    workspace: Workspace,
    #[case] code: &str,
    #[case] error: &str,
) {
    let reply = sandbox::execute(&workspace, &request(code)).await.unwrap();
    assert_eq!(reply["output_complete"], false, "{reply}");
    assert!(reply.to_string().contains(error), "{reply}");
    if error == "resource monitoring failed" {
        assert!(
            reply["stderr"].as_str().unwrap().contains("before hiding"),
            "{reply}"
        );
        assert!(reply["elapsed_seconds"].as_f64().unwrap() < 2.0, "{reply}");
    }
    assert!(!std::fs::read_dir("/tmp").unwrap().any(|entry| {
        entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with("lens-python-")
    }));
}

#[rstest]
#[case::success("print('completed')", 0, "")]
#[case::memory("x = bytearray(1024 * 1024 * 1024)", 1, "MemoryError")]
#[tokio::test]
#[ignore = "requires the native Lens Linux image"]
async fn rapid_process_exits_preserve_their_output(
    workspace: Workspace,
    #[case] code: &str,
    #[case] exit_code: i32,
    #[case] stderr: &str,
) {
    for attempt in 0..32 {
        let reply = sandbox::execute(&workspace, &request(code)).await.unwrap();
        assert_eq!(reply["exit_code"], exit_code, "attempt {attempt}: {reply}");
        assert_eq!(
            reply["output_complete"],
            exit_code == 0,
            "attempt {attempt}: {reply}"
        );
        assert!(
            reply["stderr"].as_str().unwrap().contains(stderr),
            "attempt {attempt}: {reply}"
        );
        if exit_code == 0 {
            assert_eq!(reply["stdout"], "completed\n", "attempt {attempt}: {reply}");
        }
    }
}

#[rstest]
#[tokio::test]
#[ignore = "requires the native Lens Linux image"]
async fn cancellation_kills_and_reaps_python_before_releasing_its_slot(workspace: Workspace) {
    let task = tokio::spawn(async move {
        sandbox::execute(
            &workspace,
            &request("import os,time\nopen('ready','w').write(str(os.getpid()))\ntime.sleep(60)"),
        )
        .await
    });
    let (directory, pid) = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            for entry in std::fs::read_dir("/tmp").unwrap() {
                let directory = entry.unwrap().path();
                if !directory
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with("lens-python-")
                {
                    continue;
                }
                if let Ok(pid) = std::fs::read_to_string(directory.join("ready"))
                    && let Ok(pid) = pid.parse::<u32>()
                {
                    return (directory, pid);
                }
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    tokio::time::timeout(Duration::from_secs(5), async {
        while directory.exists() || Path::new(&format!("/proc/{pid}")).exists() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
}
