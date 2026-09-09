//! Parity tests between the Python and Rust.

use libtest_mimic::{Arguments, Completion, Failed, Trial, run};
use serde::Deserialize;
use serde_json::{Value, json};
use similar::{ChangeTag, TextDiff};
use std::env;
use std::ffi::OsString;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio, exit};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const REAL_ENV: &str = "LITELLM_RUN_REAL";
const REAL_FLAG: &str = "--real";
const ACCEPT_ENV: &str = "LITELLM_PARITY_ACCEPT";
const BIN_ENV: &str = "LITELLM_BIN";
const PYTHON_ENV: &str = "LITELLM_PYTHON";
const RUST_HEADER: &str = "x-litellm-rust";
const PORT_PLACEHOLDER: &str = "$PORT";
const MASTER_KEY: &str = "sk-1234";
const HEALTH_PATH: &str = "/health/liveliness";
const CHAT_PATH: &str = "/chat/completions";
const INPUTS_DIR: &str = "inputs";
const HOST: &str = "127.0.0.1";
const HEALTH_TIMEOUT: Duration = Duration::from_secs(90);
const CHAT_TIMEOUT: Duration = Duration::from_secs(180);
const POLL_INTERVAL: Duration = Duration::from_millis(200);
const MOCK_POLL: Duration = Duration::from_millis(5);
const REDACT_KEYS: &[&str] = &["id", "created", "system_fingerprint"];

#[derive(Clone, Copy)]
struct Env {
    real_enabled: bool,
    litellm_ok: bool,
    extension_ok: bool,
}

#[derive(Clone, Deserialize)]
struct TestCase {
    model: String,
    temperature: Option<f64>,
    seed: Option<u64>,
    max_tokens: Option<u32>,
    messages: Vec<Message>,
    proxy: ProxySpec,
    mock: Option<MockSpec>,
    real: Option<RealSpec>,
}

#[derive(Clone, Deserialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Clone, Deserialize)]
struct ProxySpec {
    config: String,
}

#[derive(Clone, Deserialize)]
struct MockSpec {
    responses: Vec<MockResponse>,
}

#[derive(Clone, Deserialize)]
struct MockResponse {
    status: Option<u16>,
    body: String,
}

#[derive(Clone, Deserialize)]
struct RealSpec {
    health_path: Option<String>,
}

#[derive(Clone)]
struct ReceivedRequest {
    body: String,
}

struct RequestParity {
    parity: bool,
    diff: String,
}

struct MockState {
    responses: Vec<MockResponse>,
    next: AtomicUsize,
    stop: AtomicBool,
    requests: Mutex<Vec<ReceivedRequest>>,
}

impl MockState {
    fn new(responses: Vec<MockResponse>) -> Self {
        Self {
            responses,
            next: AtomicUsize::new(0),
            stop: AtomicBool::new(false),
            requests: Mutex::new(Vec::new()),
        }
    }
}

struct ProxyProc {
    child: Child,
    label: String,
    out_log: PathBuf,
    err_log: PathBuf,
}

impl Drop for ProxyProc {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

struct Cleanup {
    procs: Vec<ProxyProc>,
    mock_state: Option<Arc<MockState>>,
    mock_join: Option<thread::JoinHandle<()>>,
}

impl Cleanup {
    fn kill_now(&mut self) {
        for p in &mut self.procs {
            let _ = p.child.kill();
            let _ = p.child.wait();
        }
        if let Some(state) = &self.mock_state {
            state.stop.store(true, Ordering::SeqCst);
        }
        if let Some(join) = self.mock_join.take() {
            let _ = join.join();
        }
    }
}

impl Drop for Cleanup {
    fn drop(&mut self) {
        self.kill_now();
    }
}

struct ChatResult {
    status: u16,
    served_by_rust: bool,
    body: String,
    json: Value,
}

fn main() {
    let (args, real_flag) = parse_cli_args();
    if let Err(e) = ensure_environment() {
        eprintln!("Parity tests setup failed: {e}");
        exit(1);
    }
    let env_state = Env {
        real_enabled: real_flag || real_env_set(),
        litellm_ok: litellm_available(),
        extension_ok: native_extension_available(),
    };
    let loaded = load_cases();
    let mut trials: Vec<Trial> = loaded
        .cases
        .into_iter()
        .map(|(name, path, case)| {
            Trial::ignorable_test(name.clone(), move || {
                run_case(&name, &path, &case, &env_state)
            })
        })
        .collect();
    for (name, message) in loaded.invalid {
        trials.push(Trial::test(name, move || Err(Failed::from(message))));
    }
    if trials.is_empty() {
        trials.push(Trial::test("no fixtures found", || {
            Err(Failed::from(format!(
                "no fixtures found in {}; add a .toml under {INPUTS_DIR}/",
                inputs_dir().display()
            )))
        }));
    }
    run(&args, trials).exit();
}

fn parse_cli_args() -> (Arguments, bool) {
    let mut real_flag = false;
    let mut filtered: Vec<OsString> = Vec::new();
    let mut it = env::args_os();
    filtered.push(it.next().unwrap_or_default());
    for arg in it {
        if arg.to_str() == Some(REAL_FLAG) {
            real_flag = true;
        } else {
            filtered.push(arg);
        }
    }
    (Arguments::from_iter(filtered), real_flag)
}

fn real_env_set() -> bool {
    match env::var(REAL_ENV) {
        Ok(v) => !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("false"),
        Err(_) => false,
    }
}

fn run_case(
    name: &str,
    toml_path: &Path,
    case: &TestCase,
    env_state: &Env,
) -> Result<Completion, Failed> {
    let is_mock = case.mock.is_some();
    if !is_mock && !env_state.real_enabled {
        return Ok(Completion::ignored_with(format!(
            "Real mode disabled; pass {REAL_FLAG} or set {REAL_ENV}=1"
        )));
    }
    if !env_state.litellm_ok || !env_state.extension_ok {
        return Ok(Completion::ignored_with(
            "litellm CLI or native Rust extension not available after setup",
        ));
    }
    if !is_mock
        && let Some(base) = extract_api_base(&case.proxy.config)
        && let Some(health_path) = case.real.as_ref().and_then(|r| r.health_path.as_deref())
        && !upstream_reachable(&base, health_path)
    {
        return Ok(Completion::ignored_with(format!(
            "upstream unreachable at {base}"
        )));
    }

    let dir = tempfile::tempdir().map_err(Failed::from)?;

    let (mock_state, mock_join, mock_port) = match &case.mock {
        Some(mock) => {
            let listener = TcpListener::bind(format!("{HOST}:0")).map_err(Failed::from)?;
            let port = listener.local_addr().map_err(Failed::from)?.port();
            let state = Arc::new(MockState::new(mock.responses.clone()));
            let state_for_thread = state.clone();
            let join = thread::spawn(move || serve_mock(listener, state_for_thread));
            (Some(state), Some(join), Some(port))
        }
        None => (None, None, None),
    };

    let config_text = match mock_port {
        Some(p) => case.proxy.config.replace(PORT_PLACEHOLDER, &p.to_string()),
        None => case.proxy.config.clone(),
    };
    let config_path = dir.path().join("config.yml");
    fs::write(&config_path, &config_text).map_err(Failed::from)?;

    let py_port = free_port()?;
    let rust_port = free_port()?;
    let py = spawn_proxy(&config_path, py_port, false, dir.path(), "python")?;
    let rust = spawn_proxy(&config_path, rust_port, true, dir.path(), "rust")?;
    let mut cleanup = Cleanup {
        procs: vec![py, rust],
        mock_state,
        mock_join,
    };

    let deadline = Instant::now() + HEALTH_TIMEOUT;
    if let Err(e) = wait_healthy(py_port, deadline).and_then(|_| wait_healthy(rust_port, deadline))
    {
        cleanup.kill_now();
        print_proxy_logs(&cleanup.procs);
        return Err(e);
    }

    let req = build_request_body(case);
    let py_result = match send_chat(py_port, &req) {
        Ok(r) => r,
        Err(e) => {
            cleanup.kill_now();
            print_proxy_logs(&cleanup.procs);
            return Err(e);
        }
    };
    let rust_result = match send_chat(rust_port, &req) {
        Ok(r) => r,
        Err(e) => {
            cleanup.kill_now();
            print_proxy_logs(&cleanup.procs);
            return Err(e);
        }
    };

    cleanup.kill_now();
    let mock_log = cleanup
        .mock_state
        .as_ref()
        .map(|s| s.requests.lock().expect("mock log mutex").clone())
        .unwrap_or_default();

    if py_result.status != 200 {
        print_proxy_logs(&cleanup.procs);
        return Err(Failed::from(format!(
            "python path returned status {} body {}",
            py_result.status,
            truncate(&py_result.body, 500)
        )));
    }
    if rust_result.status != 200 {
        print_proxy_logs(&cleanup.procs);
        return Err(Failed::from(format!(
            "rust path returned status {} body {}",
            rust_result.status,
            truncate(&rust_result.body, 500)
        )));
    }
    if py_result.served_by_rust {
        return Err(Failed::from(
            "python path was served by Rust (x-litellm-rust leaked)",
        ));
    }
    if !rust_result.served_by_rust {
        print_proxy_logs(&cleanup.procs);
        return Err(Failed::from(
            "rust path fell back to Python (x-litellm-rust missing; is the native extension installed?)",
        ));
    }

    let py_red = redact_clone(&py_result.json);
    let rust_red = redact_clone(&rust_result.json);
    let response_parity = py_red == rust_red;
    let request_parities = compare_requests(&mock_log)?;

    let block = build_snapshot(&request_parities, response_parity, &py_red, &rust_red);

    let mut failure = String::new();
    for (i, rp) in request_parities.iter().enumerate() {
        if !rp.parity {
            failure.push_str(&format!(
                "Request {i} parity false for {name}\n{}\n",
                rp.diff
            ));
        }
    }
    if !response_parity {
        failure.push_str(&format!(
            "Response 0 parity false for {name}\n{}\n",
            pretty_diff(&pretty(&py_red), &pretty(&rust_red))
        ));
    }
    if !failure.is_empty() {
        let _ = check_snapshot(name, toml_path, &block);
        return Err(Failed::from(failure));
    }
    check_snapshot(name, toml_path, &block)?;
    drop(dir);
    Ok(Completion::Completed)
}

fn build_request_body(case: &TestCase) -> Value {
    let messages: Vec<Value> = case
        .messages
        .iter()
        .map(|m| json!({"role": m.role, "content": m.content}))
        .collect();
    let mut body = json!({"model": case.model, "messages": messages});
    if let Some(t) = case.temperature {
        body["temperature"] = json!(t);
    }
    if let Some(s) = case.seed {
        body["seed"] = json!(s);
    }
    if let Some(m) = case.max_tokens {
        body["max_tokens"] = json!(m);
    }
    body
}

fn spawn_proxy(
    config: &Path,
    port: u16,
    rust: bool,
    dir: &Path,
    label: &str,
) -> Result<ProxyProc, Failed> {
    let bin = env::var(BIN_ENV).unwrap_or_else(|_| String::from("litellm"));
    let out_path = dir.join(format!("{label}.out.log"));
    let err_path = dir.join(format!("{label}.err.log"));
    let out = fs::File::create(&out_path).map_err(Failed::from)?;
    let err = fs::File::create(&err_path).map_err(Failed::from)?;
    let mut cmd = Command::new(&bin);
    cmd.arg("--host")
        .arg(HOST)
        .arg("--port")
        .arg(port.to_string())
        .arg("--config")
        .arg(config)
        .stdout(Stdio::from(out))
        .stderr(Stdio::from(err));
    if rust {
        cmd.env("LITELLM_RUST", "true");
    } else {
        cmd.env_remove("LITELLM_RUST");
    }
    let child = cmd
        .spawn()
        .map_err(|e| Failed::from(format!("failed to spawn litellm ({bin}): {e}")))?;
    Ok(ProxyProc {
        child,
        label: label.to_string(),
        out_log: out_path,
        err_log: err_path,
    })
}

fn wait_healthy(port: u16, deadline: Instant) -> Result<(), Failed> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .map_err(Failed::from)?;
    let url = format!("http://{HOST}:{port}{HEALTH_PATH}");
    while Instant::now() < deadline {
        if let Ok(resp) = client.get(&url).send()
            && resp.status().is_success()
        {
            return Ok(());
        }
        thread::sleep(POLL_INTERVAL);
    }
    Err(Failed::from(format!(
        "proxy on port {port} did not become healthy within {HEALTH_TIMEOUT:?}"
    )))
}

fn send_chat(port: u16, req: &Value) -> Result<ChatResult, Failed> {
    let client = reqwest::blocking::Client::builder()
        .timeout(CHAT_TIMEOUT)
        .build()
        .map_err(Failed::from)?;
    let url = format!("http://{HOST}:{port}{CHAT_PATH}");
    let resp = client
        .post(&url)
        .header("Authorization", format!("Bearer {MASTER_KEY}"))
        .header("Content-Type", "application/json")
        .json(req)
        .send()
        .map_err(|e| Failed::from(format!("chat request to port {port} failed: {e}")))?;
    let status = resp.status().as_u16();
    let served_by_rust = resp
        .headers()
        .get(RUST_HEADER)
        .map(|v| v.as_bytes() == b"true")
        .unwrap_or(false);
    let text = resp.text().map_err(Failed::from)?;
    let json: Value = serde_json::from_str(&text).map_err(|_| {
        Failed::from(format!(
            "non-JSON response (status {status}) from port {port}: {}",
            truncate(&text, 500)
        ))
    })?;
    Ok(ChatResult {
        status,
        served_by_rust,
        body: text,
        json,
    })
}

fn serve_mock(listener: TcpListener, state: Arc<MockState>) {
    let _ = listener.set_nonblocking(true);
    while !state.stop.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((stream, _)) => {
                let _ = handle_mock_request(stream, &state);
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => thread::sleep(MOCK_POLL),
            Err(_) => break,
        }
    }
}

fn handle_mock_request(mut stream: TcpStream, state: &MockState) -> std::io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    stream.set_write_timeout(Some(Duration::from_secs(10)))?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut header_lines: Vec<String> = Vec::new();
    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            return Ok(());
        }
        if line.trim().is_empty() {
            break;
        }
        header_lines.push(line);
    }

    let mut content_len = 0usize;
    let mut expect_continue = false;
    for h in &header_lines {
        let lower = h.to_ascii_lowercase();
        if let Some(rest) = lower.strip_prefix("content-length:") {
            if let Ok(n) = rest.trim().parse::<usize>() {
                content_len = n;
            }
        } else if let Some(rest) = lower.strip_prefix("expect:")
            && rest.trim().contains("100-continue")
        {
            expect_continue = true;
        }
    }

    if expect_continue {
        stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n")?;
    }
    let mut body = vec![0u8; content_len];
    if content_len > 0 {
        reader.read_exact(&mut body)?;
    }
    let body_str = String::from_utf8_lossy(&body).into_owned();
    state
        .requests
        .lock()
        .expect("mock requests mutex")
        .push(ReceivedRequest { body: body_str });

    let idx = state.next.fetch_add(1, Ordering::SeqCst);
    let (status, body_out) = match state.responses.get(idx) {
        Some(r) => (r.status.unwrap_or(200), r.body.clone()),
        None => (
            500,
            String::from("{\"error\":\"mock response sequence exhausted\"}"),
        ),
    };
    let reason = status_reason(status);
    let out = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n{body_out}",
        len = body_out.len(),
    );
    stream.write_all(out.as_bytes())?;
    stream.flush()?;
    Ok(())
}

fn status_reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        201 => "Created",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "OK",
    }
}

fn redact_clone(value: &Value) -> Value {
    let mut v = value.clone();
    redact(&mut v);
    v
}

fn redact(value: &mut Value) {
    match value {
        Value::Object(map) => {
            for (k, v) in map.iter_mut() {
                if is_redact_key(k) {
                    *v = Value::String(String::from("<redacted>"));
                } else {
                    redact(v);
                }
            }
        }
        Value::Array(arr) => {
            for v in arr {
                redact(v);
            }
        }
        _ => {}
    }
}

fn is_redact_key(key: &str) -> bool {
    REDACT_KEYS.contains(&key)
}

/// Pair the mock's upstream request log into python-vs-rust pairs and compare
/// each. Python's `send_chat` completes before Rust's starts, so the first half
/// of the arrival-ordered log is Python and the second half is Rust. An odd
/// count means one path made more upstream calls than the other, which is a
/// parity failure on its own.
fn compare_requests(mock_log: &[ReceivedRequest]) -> Result<Vec<RequestParity>, Failed> {
    let total = mock_log.len();
    if total == 0 {
        return Ok(Vec::new());
    }
    if !total.is_multiple_of(2) {
        return Err(Failed::from(format!(
            "mock received {total} upstream requests; expected an even count to pair python vs rust"
        )));
    }
    let half = total / 2;
    let mut out = Vec::with_capacity(half);
    for i in 0..half {
        let py = pretty_request_body(&mock_log[i].body);
        let rust = pretty_request_body(&mock_log[i + half].body);
        out.push(RequestParity {
            parity: py == rust,
            diff: pretty_diff(&py, &rust),
        });
    }
    Ok(out)
}

fn build_snapshot(
    request_parities: &[RequestParity],
    response_parity: bool,
    py: &Value,
    rust: &Value,
) -> String {
    let mut s = String::new();
    for (i, rp) in request_parities.iter().enumerate() {
        let diff = rp.diff.trim_end_matches('\n');
        s.push_str(&format!(
            "Request {i}:\nParity: {}\n{}\n\n",
            if rp.parity { "True" } else { "False" },
            diff,
        ));
    }
    let resp_diff = pretty_diff(&pretty(py), &pretty(rust));
    let resp_diff = resp_diff.trim_end_matches('\n');
    s.push_str(&format!(
        "Response 0:\nParity: {}\n{}\n",
        if response_parity { "True" } else { "False" },
        resp_diff,
    ));
    s
}

fn pretty(value: &Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
}

fn pretty_diff(old: &str, new: &str) -> String {
    let diff = TextDiff::from_lines(old, new);
    let mut out = String::new();
    for change in diff.iter_all_changes() {
        let prefix = match change.tag() {
            ChangeTag::Delete => '-',
            ChangeTag::Insert => '+',
            ChangeTag::Equal => ' ',
        };
        out.push(prefix);
        out.push_str(change.as_str().unwrap_or(""));
    }
    out
}

fn pretty_request_body(body: &str) -> String {
    serde_json::from_str::<Value>(body)
        .map(|v| pretty(&v))
        .unwrap_or_else(|_| body.to_string())
}

fn check_snapshot(name: &str, toml_path: &Path, block: &str) -> Result<(), Failed> {
    let path = toml_path.with_extension("snap");
    let existing = fs::read_to_string(&path).unwrap_or_default();
    if normalize_snap(&existing) == normalize_snap(block) {
        return Ok(());
    }
    if env::var(ACCEPT_ENV).is_ok() {
        fs::write(&path, block).map_err(Failed::from)?;
        eprintln!("accepted snapshot: {}", path.display());
        return Ok(());
    }
    let new_path = path.with_extension("snap.new");
    fs::write(&new_path, block).map_err(Failed::from)?;
    Err(Failed::from(format!(
        "snapshot mismatch for {name}\nexisting: {}\nnew:      {}\nrun with {ACCEPT_ENV}=1 to accept, or diff the two files",
        path.display(),
        new_path.display()
    )))
}

fn normalize_snap(s: &str) -> String {
    s.trim_end().to_string()
}

fn truncate(s: &str, n: usize) -> String {
    let chars: Vec<char> = s.chars().take(n).collect();
    if s.chars().count() <= n {
        chars.into_iter().collect()
    } else {
        let mut t: String = chars.into_iter().collect();
        t.push('…');
        t
    }
}

fn free_port() -> Result<u16, Failed> {
    let listener = TcpListener::bind(format!("{HOST}:0")).map_err(Failed::from)?;
    let port = listener.local_addr().map_err(Failed::from)?.port();
    Ok(port)
}

fn extract_api_base(config: &str) -> Option<String> {
    for line in config.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("api_base:") {
            let value = rest.trim().trim_matches('\'').trim_matches('"');
            return Some(value.to_string());
        }
    }
    None
}

fn upstream_reachable(base: &str, health_path: &str) -> bool {
    let client = match reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
    {
        Ok(c) => c,
        Err(_) => return false,
    };
    client
        .get(format!("{base}{health_path}"))
        .send()
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

fn litellm_available() -> bool {
    let bin = env::var(BIN_ENV).unwrap_or_else(|_| String::from("litellm"));
    if bin.contains('/') || bin.contains('\\') {
        return Path::new(&bin).exists();
    }
    env::var_os("PATH")
        .map(|paths| env::split_paths(&paths).any(|p| p.join(&bin).exists()))
        .unwrap_or(false)
}

fn native_extension_available() -> bool {
    let python = env::var(PYTHON_ENV).unwrap_or_else(|_| python_from_bin());
    let out = Command::new(&python)
        .args([
            "-c",
            "import litellm.rust_bridge.loader as l\nraise SystemExit(0 if l.native_bridge_available() else 1)",
        ])
        .output();
    matches!(out, Ok(o) if o.status.success())
}

fn python_from_bin() -> String {
    if let Ok(b) = env::var(BIN_ENV)
        && let Some(parent) = Path::new(&b).parent()
    {
        return parent.join("python").to_string_lossy().into_owned();
    }
    String::from("python3")
}

fn ensure_environment() -> Result<(), String> {
    let repo = repo_root();
    let venv = repo.join(".venv");
    let maturin = venv.join("bin").join("maturin");
    if !maturin.exists() {
        return Err(format!(
            "pinned maturin not found at {p}; create the venv first: `just venv`",
            p = maturin.display()
        ));
    }
    let python = venv.join("bin").join("python");
    ensure_proxy_extras(&python, &repo)?;
    build_native_extension(&maturin, &repo)?;
    if env::var_os(BIN_ENV).is_none()
        && let litellm_bin = venv.join("bin").join("litellm")
        && litellm_bin.exists()
    {
        // SAFETY: main runs single-threaded before any trial threads spawn.
        unsafe { env::set_var(BIN_ENV, &litellm_bin) };
    }
    Ok(())
}

fn ensure_proxy_extras(python: &Path, repo: &Path) -> Result<(), String> {
    let installed = Command::new(python)
        .args(["-c", "import websockets"])
        .status()
        .is_ok_and(|s| s.success());
    if installed {
        return Ok(());
    }
    eprintln!("==> Installing proxy extras (litellm[proxy])");
    let status = Command::new(python)
        .args(["-m", "pip", "install", "-e", ".[proxy]"])
        .current_dir(repo)
        .status()
        .map_err(|e| format!("pip install .[proxy] failed to start: {e}"))?;
    if !status.success() {
        return Err(String::from("pip install -e .[proxy] failed"));
    }
    Ok(())
}

fn build_native_extension(maturin: &Path, repo: &Path) -> Result<(), String> {
    eprintln!("==> Building and installing the native Rust extension");
    let status = Command::new(maturin)
        .args(["develop", "--release"])
        .current_dir(repo)
        .status()
        .map_err(|e| format!("maturin develop failed to start: {e}"))?;
    if !status.success() {
        return Err(String::from("maturin develop --release failed"));
    }
    Ok(())
}

fn repo_root() -> PathBuf {
    let candidate = manifest_dir().join("../../..");
    candidate.canonicalize().unwrap_or(candidate)
}

struct LoadedCases {
    cases: Vec<(String, PathBuf, TestCase)>,
    invalid: Vec<(String, String)>,
}

fn load_cases() -> LoadedCases {
    let root = inputs_dir();
    let mut paths: Vec<PathBuf> = Vec::new();
    collect_tomls(&root, &mut paths);
    paths.sort();
    let mut cases = Vec::with_capacity(paths.len());
    let mut invalid: Vec<(String, String)> = Vec::new();
    for p in paths {
        let name = match p.strip_prefix(&root) {
            Ok(rel) => rel
                .with_extension("")
                .to_string_lossy()
                .replace(std::path::MAIN_SEPARATOR, "/"),
            Err(e) => {
                invalid.push((
                    p.to_string_lossy().into_owned(),
                    format!("{}: {e}", p.display()),
                ));
                continue;
            }
        };
        match load_fixture(&p) {
            Ok(case) => cases.push((name, p, case)),
            Err(e) => invalid.push((name, format!("{}: {e}", p.display()))),
        }
    }
    LoadedCases { cases, invalid }
}

fn load_fixture(path: &Path) -> Result<TestCase, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("read failed: {e}"))?;
    toml::from_str::<TestCase>(&text).map_err(|e| format!("parse failed: {e}"))
}

fn collect_tomls(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = fs::read_dir(dir) else {
        return;
    };
    for entry in rd.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect_tomls(&p, out);
        } else if p.extension().is_some_and(|e| e == "toml") {
            out.push(p);
        }
    }
}

fn print_proxy_logs(procs: &[ProxyProc]) {
    for p in procs {
        for (kind, path) in [("out", &p.out_log), ("err", &p.err_log)] {
            if let Ok(content) = fs::read_to_string(path) {
                if content.is_empty() {
                    continue;
                }
                let tail: String = content
                    .lines()
                    .rev()
                    .take(80)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect::<Vec<_>>()
                    .join("\n");
                eprintln!("--- {} {kind} ({}) ---\n{tail}", p.label, path.display());
            }
        }
    }
}

fn manifest_dir() -> PathBuf {
    PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| String::from(".")))
}

fn inputs_dir() -> PathBuf {
    manifest_dir().join(INPUTS_DIR)
}
