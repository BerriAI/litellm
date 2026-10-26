#![cfg(unix)]

use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use litellm_http::{HttpSettings, Resolution};
use litellm_model_catalog::{BUNDLED_REGISTRY, Catalog, Installed, Origin};
use litellm_model_catalog_refresh::{Adopted, Checked, Install, Refresher, Source, Stale};
use nix::{
    sys::{
        signal::{Signal, kill},
        wait::{WaitPidFlag, WaitStatus, waitpid},
    },
    unistd::{ForkResult, Pid, fork},
};
use rstest::rstest;

#[derive(Default)]
struct Host {
    generation: AtomicU64,
    failures: AtomicU64,
}

impl Install for Host {
    fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    fn install(&self, seen: u64, adopted: Adopted) -> Result<u64, Stale> {
        assert!(adopted.installed.catalog.model_count() > 50);
        self.generation
            .compare_exchange(seen, seen + 1, Ordering::AcqRel, Ordering::Acquire)
            .map(|_| seen + 1)
            .map_err(|_| Stale)
    }

    fn checked(&self, outcome: Checked) {
        if matches!(outcome, Checked::Failed { .. }) {
            self.failures.fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn accepted(listener: &TcpListener) -> TcpStream {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match listener.accept() {
            Ok((mut stream, _)) => {
                stream
                    .set_read_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                stream
                    .set_write_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                let mut request = [0; 4096];
                assert!(stream.read(&mut request).unwrap() > 0);
                return stream;
            }
            Err(error)
                if error.kind() == std::io::ErrorKind::WouldBlock && Instant::now() < deadline =>
            {
                thread::sleep(Duration::from_millis(1))
            }
            error => panic!("refresh did not connect: {error:?}"),
        }
    }
}

fn respond(stream: &mut TcpStream) {
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nETag: new\r\nConnection: close\r\n\r\n",
        BUNDLED_REGISTRY.len()
    )
    .unwrap();
    stream.write_all(BUNDLED_REGISTRY).unwrap();
}

fn adopted(host: &Host) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while host.generation() == 0
        && host.failures.load(Ordering::Relaxed) == 0
        && Instant::now() < deadline
    {
        thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(host.generation(), 1, "child or parent did not adopt");
    assert_eq!(host.failures.load(Ordering::Relaxed), 0);
}

struct Child(Option<Pid>);

impl Child {
    fn finish(mut self) {
        let deadline = Instant::now() + Duration::from_secs(8);
        loop {
            match waitpid(self.0.unwrap(), Some(WaitPidFlag::WNOHANG)).unwrap() {
                WaitStatus::StillAlive if Instant::now() < deadline => {
                    thread::sleep(Duration::from_millis(1))
                }
                WaitStatus::Exited(_, 0) => {
                    self.0 = None;
                    return;
                }
                status => panic!("forked refresher failed or hung: {status:?}"),
            }
        }
    }
}

impl Drop for Child {
    fn drop(&mut self) {
        if let Some(pid) = self.0 {
            let _ = kill(pid, Signal::SIGKILL);
            let _ = waitpid(pid, None);
        }
    }
}

#[rstest]
#[case::in_flight_bundled(false)]
#[case::already_remote(true)]
fn fork_recovers_http_without_disturbing_parent(#[case] remote: bool) {
    if std::env::var_os("LITELLM_CATALOG_FORK_CHILD").is_none() {
        let current = thread::current();
        let status = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                current.name().unwrap(),
                "--nocapture",
                "--test-threads=1",
            ])
            .env("LITELLM_CATALOG_FORK_CHILD", "1")
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let source = Source::parse(&format!(
        "http://{}/catalog",
        listener.local_addr().unwrap()
    ))
    .unwrap();
    let fetch = source
        .fetcher(&Resolution::from(&HttpSettings::default()).config)
        .unwrap();
    let initial = Installed {
        catalog: Arc::new(Catalog::parse(BUNDLED_REGISTRY).unwrap().value),
        revision: "base".to_owned(),
        validator: None,
        origin: if remote {
            Origin::Remote {
                source: source.id(),
            }
        } else {
            Origin::Bundled
        },
    };
    let host = Arc::new(Host::default());
    let refresher = Refresher::start(source.id(), fetch, &initial, host.clone());
    let mut parent_request = accepted(&listener);

    // Isolated subprocess: the only background thread is the refresher being tested.
    match unsafe { fork() }.unwrap() {
        ForkResult::Child => {
            refresher.after_fork();
            refresher.after_fork();
            if remote {
                thread::sleep(Duration::from_millis(100));
                assert_eq!(host.generation(), 0);
            } else {
                adopted(&host);
            }
            refresher.stop();
            drop(refresher);
            // The inherited test harness and runtime belong to the parent, not to this child.
            unsafe { nix::libc::_exit(0) };
        }
        ForkResult::Parent { child } => {
            let child = Child(Some(child));
            if !remote {
                let mut child_request = accepted(&listener);
                respond(&mut child_request);
            }
            child.finish();
            assert!(
                matches!(listener.accept(), Err(error) if error.kind() == std::io::ErrorKind::WouldBlock)
            );
            assert_eq!(host.generation(), 0);
            respond(&mut parent_request);
            adopted(&host);
            refresher.stop();
        }
    }
}
