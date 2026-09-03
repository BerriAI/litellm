use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeMap;
use std::ffi::CString;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant, UNIX_EPOCH};

use litellm_core::audio_transcription::transformation::AudioTranscriptionProviderConfig;
use litellm_core::http_utils::body::PreparedJsonBody;
use litellm_core::http_utils::replay::send_json;
use litellm_core::providers::bedrock::audio_transcription::BEDROCK_AUDIO_TRANSCRIPTION_CONFIG;
use litellm_core::providers::bedrock::aws_base::{
    AwsAuthConfig, resolve_credentials, sign_bedrock_digest,
};
use litellm_python_interop::from_py;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

#[path = "../../src/payload.rs"]
mod payload;

struct Allocator;
static TRACK: AtomicBool = AtomicBool::new(false);
static ALLOCATED: AtomicU64 = AtomicU64::new(0);

#[global_allocator]
static ALLOCATOR: Allocator = Allocator;

unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACK.load(Ordering::Relaxed) {
            ALLOCATED.fetch_add(layout.size() as u64, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if TRACK.load(Ordering::Relaxed) {
            ALLOCATED.fetch_add(size as u64, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, size) }
    }
}

fn sink(concurrency: usize) -> (String, std::thread::JoinHandle<Vec<(usize, String)>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!(
        "http://{}/model/benchmark/converse",
        listener.local_addr().unwrap()
    );
    let thread = std::thread::spawn(move || {
        let workers: Vec<_> = (0..concurrency)
            .map(|_| {
                let (socket, _) = listener.accept().unwrap();
                std::thread::spawn(move || {
                    let mut reader = BufReader::new(socket);
                    let mut length = 0;
                    loop {
                        let mut line = String::new();
                        assert_ne!(reader.read_line(&mut line).unwrap(), 0);
                        if line == "\r\n" {
                            break;
                        }
                        if let Some(value) = line.to_lowercase().strip_prefix("content-length:") {
                            length = value.trim().parse::<usize>().unwrap();
                        }
                    }
                    let mut digest = Sha256::new();
                    let mut remaining = length;
                    let mut chunk = [0; 65536];
                    while remaining > 0 {
                        let count = remaining.min(chunk.len());
                        reader.read_exact(&mut chunk[..count]).unwrap();
                        digest.update(&chunk[..count]);
                        remaining -= count;
                    }
                    let digest = format!("{:x}", digest.finalize());
                    write!(
                        reader.get_mut(),
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        digest.len(),
                        digest
                    )
                    .unwrap();
                    (length, digest)
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect()
    });
    (url, thread)
}

pub fn run() {
    let args: Vec<_> = std::env::args().collect();
    let approach = &args[2];
    let mib: usize = args[3].parse().unwrap();
    let concurrency: usize = args[4].parse().unwrap();
    let encoding = &args[5];
    let measurement = &args[6];
    Python::initialize();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();
    let client = reqwest::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .retry(reqwest::retry::never())
        .build()
        .unwrap();
    let (url, server) = sink(concurrency);
    let credentials = runtime
        .block_on(resolve_credentials(
            AwsAuthConfig {
                region_name: Some("us-east-1".into()),
                access_key_id: Some("benchmark".into()),
                secret_access_key: Some("benchmark".into()),
                ..Default::default()
            },
            &|_| None,
        ))
        .unwrap();
    let module = Python::attach(|py| {
        PyModule::from_code(
            py,
            &CString::new(include_str!("pipeline.py")).unwrap(),
            c"pipeline.py",
            c"pipeline",
        )
        .unwrap()
        .unbind()
    });
    let inputs = Python::attach(|py| {
        module
            .bind(py)
            .call_method1("inputs", (mib * 1024 * 1024, concurrency, encoding))
            .unwrap()
            .unbind()
    });
    let before: (f64, u64) = Python::attach(|py| {
        module
            .bind(py)
            .call_method1("start", (measurement,))
            .unwrap()
            .extract()
            .unwrap()
    });
    TRACK.store(measurement == "allocation", Ordering::Relaxed);
    let started = Instant::now();
    let stages = if approach == "python" {
        Python::attach(|py| {
            module
                .bind(py)
                .call_method1("run", (inputs.bind(py), &url))
                .unwrap()
                .extract::<Vec<f64>>()
                .unwrap()
        })
    } else {
        let extracted = Python::attach(|py| {
            inputs
                .bind(py)
                .try_iter()
                .unwrap()
                .map(|input| {
                    let input = input.unwrap();
                    if approach == "buffered" {
                        let encoded = module.bind(py).call_method1("encode", (&input,)).unwrap();
                        from_py::<Value>(&encoded).unwrap().into()
                    } else if encoding == "raw" {
                        payload::audio_payload_from_py(&input).unwrap()
                    } else {
                        payload::payload_from_py(&input).unwrap()
                    }
                })
                .collect::<Vec<_>>()
        });
        let extraction = started.elapsed().as_secs_f64();
        let transformed = extracted
            .into_iter()
            .map(|input| {
                BEDROCK_AUDIO_TRANSCRIPTION_CONFIG
                    .transform_transcription_payload("benchmark", input, Default::default())
                    .unwrap()
                    .body
            })
            .collect::<Vec<_>>();
        let transform = started.elapsed().as_secs_f64();
        let bodies = transformed
            .into_iter()
            .map(|body| {
                if approach == "buffered" {
                    PreparedJsonBody::buffered(serde_json::to_vec(&body).unwrap().into())
                } else {
                    PreparedJsonBody::new(body).unwrap()
                }
            })
            .collect::<Vec<_>>();
        let preparation = started.elapsed().as_secs_f64();
        let signed = bodies
            .iter()
            .map(|body| {
                let digest = body.sha256();
                let headers = sign_bedrock_digest(
                    &url,
                    &digest,
                    &BTreeMap::new(),
                    "us-east-1",
                    &credentials,
                    UNIX_EPOCH + Duration::from_secs(1_700_000_000),
                )
                .unwrap();
                (digest, headers.into_iter().collect::<Vec<_>>())
            })
            .collect::<Vec<_>>();
        let signing = started.elapsed().as_secs_f64();
        runtime.block_on(async {
            let responses = futures_util::future::join_all(bodies.iter().zip(&signed).map(
                |(body, (digest, headers))| async {
                    let response =
                        send_json(&client, &url, body, headers, Duration::from_secs(120), None)
                            .await
                            .unwrap();
                    assert_eq!(response.text().await.unwrap(), *digest);
                },
            ))
            .await;
            std::hint::black_box(responses);
        });
        vec![
            extraction,
            transform - extraction,
            preparation - transform,
            signing - preparation,
            started.elapsed().as_secs_f64() - signing,
        ]
    };
    let elapsed = started.elapsed().as_secs_f64();
    TRACK.store(false, Ordering::Relaxed);
    let allocated = ALLOCATED.load(Ordering::Relaxed);
    let after: (f64, u64, u64) = Python::attach(|py| {
        module
            .bind(py)
            .call_method1("finish", (measurement,))
            .unwrap()
            .extract()
            .unwrap()
    });
    let received = server.join().unwrap();
    let wire_bytes: usize = received.iter().map(|(length, _)| length).sum();
    assert!(received.iter().all(|(_, hash)| hash == &received[0].1));
    println!(
        "{}",
        json!({"approach":approach,"mib":mib,"concurrency":concurrency,"encoding":encoding,"measurement":measurement,"seconds":elapsed,"cpu_seconds":after.0-before.0,"peak_rss_bytes":after.1,"input_rss_bytes":before.1,"rust_allocated_bytes":allocated,"python_peak_traced_bytes":after.2,"wire_mib_per_second":wire_bytes as f64/1048576.0/elapsed,"stage_seconds":stages,"sha256":received[0].1})
    );
}
