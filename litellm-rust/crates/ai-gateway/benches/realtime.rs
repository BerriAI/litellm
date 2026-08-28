use std::env;
use std::num::NonZeroUsize;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

use clap::Parser;
use futures_util::{SinkExt, StreamExt, stream};
use serde::Deserialize;
use tokio::net::TcpStream;
use tokio::time::{Instant, timeout};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::HeaderValue;
use tokio_tungstenite::tungstenite::http::header::AUTHORIZATION;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async};

const DEFAULT_HOST: &str = "api.openai.com";
const DEFAULT_MODEL: &str = "gpt-realtime-2.1";
const DEFAULT_KEY_ENV: &str = "OPENAI_API_KEY";
const CONVERSATION_ITEM: &str = r#"{"type":"conversation.item.create","item":{"type":"message","role":"user","content":[{"type":"input_text","text":"Say hi."}]}}"#;
const RESPONSE_CREATE: &str = r#"{"type":"response.create"}"#;

type BenchResult<T> = Result<T, String>;
type Socket = WebSocketStream<MaybeTlsStream<TcpStream>>;

#[derive(Parser)]
#[command(
    name = "realtime",
    bin_name = "cargo bench -p litellm-ai-gateway --bench realtime --",
    about = "Load benchmark for the Realtime WebSocket endpoint",
    after_help = "Metrics:\n  dial       TCP, TLS, and WebSocket upgrade\n  session    WebSocket upgrade to session.created\n  1st audio  response.create to response.output_audio.delta\n  total      Full connection wall-clock time"
)]
struct Cli {
    #[arg(long, default_value = DEFAULT_HOST, help = "Target host[:port]")]
    host: String,
    #[arg(long, help = "Bearer token; prefer --api-key-env")]
    api_key: Option<String>,
    #[arg(
        long,
        default_value = DEFAULT_KEY_ENV,
        help = "Environment variable containing the bearer token"
    )]
    api_key_env: String,
    #[arg(short, long, default_value = DEFAULT_MODEL, help = "Realtime model")]
    model: String,
    #[arg(
        short = 'n',
        long,
        default_value = "5",
        help = "Total WebSocket connections"
    )]
    connections: NonZeroUsize,
    #[arg(
        short = 'c',
        long,
        default_value = "10",
        help = "Maximum concurrent connections"
    )]
    concurrency: NonZeroUsize,
    #[arg(
        short = 't',
        long,
        default_value = "30",
        value_parser = parse_duration,
        help = "Per-connection timeout in seconds"
    )]
    timeout: Duration,
    #[arg(long, help = "Use ws:// instead of wss://")]
    insecure: bool,
    #[arg(short, long, help = "Print sent and received events")]
    verbose: bool,
    #[arg(long, hide = true)]
    bench: bool,
}

#[derive(Clone)]
struct Config {
    host: String,
    api_key: String,
    model: String,
    connections: usize,
    concurrency: usize,
    timeout: Duration,
    insecure: bool,
    verbose: bool,
}

impl TryFrom<Cli> for Config {
    type Error = String;

    fn try_from(cli: Cli) -> Result<Self, Self::Error> {
        if cli.host.is_empty() {
            return Err("--host must not be empty".to_string());
        }
        if cli.model.is_empty() {
            return Err("--model must not be empty".to_string());
        }
        let api_key = match cli.api_key {
            Some(api_key) => api_key,
            None => env::var(&cli.api_key_env).map_err(|_| {
                format!(
                    "{} is not set; use --api-key-env or --api-key",
                    cli.api_key_env
                )
            })?,
        };
        if api_key.is_empty() {
            return Err("API key must not be empty".to_string());
        }

        Ok(Self {
            host: cli.host,
            api_key,
            model: cli.model,
            connections: cli.connections.get(),
            concurrency: cli.concurrency.get().min(cli.connections.get()),
            timeout: cli.timeout,
            insecure: cli.insecure,
            verbose: cli.verbose,
        })
    }
}

#[derive(Deserialize)]
struct RealtimeEvent {
    #[serde(rename = "type")]
    event_type: String,
    response: Option<Response>,
}

#[derive(Deserialize)]
struct Response {
    #[serde(default)]
    output: Vec<OutputItem>,
}

#[derive(Deserialize)]
struct OutputItem {
    #[serde(default)]
    content: Vec<ContentPart>,
}

#[derive(Deserialize)]
struct ContentPart {
    #[serde(rename = "type")]
    content_type: String,
}

struct ReceivedEvent {
    event: RealtimeEvent,
    raw: String,
}

struct Timings {
    dial_ms: f64,
    session_ms: f64,
    first_audio_ms: f64,
    total_ms: f64,
}

enum ConnectionOutcome {
    Success(Timings),
    Failure(String),
}

struct ConnectionResult {
    id: usize,
    outcome: ConnectionOutcome,
}

struct MetricStats {
    median: f64,
    mean: f64,
    p95: f64,
    p99: f64,
}

#[tokio::main]
async fn main() -> ExitCode {
    match Config::try_from(Cli::parse()) {
        Ok(config) => run(config).await,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

async fn run(config: Config) -> ExitCode {
    println!(
        "Running {} connection(s) (concurrency={}, timeout={}s)",
        config.connections,
        config.concurrency,
        config.timeout.as_secs_f64()
    );

    let config = Arc::new(config);
    let results = stream::iter(1..=config.connections)
        .map(|id| run_connection(id, Arc::clone(&config)))
        .buffer_unordered(config.concurrency)
        .collect::<Vec<_>>()
        .await;
    print_summary(&results, &config);

    if results
        .iter()
        .all(|result| matches!(result.outcome, ConnectionOutcome::Success(_)))
    {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

async fn run_connection(id: usize, config: Arc<Config>) -> ConnectionResult {
    let outcome = match timeout(config.timeout, measure_connection(id, &config)).await {
        Ok(Ok(timings)) => ConnectionOutcome::Success(timings),
        Ok(Err(error)) => ConnectionOutcome::Failure(error),
        Err(_) => ConnectionOutcome::Failure(format!(
            "connection timed out after {}s",
            config.timeout.as_secs_f64()
        )),
    };
    ConnectionResult { id, outcome }
}

async fn measure_connection(id: usize, config: &Config) -> BenchResult<Timings> {
    let overall_started = Instant::now();
    let endpoint = endpoint(config);
    let mut request = endpoint
        .into_client_request()
        .map_err(|error| format!("invalid WebSocket endpoint: {error}"))?;
    let authorization = HeaderValue::from_str(&format!("Bearer {}", config.api_key))
        .map_err(|error| format!("invalid API key header: {error}"))?;
    request.headers_mut().insert(AUTHORIZATION, authorization);

    let dial_started = Instant::now();
    let (mut socket, _) = connect_async(request)
        .await
        .map_err(|error| format!("dial failed: {error}"))?;
    let dial_ms = elapsed_ms(dial_started);

    let session_started = Instant::now();
    wait_for_session(id, config.verbose, &mut socket)
        .await
        .map_err(|error| format!("session failed: {error}"))?;
    let session_ms = elapsed_ms(session_started);

    send_event(id, config.verbose, &mut socket, CONVERSATION_ITEM)
        .await
        .map_err(|error| format!("sending conversation item failed: {error}"))?;
    let first_audio_started = Instant::now();
    send_event(id, config.verbose, &mut socket, RESPONSE_CREATE)
        .await
        .map_err(|error| format!("sending response.create failed: {error}"))?;
    wait_for_first_audio(id, config.verbose, &mut socket)
        .await
        .map_err(|error| format!("waiting for audio failed: {error}"))?;
    let first_audio_ms = elapsed_ms(first_audio_started);

    socket
        .close(None)
        .await
        .map_err(|error| format!("closing WebSocket failed: {error}"))?;

    Ok(Timings {
        dial_ms,
        session_ms,
        first_audio_ms,
        total_ms: elapsed_ms(overall_started),
    })
}

async fn wait_for_session(id: usize, verbose: bool, socket: &mut Socket) -> BenchResult<()> {
    loop {
        let received = receive_event(id, verbose, socket).await?;
        match received.event.event_type.as_str() {
            "session.created" => return Ok(()),
            "error" => return Err(format!("upstream error: {}", received.raw)),
            _ => {}
        }
    }
}

async fn wait_for_first_audio(id: usize, verbose: bool, socket: &mut Socket) -> BenchResult<()> {
    loop {
        let received = receive_event(id, verbose, socket).await?;
        match received.event.event_type.as_str() {
            "response.output_audio.delta" => return Ok(()),
            "error" => return Err(format!("upstream error: {}", received.raw)),
            "response.done" => {
                let output_types = received
                    .event
                    .response
                    .into_iter()
                    .flat_map(|response| response.output)
                    .flat_map(|item| item.content)
                    .map(|content| content.content_type)
                    .collect::<Vec<_>>();
                return Err(format!(
                    "response.done arrived without audio; output types: [{}]",
                    output_types.join(", ")
                ));
            }
            _ => {}
        }
    }
}

async fn send_event(
    id: usize,
    verbose: bool,
    socket: &mut Socket,
    payload: &str,
) -> BenchResult<()> {
    if verbose {
        println!("  [conn {id}] >> {payload}");
    }
    socket
        .send(Message::Text(payload.to_string()))
        .await
        .map_err(|error| error.to_string())
}

async fn receive_event(
    id: usize,
    verbose: bool,
    socket: &mut Socket,
) -> BenchResult<ReceivedEvent> {
    loop {
        let message = socket
            .next()
            .await
            .ok_or_else(|| "WebSocket closed before the expected event".to_string())?
            .map_err(|error| format!("reading WebSocket failed: {error}"))?;
        let raw = match message {
            Message::Text(text) => text,
            Message::Binary(bytes) => String::from_utf8(bytes)
                .map_err(|error| format!("received non-UTF-8 event: {error}"))?,
            Message::Close(frame) => {
                return Err(format!(
                    "WebSocket closed before the expected event: {frame:?}"
                ));
            }
            Message::Ping(_) | Message::Pong(_) | Message::Frame(_) => continue,
        };
        let event = serde_json::from_str::<RealtimeEvent>(&raw)
            .map_err(|error| format!("invalid realtime event: {error}; payload: {raw}"))?;
        if verbose {
            if event.event_type == "response.output_audio.delta" {
                println!("  [conn {id}] << response.output_audio.delta: <audio omitted>");
            } else {
                println!("  [conn {id}] << {}: {raw}", event.event_type);
            }
        }
        return Ok(ReceivedEvent { event, raw });
    }
}

fn endpoint(config: &Config) -> String {
    let scheme = if config.insecure { "ws" } else { "wss" };
    format!(
        "{scheme}://{}/v1/realtime?model={}",
        config.host.trim_end_matches('/'),
        config.model
    )
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

fn print_summary(results: &[ConnectionResult], config: &Config) {
    let successes = results
        .iter()
        .filter_map(|result| match &result.outcome {
            ConnectionOutcome::Success(timings) => Some(timings),
            ConnectionOutcome::Failure(_) => None,
        })
        .collect::<Vec<_>>();
    let separator = "────────────────────────────────────────────────────────────────";

    println!("\n{separator}\n  SUMMARY\n{separator}");
    println!("  Target  : {}", endpoint(config));
    println!("  Model   : {}", config.model);
    println!("  Success : {}/{}", successes.len(), results.len());

    if !successes.is_empty() {
        println!("{separator}");
        println!(
            "  {:<12} {:>10} {:>10} {:>10} {:>10}",
            "metric", "mean", "p50", "p95", "p99"
        );
        print_metric("dial", metric_stats(&successes, |timings| timings.dial_ms));
        print_metric(
            "session",
            metric_stats(&successes, |timings| timings.session_ms),
        );
        print_metric(
            "1st audio",
            metric_stats(&successes, |timings| timings.first_audio_ms),
        );
        print_metric(
            "total",
            metric_stats(&successes, |timings| timings.total_ms),
        );
    }
    println!("{separator}");

    let mut ordered = results.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|result| result.id);
    println!("\n  Per-connection breakdown:");
    println!(
        "  {:>5} {:>10} {:>10} {:>12} {:>10}",
        "#", "dial", "session", "1st audio", "total"
    );
    for result in ordered {
        match &result.outcome {
            ConnectionOutcome::Success(timings) => println!(
                "  {:>5} {:>8.0}ms {:>8.0}ms {:>10.0}ms {:>8.0}ms",
                result.id,
                timings.dial_ms,
                timings.session_ms,
                timings.first_audio_ms,
                timings.total_ms
            ),
            ConnectionOutcome::Failure(error) => {
                println!("  {:>5} FAILED: {error}", result.id);
            }
        }
    }
}

fn metric_stats(timings: &[&Timings], value: impl Fn(&Timings) -> f64) -> MetricStats {
    let mut values = timings
        .iter()
        .map(|timing| value(timing))
        .collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    MetricStats {
        median: percentile(&values, 50),
        mean: values.iter().sum::<f64>() / values.len() as f64,
        p95: percentile(&values, 95),
        p99: percentile(&values, 99),
    }
}

fn percentile(sorted: &[f64], percentile: usize) -> f64 {
    let index = (sorted.len() * percentile).div_ceil(100).saturating_sub(1);
    sorted[index]
}

fn print_metric(name: &str, stats: MetricStats) {
    println!(
        "  {name:<12} {:>8.0}ms {:>8.0}ms {:>8.0}ms {:>8.0}ms",
        stats.mean, stats.median, stats.p95, stats.p99
    );
}

fn parse_duration(value: &str) -> Result<Duration, String> {
    let seconds = value
        .parse::<f64>()
        .map_err(|error| format!("invalid number: {error}"))?;
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err("must be a positive number".to_string());
    }
    Duration::try_from_secs_f64(seconds).map_err(|error| error.to_string())
}
