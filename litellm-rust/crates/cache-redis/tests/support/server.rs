use std::{
    collections::HashMap,
    io::{Read, Write},
    net::{Shutdown, SocketAddr, TcpListener, TcpStream},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
};

#[derive(Default)]
struct Registry {
    subscribers: HashMap<String, Vec<(u64, TcpStream)>>,
    unsubscribed: Vec<String>,
}

/// A RESP2 server over a real socket that speaks PING, SUBSCRIBE, UNSUBSCRIBE and PUBLISH, so
/// the subscription path runs against the same wire protocol as a live Redis.
pub struct PubSubServer {
    address: SocketAddr,
    registry: Arc<Mutex<Registry>>,
    shutdown: Arc<AtomicBool>,
    acceptor: Option<thread::JoinHandle<()>>,
}

impl PubSubServer {
    pub fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let registry = Arc::new(Mutex::new(Registry::default()));
        let shutdown = Arc::new(AtomicBool::new(false));
        let acceptor = thread::spawn({
            let registry = Arc::clone(&registry);
            let shutdown = Arc::clone(&shutdown);
            let next_id = AtomicU64::new(1);
            move || {
                for stream in listener.incoming() {
                    if shutdown.load(Ordering::Acquire) {
                        break;
                    }
                    let Ok(stream) = stream else { break };
                    let id = next_id.fetch_add(1, Ordering::Relaxed);
                    let registry = Arc::clone(&registry);
                    thread::spawn(move || serve(stream, id, &registry));
                }
            }
        });
        Self {
            address,
            registry,
            shutdown,
            acceptor: Some(acceptor),
        }
    }

    pub fn url(&self) -> String {
        format!("redis://{}", self.address)
    }

    pub fn unsubscribed(&self) -> Vec<String> {
        self.registry.lock().unwrap().unsubscribed.clone()
    }

    /// Closes every subscriber socket, as a restarted server would.
    pub fn drop_subscribers(&self) {
        let mut registry = self.registry.lock().unwrap();
        for (_, stream) in registry
            .subscribers
            .drain()
            .flat_map(|(_, subscribers)| subscribers)
        {
            let _ = stream.shutdown(Shutdown::Both);
        }
    }
}

impl Drop for PubSubServer {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        let _ = TcpStream::connect(self.address);
        if let Some(acceptor) = self.acceptor.take() {
            let _ = acceptor.join();
        }
    }
}

fn serve(mut stream: TcpStream, id: u64, registry: &Mutex<Registry>) {
    let mut buffer = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        let read = match stream.read(&mut chunk) {
            Ok(0) | Err(_) => break,
            Ok(read) => read,
        };
        buffer.extend_from_slice(&chunk[..read]);
        while let Some((command, consumed)) = parse_command(&buffer) {
            buffer.drain(..consumed);
            let reply = respond(&command, &stream, id, registry);
            if stream.write_all(&reply).is_err() {
                return;
            }
        }
    }
    let mut registry = registry.lock().unwrap();
    for subscribers in registry.subscribers.values_mut() {
        subscribers.retain(|(subscriber, _)| *subscriber != id);
    }
}

fn respond(
    command: &[Vec<u8>],
    stream: &TcpStream,
    id: u64,
    registry: &Mutex<Registry>,
) -> Vec<u8> {
    let name = String::from_utf8_lossy(&command[0]).to_ascii_uppercase();
    let text = |bytes: &[u8]| String::from_utf8_lossy(bytes).into_owned();
    match name.as_str() {
        "PING" => b"+PONG\r\n".to_vec(),
        "SUBSCRIBE" => {
            let mut registry = registry.lock().unwrap();
            command[1..]
                .iter()
                .flat_map(|channel| {
                    let channel = text(channel);
                    registry
                        .subscribers
                        .entry(channel.clone())
                        .or_default()
                        .push((id, stream.try_clone().unwrap()));
                    let count = subscriptions(&registry, id);
                    array(&[bulk(b"subscribe"), bulk(channel.as_bytes()), integer(count)])
                })
                .collect()
        }
        "UNSUBSCRIBE" | "PUNSUBSCRIBE" => {
            let mut registry = registry.lock().unwrap();
            let named: Vec<String> = command[1..].iter().map(|channel| text(channel)).collect();
            let channels: Vec<String> = if named.is_empty() {
                registry
                    .subscribers
                    .iter()
                    .filter(|(_, subscribers)| subscribers.iter().any(|(sub, _)| *sub == id))
                    .map(|(channel, _)| channel.clone())
                    .collect()
            } else {
                named
            };
            if channels.is_empty() {
                return array(&[
                    bulk(&name.to_ascii_lowercase().into_bytes()),
                    b"$-1\r\n".to_vec(),
                    integer(0),
                ]);
            }
            channels
                .into_iter()
                .flat_map(|channel| {
                    if let Some(subscribers) = registry.subscribers.get_mut(&channel) {
                        let before = subscribers.len();
                        subscribers.retain(|(sub, _)| *sub != id);
                        if subscribers.len() < before {
                            registry.unsubscribed.push(channel.clone());
                        }
                    }
                    let count = subscriptions(&registry, id);
                    array(&[
                        bulk(b"unsubscribe"),
                        bulk(channel.as_bytes()),
                        integer(count),
                    ])
                })
                .collect()
        }
        "PUBLISH" => {
            let channel = text(&command[1]);
            let payload = &command[2];
            let mut registry = registry.lock().unwrap();
            let delivered = registry
                .subscribers
                .get_mut(&channel)
                .map_or(0, |subscribers| {
                    subscribers.retain_mut(|(_, subscriber)| {
                        subscriber
                            .write_all(&array(&[
                                bulk(b"message"),
                                bulk(channel.as_bytes()),
                                bulk(payload),
                            ]))
                            .is_ok()
                    });
                    subscribers.len()
                });
            integer(delivered)
        }
        _ => b"+OK\r\n".to_vec(),
    }
}

fn subscriptions(registry: &Registry, id: u64) -> usize {
    registry
        .subscribers
        .values()
        .filter(|subscribers| subscribers.iter().any(|(sub, _)| *sub == id))
        .count()
}

fn parse_command(buffer: &[u8]) -> Option<(Vec<Vec<u8>>, usize)> {
    fn line(buffer: &[u8], start: usize) -> Option<(&[u8], usize)> {
        let end = buffer[start..]
            .windows(2)
            .position(|window| window == b"\r\n")?
            + start;
        Some((&buffer[start..end], end + 2))
    }
    fn length(line: &[u8]) -> Option<usize> {
        std::str::from_utf8(line.get(1..)?).ok()?.parse().ok()
    }
    let (header, mut cursor) = line(buffer, 0)?;
    let count = length(header)?;
    let mut command = Vec::with_capacity(count);
    for _ in 0..count {
        let (size, start) = line(buffer, cursor)?;
        let size = length(size)?;
        let argument = buffer.get(start..start + size)?;
        command.push(argument.to_vec());
        cursor = start + size + 2;
        if buffer.len() < cursor {
            return None;
        }
    }
    Some((command, cursor))
}

fn bulk(bytes: &[u8]) -> Vec<u8> {
    let mut out = format!("${}\r\n", bytes.len()).into_bytes();
    out.extend_from_slice(bytes);
    out.extend_from_slice(b"\r\n");
    out
}

fn integer(value: usize) -> Vec<u8> {
    format!(":{value}\r\n").into_bytes()
}

fn array(items: &[Vec<u8>]) -> Vec<u8> {
    let mut out = format!("*{}\r\n", items.len()).into_bytes();
    for item in items {
        out.extend_from_slice(item);
    }
    out
}
