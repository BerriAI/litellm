use std::io::{IsTerminal, Write};
use std::sync::OnceLock;
use std::sync::{Arc, Mutex};

use colored_json::{ColorMode, ColoredFormatter, Output, PrettyFormatter};

use super::{LogEvent, LogSink};

#[derive(Clone, Copy)]
enum RenderMode {
    Compact,
    Pretty,
}

pub struct ConsoleDebugHook {
    mode: RenderMode,
    color_mode: ColorMode,
    output: Mutex<Box<dyn Write + Send>>,
}

impl ConsoleDebugHook {
    pub fn from_env() -> Self {
        Self::with_writer(Box::new(std::io::stderr()))
    }

    pub fn with_writer(writer: Box<dyn Write + Send>) -> Self {
        Self::with_writer_and_mode(writer, matches!(*render_mode(), RenderMode::Pretty))
    }

    pub fn with_writer_and_mode(writer: Box<dyn Write + Send>, pretty: bool) -> Self {
        Self {
            mode: if pretty {
                RenderMode::Pretty
            } else {
                RenderMode::Compact
            },
            color_mode: ColorMode::Auto(Output::StdErr).eval(),
            output: Mutex::new(writer),
        }
    }
}

pub fn hook(enabled: bool) -> Option<Arc<dyn LogSink>> {
    enabled.then(|| Arc::new(ConsoleDebugHook::from_env()) as Arc<dyn LogSink>)
}

fn render_mode() -> &'static RenderMode {
    static MODE: OnceLock<RenderMode> = OnceLock::new();
    MODE.get_or_init(|| {
        if std::env::var("JSON_LOGS")
            .map(|value| value.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
            || !std::io::stderr().is_terminal()
        {
            RenderMode::Compact
        } else {
            RenderMode::Pretty
        }
    })
}

fn header(event: &LogEvent) -> String {
    match event {
        LogEvent::Request(value) => {
            format!("provider.request {} {}", value.call_id, value.provider)
        }
        LogEvent::Response(value) => format!(
            "provider.response {} {} status={} duration_ms={}",
            value.call_id, value.provider, value.status, value.duration_ms
        ),
        LogEvent::StreamStarted(value) => format!(
            "provider.stream.started {} {} status={}",
            value.call_id, value.provider, value.status
        ),
        LogEvent::StreamCompleted(value) => format!(
            "provider.stream.completed {} {} duration_ms={}",
            value.call_id, value.provider, value.duration_ms
        ),
        LogEvent::Error(value) => format!(
            "provider.error {} {}{} duration_ms={}",
            value.call_id,
            value.provider,
            value
                .status
                .map_or(String::new(), |status| format!(" status={status}")),
            value.duration_ms
        ),
    }
}

fn decorate(value: &str, color_mode: ColorMode, code: &str) -> String {
    if color_mode == ColorMode::On {
        format!("\x1b[{code}m{value}\x1b[0m")
    } else {
        value.to_string()
    }
}

impl LogSink for ConsoleDebugHook {
    fn emit(&self, event: &LogEvent) {
        let Ok(mut output) = self.output.lock() else {
            return;
        };
        let Ok(json) = serde_json::to_string(event) else {
            return;
        };
        match self.mode {
            RenderMode::Compact => {
                let _ = writeln!(output, "{json}");
            }
            RenderMode::Pretty => {
                let pretty = serde_json::to_string_pretty(event).unwrap_or(json);
                let _ = writeln!(
                    output,
                    "{}",
                    decorate(&header(event), self.color_mode, "36")
                );
                let rendered = if self.color_mode == ColorMode::Off {
                    pretty
                } else {
                    ColoredFormatter::new(PrettyFormatter::new())
                        .to_colored_json(event, self.color_mode)
                        .unwrap_or(pretty)
                };
                let _ = writeln!(output, "{rendered}");
                let _ = writeln!(
                    output,
                    "{}",
                    decorate(
                        "────────────────────────────────────────",
                        self.color_mode,
                        "2"
                    )
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use serde_json::json;

    use crate::logging::ProviderRequestEvent;

    use super::*;
    struct Buffer(Arc<Mutex<Vec<u8>>>);

    impl Write for Buffer {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().expect("buffer lock").extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn compact_output_is_canonical_json() {
        let buffer = Arc::new(Mutex::new(Vec::new()));
        let hook = ConsoleDebugHook::with_writer_and_mode(Box::new(Buffer(buffer.clone())), false);
        let event = LogEvent::Request(ProviderRequestEvent {
            source: "litellm-rust",
            call_id: "call_01".to_string(),
            provider: "anthropic".to_string(),
            model: "claude".to_string(),
            stream: false,
            method: "POST",
            url: "https://example.test".to_string(),
            headers: Default::default(),
            body: json!({"prompt": "visible"}),
            body_truncated: None,
            body_original_bytes: None,
        });
        let expected = serde_json::to_value(&event).expect("event serializes");
        hook.emit(&event);
        let output =
            String::from_utf8(buffer.lock().expect("buffer lock").clone()).expect("output is utf8");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(output.trim()).expect("output is JSON"),
            expected
        );
    }

    #[test]
    fn pretty_output_has_header_separator_and_indented_payload() {
        let buffer = Arc::new(Mutex::new(Vec::new()));
        let hook = ConsoleDebugHook::with_writer_and_mode(Box::new(Buffer(buffer.clone())), true);
        let event = LogEvent::Request(ProviderRequestEvent {
            source: "litellm-rust",
            call_id: "call_01".to_string(),
            provider: "anthropic".to_string(),
            model: "claude".to_string(),
            stream: false,
            method: "POST",
            url: "https://example.test".to_string(),
            headers: Default::default(),
            body: json!({"prompt": "visible"}),
            body_truncated: None,
            body_original_bytes: None,
        });
        hook.emit(&event);
        let output =
            String::from_utf8(buffer.lock().expect("buffer lock").clone()).expect("output is utf8");
        assert!(!output.contains('\x1b'));
        let payload = &output
            [output.find('{').expect("payload starts")..=output.rfind('}').expect("payload ends")];
        let expected = serde_json::to_string_pretty(&event).expect("event pretty serializes");
        assert_eq!(payload, expected);
        assert!(
            payload.find("\"event\"").expect("event key")
                < payload.find("\"body\"").expect("body key")
        );
        assert!(output.contains("provider.request call_01 anthropic"));
        assert!(output.contains("────────────────"));
        assert!(output.contains("\n  \"event\""));
    }
}
