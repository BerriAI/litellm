use std::io::{IsTerminal, Write};
use std::sync::OnceLock;
use std::sync::{Arc, Mutex};

use colored_json::{ColorMode, ColoredFormatter, Output, PrettyFormatter};

use super::{LogEvent, LogSink};

#[derive(Clone, Copy)]
pub enum RenderMode {
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
        Self::new(
            Box::new(std::io::stderr()),
            *render_mode(),
            ColorMode::Auto(Output::StdErr),
        )
    }

    pub fn new(writer: Box<dyn Write + Send>, mode: RenderMode, color_mode: ColorMode) -> Self {
        Self {
            mode,
            color_mode: color_mode.eval(),
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

    fn sample_event() -> LogEvent {
        LogEvent::Request(ProviderRequestEvent {
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
        })
    }

    fn emit_to_string(event: &LogEvent, mode: RenderMode, color_mode: ColorMode) -> String {
        let buffer = Arc::new(Mutex::new(Vec::new()));
        let hook = ConsoleDebugHook::new(Box::new(Buffer(buffer.clone())), mode, color_mode);
        hook.emit(event);
        let bytes = buffer.lock().expect("buffer lock").clone();
        String::from_utf8(bytes).expect("output is utf8")
    }

    #[test]
    fn compact_output_is_canonical_json() {
        let event = sample_event();
        let expected = serde_json::to_value(&event).expect("event serializes");
        let output = emit_to_string(&event, RenderMode::Compact, ColorMode::Off);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(output.trim()).expect("output is JSON"),
            expected
        );
    }

    #[test]
    fn pretty_output_has_header_separator_and_indented_payload() {
        let event = sample_event();
        let output = emit_to_string(&event, RenderMode::Pretty, ColorMode::Off);
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

    fn strip_ansi(text: &str) -> String {
        text.split('\x1b')
            .enumerate()
            .map(|(index, chunk)| {
                if index == 0 {
                    return chunk;
                }
                chunk
                    .split_once(|character: char| character.is_ascii_alphabetic())
                    .map_or("", |(_, rest)| rest)
            })
            .collect()
    }

    #[test]
    fn color_mode_is_injected_not_read_from_process_stderr() {
        let event = sample_event();
        let colored = emit_to_string(&event, RenderMode::Pretty, ColorMode::On);
        let plain = emit_to_string(&event, RenderMode::Pretty, ColorMode::Off);

        let payload_start = colored.find('{').expect("payload starts");
        let colored_header = format!("\x1b[36m{}\x1b[0m", header(&event));

        assert!(colored.contains(&colored_header));
        assert!(colored.contains("\x1b[2m────────"));
        assert!(colored[payload_start..].contains('\x1b'));
        assert_eq!(strip_ansi(&colored), plain);
        assert!(!plain.contains('\x1b'));
    }

    #[test]
    fn compact_output_is_never_colored() {
        let event = sample_event();
        let output = emit_to_string(&event, RenderMode::Compact, ColorMode::On);
        assert!(!output.contains('\x1b'));
    }
}
