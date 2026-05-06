"use client";

/**
 * TerminalTab — right pane Terminal view.
 *
 * Renders an ANSI-colored tail of `terminal_chunk` events. We intentionally
 * avoid xterm.js (heavy dep, virtual cursor model) and instead split the
 * stream into <span>s with computed CSS colors. That's enough for the
 * read-only "watch the build run" UX in the Cursor Cloud Agents reference.
 *
 * Validation #7 expects a span with computed color: red for ANSI \x1b[31m.
 */
import { useMemo } from "react";
import { Empty, Typography } from "antd";
import type { CloudAgentRunEvent, TerminalChunkPayload } from "@/types/cloud-agents";

const { Title } = Typography;

const ANSI_TO_COLOR: Record<string, string> = {
  "30": "#000000",
  "31": "#ff0000",
  "32": "#00aa00",
  "33": "#aa6600",
  "34": "#0000ff",
  "35": "#aa00aa",
  "36": "#00aaaa",
  "37": "#aaaaaa",
  "90": "#666666",
  "91": "#ff5555",
  "92": "#55ff55",
  "93": "#ffff55",
  "94": "#5555ff",
  "95": "#ff55ff",
  "96": "#55ffff",
  "97": "#ffffff",
};

interface AnsiSpan {
  key: string;
  text: string;
  color: string | null;
  bold: boolean;
}

const ESC = "\x1b";

/**
 * Tiny SGR (Select Graphic Rendition) renderer. Handles \x1b[<n>m sequences
 * for foreground color and bold; resets on \x1b[0m. Anything else (cursor
 * moves, background colors, 256-color, truecolor) is dropped silently —
 * good enough for the read-only build-log UX.
 */
function parseAnsi(text: string): AnsiSpan[] {
  const spans: AnsiSpan[] = [];
  let color: string | null = null;
  let bold = false;
  let buf = "";
  let i = 0;
  let key = 0;
  const flush = () => {
    if (buf.length > 0) {
      spans.push({ key: `s-${key++}`, text: buf, color, bold });
      buf = "";
    }
  };
  while (i < text.length) {
    if (text[i] === ESC && text[i + 1] === "[") {
      flush();
      const end = text.indexOf("m", i + 2);
      if (end === -1) {
        // malformed — drop the rest
        break;
      }
      const codes = text.slice(i + 2, end).split(";");
      for (const c of codes) {
        if (c === "" || c === "0") {
          color = null;
          bold = false;
        } else if (c === "1") {
          bold = true;
        } else if (ANSI_TO_COLOR[c]) {
          color = ANSI_TO_COLOR[c];
        }
      }
      i = end + 1;
    } else {
      buf += text[i];
      i += 1;
    }
  }
  flush();
  return spans;
}

interface TerminalTabProps {
  events: CloudAgentRunEvent[];
}

export default function TerminalTab({ events }: TerminalTabProps) {
  const spans = useMemo(() => {
    const text = events
      .filter((e) => e.type === "terminal_chunk")
      .map((e) => (e.payload as unknown as TerminalChunkPayload).text)
      .join("");
    return parseAnsi(text);
  }, [events]);

  return (
    <div className="flex h-full flex-col" data-testid="terminal-tab">
      <div className="border-b border-gray-200 px-4 py-2">
        <Title level={5} className="!mb-0">
          Terminal
        </Title>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto bg-black p-3 font-mono text-xs leading-snug">
        {spans.length === 0 ? (
          <Empty
            image={Empty.PRESENTED_IMAGE_SIMPLE}
            description={<span className="text-gray-400">No terminal output yet</span>}
          />
        ) : (
          <pre className="whitespace-pre-wrap !m-0" data-testid="terminal-output">
            {spans.map((s) => (
              <span
                key={s.key}
                data-testid={s.color ? `ansi-${s.color}` : undefined}
                style={{ color: s.color ?? "#e5e5e5", fontWeight: s.bold ? 700 : 400 }}
              >
                {s.text}
              </span>
            ))}
          </pre>
        )}
      </div>
    </div>
  );
}
