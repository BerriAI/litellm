import { parseJson, parseMessages, previewText } from "@/components/view_logs/TraceView/traceUtils";

export interface PreviewLine {
  label: string;
  text: string;
  error: boolean;
}

const SECTION = /^(Input|Output|Status): ?/;
const OMITTED = /\n?\[\.\.\. preview omitted; read this span for evidence \.\.\.\]\n?/g;
const OK_STATUS = /^STATUS_CODE_(UNSET|OK)\b/;
const ROLE_CONTENT = /"role":\s*"(\w+)",\s*"content":\s*"((?:[^"\\]|\\.)*)/g;

function sections(preview: string): Readonly<Record<string, string>> {
  const lines = preview.split("\n");
  const starts = lines.flatMap((line, index) => (SECTION.test(line) ? [index] : []));
  if (!starts.length) return { Body: preview };
  return Object.fromEntries(
    starts.map((start, n) => {
      const body = lines.slice(start, starts[n + 1] ?? lines.length).join("\n");
      const name = SECTION.exec(body)?.[1] ?? "Body";
      return [name, body.replace(SECTION, "").trim()];
    }),
  );
}

function decode(escaped: string): string {
  const parsed = parseJson(`"${escaped.replace(/\\u[0-9a-fA-F]{0,3}$|\\$/, "")}"`);
  return typeof parsed === "string" ? parsed : escaped;
}

function lastMessage(value: string): string | null {
  const messages = parseMessages(value);
  if (messages?.length) {
    const last = messages.at(-1);
    return last ? `${last.role}: ${last.content}` : null;
  }
  const matches = [...value.matchAll(ROLE_CONTENT)];
  const match = matches.at(-1);
  return match ? `${match[1]}: ${decode(match[2])}` : null;
}

function compactJson(value: string): string {
  const parsed = parseJson(value);
  if (parsed === null || typeof parsed !== "object") return value;
  return JSON.stringify(parsed);
}

function readable(value: string): string {
  const joined = value.replace(OMITTED, " … ");
  return (lastMessage(joined) ?? previewText(compactJson(joined))).replace(/\s+/g, " ").trim();
}

export function spanPreviewLines(preview: string): PreviewLine[] {
  const parts = sections(preview);
  const status = parts.Status ?? "";
  const failed = !!status && !OK_STATUS.test(status);
  const content = [
    ...(parts.Body ? [{ label: "", text: readable(parts.Body), error: false }] : []),
    ...(parts.Input ? [{ label: "in", text: readable(parts.Input), error: false }] : []),
    ...(parts.Output ? [{ label: "out", text: readable(parts.Output), error: failed }] : []),
    ...(failed ? [{ label: "error", text: status.replace(/^STATUS_CODE_ERROR\s*/, ""), error: true }] : []),
  ];
  return content.filter((line) => line.text);
}
