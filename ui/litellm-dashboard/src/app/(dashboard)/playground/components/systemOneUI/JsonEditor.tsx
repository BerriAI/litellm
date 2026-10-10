import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/cva.config";
import { CircleAlert, CircleCheck, TriangleAlert } from "lucide-react";
import { useTheme } from "next-themes";
import { useId, useRef } from "react";
import { createElement, PrismLight as SyntaxHighlighter } from "react-syntax-highlighter";
import type { SyntaxHighlighterProps } from "react-syntax-highlighter";
import json from "react-syntax-highlighter/dist/esm/languages/prism/json";
import { vs, vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";
import type { SystemOnePayloadValidation } from "./lib/validatePayload";

SyntaxHighlighter.registerLanguage("json", json);

const EDITOR_TEXT = "m-0 whitespace-pre-wrap wrap-anywhere py-3 font-mono text-xs leading-5 [scrollbar-gutter:stable]";
const GUTTER_WIDTH = "w-11";
type LineRendererProps = Parameters<NonNullable<SyntaxHighlighterProps["renderer"]>>[0];
const CONTENT_INSET = "pl-14 pr-3";
const VS_CODE_LIGHT_PLUS = {
  ...vs,
  property: { color: "#0451a5" },
  string: { color: "#a31515" },
  number: { color: "#098658" },
  boolean: { color: "#0000ff" },
  punctuation: { color: "#000000" },
  operator: { color: "#000000" },
} as const;
const INHERIT_TEXT = {
  fontFamily: "inherit",
  fontSize: "inherit",
  lineHeight: "inherit",
  whiteSpace: "pre-wrap",
  overflowWrap: "anywhere",
  textShadow: "none",
} as const;
const CODE_TAG_PROPS = { className: "language-json", style: INHERIT_TEXT } as const;
const TRANSPARENT_PRE = {
  ...INHERIT_TEXT,
  background: "transparent",
  border: 0,
  margin: 0,
  padding: 0,
  overflow: "visible",
} as const;

interface JsonEditorProps {
  value: string;
  onChange: (value: string) => void;
  validation: SystemOnePayloadValidation;
}

export function ValidationStatus({ validation }: { validation: SystemOnePayloadValidation }) {
  const errorCount = validation.issues.filter((issue) => issue.severity === "error").length;
  if (errorCount > 0) {
    return (
      <Badge variant="destructive">
        {errorCount} {errorCount === 1 ? "issue" : "issues"}
      </Badge>
    );
  }
  return <Badge variant="secondary">Valid payload</Badge>;
}

export function IssueList({ id, validation }: { id: string; validation: SystemOnePayloadValidation }) {
  if (validation.issues.length === 0) {
    return (
      <p id={id} role="status" className="flex items-center gap-1.5 border-t px-3 py-2 text-xs text-muted-foreground">
        <CircleCheck className="size-3.5 text-emerald-600 dark:text-emerald-400" />
        Ready to send
      </p>
    );
  }
  return (
    <ul id={id} aria-label="Payload validation issues" className="grid max-h-36 gap-1 overflow-auto border-t px-3 py-2">
      {validation.issues.map((issue, index) => (
        <li key={`${issue.path}-${index}`} className="flex items-start gap-1.5 text-xs">
          {issue.severity === "error" ? (
            <CircleAlert className="mt-0.5 size-3.5 shrink-0 text-destructive" />
          ) : (
            <TriangleAlert className="mt-0.5 size-3.5 shrink-0 text-amber-600 dark:text-amber-400" />
          )}
          <code className="shrink-0 rounded bg-muted px-1 font-mono">{issue.path}</code>
          <span>{issue.message}</span>
        </li>
      ))}
    </ul>
  );
}

function LineRows({ rows, stylesheet, useInlineStyles }: LineRendererProps) {
  return rows.map((row, line) => {
    const lineElement = { node: row, stylesheet, useInlineStyles, key: line };
    return (
      <div key={line} className="flex">
        <span className={cn(GUTTER_WIDTH, "shrink-0 select-none pr-3 text-right text-zinc-400 dark:text-zinc-500")}>
          {line + 1}
        </span>
        <span className="min-h-5 min-w-0 flex-1 px-3">{createElement(lineElement)}</span>
      </div>
    );
  });
}

export default function JsonEditor({ value, onChange, validation }: JsonEditorProps) {
  const issuesId = useId();
  const highlightRef = useRef<HTMLDivElement>(null);
  const syntaxTheme = useTheme().resolvedTheme === "dark" ? vscDarkPlus : VS_CODE_LIGHT_PLUS;
  const lineCount = value.split("\n").length;
  const hasErrors = !validation.isValid;

  return (
    <div
      className={cn(
        "flex min-h-96 flex-1 flex-col overflow-hidden rounded-md border bg-background",
        hasErrors && "border-destructive/60",
      )}
    >
      <div className="flex items-center justify-between gap-2 border-b px-3 py-2">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">Request JSON</span>
          <ValidationStatus validation={validation} />
        </div>
        <span className="text-xs text-muted-foreground tabular-nums">
          {lineCount} {lineCount === 1 ? "line" : "lines"}
        </span>
      </div>
      <div className="relative min-h-80 flex-1 bg-background dark:bg-[#1e1e1e]">
        <div
          aria-hidden="true"
          className={cn(
            GUTTER_WIDTH,
            "absolute inset-y-0 left-0 border-r bg-muted/50 dark:border-zinc-700 dark:bg-zinc-800/60",
          )}
        />
        <div ref={highlightRef} aria-hidden="true" className={cn("absolute inset-0 overflow-hidden", EDITOR_TEXT)}>
          <SyntaxHighlighter
            language="json"
            style={syntaxTheme}
            customStyle={TRANSPARENT_PRE}
            PreTag="div"
            codeTagProps={CODE_TAG_PROPS}
            renderer={LineRows}
          >
            {value}
          </SyntaxHighlighter>
        </div>
        <textarea
          aria-label="System One JSON payload"
          aria-invalid={hasErrors}
          aria-describedby={issuesId}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onScroll={(event) => {
            if (highlightRef.current) {
              highlightRef.current.scrollTop = event.currentTarget.scrollTop;
            }
          }}
          spellCheck={false}
          autoCapitalize="off"
          autoComplete="off"
          placeholder="Paste or write a System One request"
          className={cn(
            EDITOR_TEXT,
            CONTENT_INSET,
            "absolute inset-0 size-full resize-none overflow-y-auto bg-transparent text-transparent caret-foreground outline-none selection:bg-primary/20 placeholder:text-muted-foreground",
          )}
        />
      </div>
      <IssueList id={issuesId} validation={validation} />
    </div>
  );
}
