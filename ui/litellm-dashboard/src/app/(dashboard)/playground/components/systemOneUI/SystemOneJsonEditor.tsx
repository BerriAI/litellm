import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/cva.config";
import { CircleAlert, CircleCheck, TriangleAlert } from "lucide-react";
import { useId, useRef } from "react";
import { PrismLight as SyntaxHighlighter } from "react-syntax-highlighter";
import json from "react-syntax-highlighter/dist/esm/languages/prism/json";
import type { SystemOnePayloadValidation } from "./validate_system_one_payload";

SyntaxHighlighter.registerLanguage("json", json);

const EDITOR_TEXT = "m-0 whitespace-pre p-3 font-mono text-xs leading-5";

const TOKEN_COLORS = [
  "[&_.token.property]:text-sky-700 dark:[&_.token.property]:text-sky-300",
  "[&_.token.string]:text-emerald-700 dark:[&_.token.string]:text-emerald-300",
  "[&_.token.number]:text-amber-700 dark:[&_.token.number]:text-amber-300",
  "[&_.token.boolean]:text-violet-700 dark:[&_.token.boolean]:text-violet-300",
  "[&_.token.null]:text-violet-700 dark:[&_.token.null]:text-violet-300",
  "[&_.token.punctuation]:text-muted-foreground [&_.token.operator]:text-muted-foreground",
].join(" ");

interface SystemOneJsonEditorProps {
  value: string;
  onChange: (value: string) => void;
  validation: SystemOnePayloadValidation;
}

function ValidationStatus({ validation }: { validation: SystemOnePayloadValidation }) {
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

function IssueList({ id, validation }: { id: string; validation: SystemOnePayloadValidation }) {
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

export default function SystemOneJsonEditor({ value, onChange, validation }: SystemOneJsonEditorProps) {
  const issuesId = useId();
  const gutterRef = useRef<HTMLDivElement>(null);
  const highlightRef = useRef<HTMLDivElement>(null);
  const lineCount = value.split("\n").length;
  const hasErrors = !validation.isValid;

  function syncScroll(textarea: HTMLTextAreaElement) {
    if (gutterRef.current) {
      gutterRef.current.scrollTop = textarea.scrollTop;
    }
    if (highlightRef.current) {
      highlightRef.current.scrollTop = textarea.scrollTop;
      highlightRef.current.scrollLeft = textarea.scrollLeft;
    }
  }

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
      <div className="relative flex min-h-80 flex-1">
        <div
          ref={gutterRef}
          aria-hidden="true"
          className={cn(
            EDITOR_TEXT,
            "select-none overflow-hidden border-r bg-muted/50 text-right text-muted-foreground",
          )}
        >
          {Array.from({ length: lineCount }, (_, index) => (
            <div key={index}>{index + 1}</div>
          ))}
        </div>
        <div className="relative flex-1">
          <div ref={highlightRef} aria-hidden="true" className={cn("absolute inset-0 overflow-hidden", TOKEN_COLORS)}>
            <SyntaxHighlighter
              language="json"
              style={{}}
              useInlineStyles={false}
              PreTag="div"
              className={EDITOR_TEXT}
              codeTagProps={{ className: "language-json" }}
            >
              {`${value}\n`}
            </SyntaxHighlighter>
          </div>
          <textarea
            aria-label="System One JSON payload"
            aria-invalid={hasErrors}
            aria-describedby={issuesId}
            value={value}
            onChange={(event) => onChange(event.target.value)}
            onScroll={(event) => syncScroll(event.currentTarget)}
            spellCheck={false}
            autoCapitalize="off"
            autoComplete="off"
            wrap="off"
            placeholder="Paste or write a System One request"
            className={cn(
              EDITOR_TEXT,
              "absolute inset-0 size-full resize-none overflow-auto bg-transparent text-transparent caret-foreground outline-none selection:bg-primary/20 placeholder:text-muted-foreground",
            )}
          />
        </div>
      </div>
      <IssueList id={issuesId} validation={validation} />
    </div>
  );
}
