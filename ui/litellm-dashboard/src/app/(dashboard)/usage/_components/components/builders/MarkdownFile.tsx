"use client";

import { Check, Copy, FileText } from "lucide-react";
import { useState } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import { copyToClipboard } from "@/utils/dataUtils";

const markdownComponents: Components = {
  h1: ({ children }) => <p className="mb-4 border-b border-border pb-2 text-base font-semibold">{children}</p>,
  h2: ({ children }) => (
    <h3 className="mt-5 mb-1.5 text-xs font-semibold tracking-wide text-muted-foreground uppercase">{children}</h3>
  ),
  h3: ({ children }) => <h4 className="mt-4 mb-1 text-sm font-semibold">{children}</h4>,
  p: ({ children }) => <p className="text-sm leading-6">{children}</p>,
  ol: ({ children }) => (
    <ol className="list-decimal space-y-2 pl-5 text-sm leading-6 marker:text-muted-foreground">{children}</ol>
  ),
  ul: ({ children }) => (
    <ul className="list-disc space-y-2 pl-5 text-sm leading-6 marker:text-muted-foreground">{children}</ul>
  ),
  li: ({ children }) => <li className="pl-1">{children}</li>,
  strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
  code: ({ children }) => <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">{children}</code>,
};

export function MarkdownFile({
  filename,
  markdown,
  variant = "compact",
  showToolbar = true,
  showCopy = true,
}: {
  filename: string;
  markdown: string;
  variant?: "compact" | "document";
  showToolbar?: boolean;
  showCopy?: boolean;
}) {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    if (!(await copyToClipboard(markdown, "Markdown copied"))) return;
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div className={showToolbar ? "overflow-hidden rounded-xl border border-border" : undefined}>
      {showToolbar && (
        <div className="flex h-10 items-center gap-2 border-b border-border bg-muted/40 px-3">
          <FileText aria-hidden="true" className="size-3.5 text-muted-foreground" />
          <span className="font-mono text-xs text-muted-foreground">{filename}</span>
          {showCopy && (
            <button
              type="button"
              onClick={() => void copy()}
              aria-label={`Copy ${filename}`}
              className="ml-auto inline-flex h-7 items-center gap-1.5 rounded-md border border-border bg-background px-2 text-xs font-medium hover:bg-muted"
            >
              {copied ? (
                <Check aria-hidden="true" className="size-3.5 text-success" />
              ) : (
                <Copy aria-hidden="true" className="size-3.5" />
              )}
              {copied ? "Copied" : "Copy"}
            </button>
          )}
        </div>
      )}
      <article className="bg-background">
        <ReactMarkdown
          components={
            variant === "document"
              ? {
                  ...markdownComponents,
                  h1: ({ children }) => <h2 className="mb-3 text-xl font-semibold">{children}</h2>,
                  h2: ({ children }) => <h2 className="mt-7 mb-2 text-lg font-semibold">{children}</h2>,
                  h3: ({ children }) => <h3 className="mt-5 mb-2 text-base font-semibold">{children}</h3>,
                }
              : markdownComponents
          }
        >
          {markdown}
        </ReactMarkdown>
      </article>
    </div>
  );
}
