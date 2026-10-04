"use client";

import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

import { cn } from "@/lib/cva.config";

const BLOCK_GAP = "mb-3 last:mb-0";

const components: Components = {
  p: ({ node, ...props }) => <p className={BLOCK_GAP} {...props} />,
  ul: ({ node, ...props }) => <ul className={cn(BLOCK_GAP, "list-disc space-y-1 pl-5")} {...props} />,
  ol: ({ node, ...props }) => <ol className={cn(BLOCK_GAP, "list-decimal space-y-1 pl-5")} {...props} />,
  li: ({ node, ...props }) => (
    <li className="pl-0.5 [&>ol]:mt-1 [&>ol]:mb-0 [&>p]:mb-0 [&>ul]:mt-1 [&>ul]:mb-0" {...props} />
  ),
  h1: ({ node, ...props }) => <h1 className={cn(BLOCK_GAP, "text-base font-semibold")} {...props} />,
  h2: ({ node, ...props }) => <h2 className={cn(BLOCK_GAP, "font-semibold")} {...props} />,
  h3: ({ node, ...props }) => <h3 className={cn(BLOCK_GAP, "font-semibold")} {...props} />,
  strong: ({ node, ...props }) => <strong className="font-semibold text-foreground" {...props} />,
  blockquote: ({ node, ...props }) => (
    <blockquote className={cn(BLOCK_GAP, "border-l-2 border-border pl-3 text-muted-foreground")} {...props} />
  ),
  pre: ({ node, ...props }) => (
    <pre
      className={cn(BLOCK_GAP, "max-w-full overflow-x-auto rounded-md bg-muted p-3 font-mono text-xs leading-relaxed")}
      {...props}
    />
  ),
  code: ({ node, className, ...props }) => (
    <code
      className={cn(
        "rounded-sm bg-muted px-1 py-0.5 font-mono text-xs text-foreground [pre_&]:bg-transparent [pre_&]:p-0",
        className,
      )}
      {...props}
    />
  ),
  a: ({ node, ...props }) => (
    <a className="text-primary underline underline-offset-2" target="_blank" rel="noreferrer" {...props} />
  ),
  table: ({ node, ...props }) => (
    <div className={cn(BLOCK_GAP, "overflow-x-auto")}>
      <table className="w-full border-collapse text-left text-xs" {...props} />
    </div>
  ),
  th: ({ node, ...props }) => <th className="border-b border-border px-2 py-1.5 font-medium" {...props} />,
  td: ({ node, ...props }) => <td className="border-b border-border/60 px-2 py-1.5 align-top" {...props} />,
  img: ({ alt }) => <span>{alt || "Image omitted"}</span>,
};

export function Markdown({ text, className }: { text: string; className?: string }) {
  return (
    <div className={cn("text-sm leading-relaxed break-words text-foreground", className)}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {text}
      </ReactMarkdown>
    </div>
  );
}
