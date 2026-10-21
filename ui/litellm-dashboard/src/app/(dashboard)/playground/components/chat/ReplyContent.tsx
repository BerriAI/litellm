"use client";

import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { coy } from "react-syntax-highlighter/dist/esm/styles/prism";
import remarkGfm from "remark-gfm";
import type { MessageType } from "@/components/chat_ui/types";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useSyntaxTheme } from "@/hooks/useSyntaxTheme";
import AudioRenderer from "../chat_ui/AudioRenderer";

const textOf = (message: MessageType): string => (typeof message.content === "string" ? message.content : "");

function CodeBlock({ language, code }: { language: string; code: string }) {
  const theme = useSyntaxTheme(coy);
  return (
    <SyntaxHighlighter
      style={theme}
      language={language}
      PreTag="div"
      customStyle={{ margin: 0, fontSize: "0.75rem" }}
      className="my-2 overflow-x-auto rounded-lg border border-border"
    >
      {code}
    </SyntaxHighlighter>
  );
}

export function Markdown({ children }: { children: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        p: ({ node: _node, ...props }) => <p className="my-2 first:mt-0 last:mb-0" {...props} />,
        ul: ({ node: _node, ...props }) => <ul className="my-2 list-disc space-y-1 pl-5" {...props} />,
        ol: ({ node: _node, ...props }) => <ol className="my-2 list-decimal space-y-1 pl-5" {...props} />,
        a: ({ node: _node, ...props }) => (
          <a className="underline underline-offset-3" target="_blank" rel="noreferrer" {...props} />
        ),
        pre: ({ node: _node, children }) => <>{children}</>,
        code: ({ node: _node, className, children, ...props }) => {
          const language = /language-(\w+)/.exec(className ?? "")?.[1];
          const code = String(children).replace(/\n$/, "");
          if (language || code.includes("\n")) return <CodeBlock language={language ?? "text"} code={code} />;
          return (
            <code className="rounded-sm bg-muted px-1 py-0.5 font-mono text-xs" {...props}>
              {children}
            </code>
          );
        },
        table: ({ node: _node, ...props }) => (
          <div className="my-2 rounded-lg border border-border">
            <Table {...props} />
          </div>
        ),
        thead: ({ node: _node, ...props }) => <TableHeader {...props} />,
        tbody: ({ node: _node, ...props }) => <TableBody {...props} />,
        tr: ({ node: _node, ...props }) => <TableRow {...props} />,
        th: ({ node: _node, ...props }) => <TableHead {...props} />,
        td: ({ node: _node, ...props }) => <TableCell className="whitespace-normal" {...props} />,
      }}
    >
      {children}
    </ReactMarkdown>
  );
}

function GeneratedImage({ src }: { src: string }) {
  return <img src={src} alt="Generated image" className="max-h-[500px] max-w-full rounded-lg border border-border" />;
}

function Embeddings({ value }: { value: string }) {
  return (
    <ScrollArea className="max-h-64 w-full rounded-lg border border-border">
      <pre className="p-3 font-mono text-xs leading-5 whitespace-pre">{value}</pre>
      <ScrollBar orientation="horizontal" />
    </ScrollArea>
  );
}

export function ReplyBody({ message }: { message: MessageType }) {
  if (message.isImage) return <GeneratedImage src={textOf(message)} />;
  if (message.isAudio) return <AudioRenderer message={message} />;
  if (message.isEmbeddings) return <Embeddings value={textOf(message)} />;
  return (
    <>
      <Markdown>{textOf(message)}</Markdown>
      {message.image && <GeneratedImage src={message.image.url} />}
    </>
  );
}
