"use client";

import { Braces } from "lucide-react";

import { cn } from "@/lib/cva.config";

import { Block, BlockBadge } from "./Block";
import { FieldTree } from "./FieldTree";
import { Markdown } from "./Markdown";
import { MessageList, ToolResultCard } from "./Messages";
import type { PayloadView, TextFormat } from "./payload";

const TEXT_CARD = "rounded-lg border border-border bg-card px-3.5 py-3";

export function TextBody({ text, format }: { text: string; format: TextFormat }) {
  if (format === "markdown") return <Markdown text={text} className={TEXT_CARD} />;
  return (
    <pre
      className={cn(
        TEXT_CARD,
        "max-h-[32rem] overflow-auto leading-relaxed break-words whitespace-pre-wrap text-foreground",
        format === "code" ? "bg-muted/40 font-mono text-xs" : "font-sans text-sm",
      )}
    >
      {text}
    </pre>
  );
}

export function Payload({ view, name, failed }: { view: PayloadView; name: string; failed: boolean }) {
  switch (view.kind) {
    case "messages":
      return <MessageList messages={view.messages} />;
    case "tool-result":
      return <ToolResultCard name={name} result={view.text} failed={failed} />;
    case "fields":
      return (
        <Block
          icon={
            <BlockBadge tone="neutral">
              <Braces />
            </BlockBadge>
          }
          label="Fields"
          name="fields"
          meta={<span className="text-xs text-muted-foreground tabular-nums">{view.entries.length}</span>}
        >
          <FieldTree entries={view.entries} />
        </Block>
      );
    case "text":
      return <TextBody text={view.text} format={view.format} />;
  }
}
