import { useState } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import { Check } from "lucide-react";
import { copyToClipboard } from "@/utils/dataUtils";
import anthropicLogo from "@/../public/assets/logos/anthropic.svg";
import openaiLogo from "@/../public/assets/logos/openai_small.svg";
import { briefMarkdown } from "../model/findings";
import { type IssueBrief } from "../model/types";

const AGENTS = [
  { name: "Claude Code", logo: anthropicLogo.src },
  { name: "Codex", logo: openaiLogo.src },
] as const;

const COPIED_RESET_MS = 1500;

const markdown: Components = {
  h1: ({ children }) => <h1 className="mb-4 border-b border-border pb-2 text-base font-semibold">{children}</h1>,
  h2: ({ children }) => (
    <h2 className="mt-5 mb-1.5 text-xs font-semibold tracking-wide text-muted-foreground uppercase">{children}</h2>
  ),
  p: ({ children }) => <p className="text-sm leading-6">{children}</p>,
  ol: ({ children }) => (
    <ol className="list-decimal space-y-3 pl-5 text-sm leading-6 marker:text-muted-foreground">{children}</ol>
  ),
  li: ({ children }) => <li className="pl-1">{children}</li>,
  strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
  code: ({ children }) => <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">{children}</code>,
};

export function IssueBrief({ title, brief }: { title: string; brief: IssueBrief }) {
  const [copied, setCopied] = useState<string | null>(null);
  const source = briefMarkdown(title, brief);
  const copy = async (agent: string) => {
    if (await copyToClipboard(source, `Copied for ${agent}`)) {
      setCopied(agent);
      window.setTimeout(() => setCopied(null), COPIED_RESET_MS);
    }
  };
  return (
    <div className="overflow-hidden rounded-lg border border-border">
      <div className="flex h-10 items-center gap-1 border-b border-border bg-muted/40 px-3">
        <span className="font-mono text-xs text-muted-foreground">issue-brief.md</span>
        <span className="mr-1 ml-auto text-xs text-muted-foreground">Copy for</span>
        {AGENTS.map((agent) => (
          <button
            key={agent.name}
            type="button"
            onClick={() => void copy(agent.name)}
            aria-label={`Copy for ${agent.name}`}
            className="inline-flex h-7 items-center gap-1.5 rounded-md border border-border bg-background px-2 text-xs font-medium hover:bg-muted"
          >
            {copied === agent.name ? (
              <Check className="size-3.5 text-success" />
            ) : (
              <img src={agent.logo} alt="" aria-hidden className="size-3.5" />
            )}
            {agent.name}
          </button>
        ))}
      </div>
      <article className="max-h-[32rem] overflow-auto bg-background px-5 py-4">
        <ReactMarkdown components={markdown}>{source}</ReactMarkdown>
      </article>
    </div>
  );
}
