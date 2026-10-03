import ReactMarkdown, { type Components } from "react-markdown";
import { Copy } from "lucide-react";
import { Button } from "@/components/ui/button";
import { copyToClipboard } from "@/utils/dataUtils";
import anthropicLogo from "../../../../../public/assets/logos/anthropic.svg";
import openaiLogo from "../../../../../public/assets/logos/openai_small.svg";
import { fixPrompt, fixSummary, type AgentFix } from "./lensData";

const markdown: Components = {
  h1: ({ children }) => <h3 className="mb-3 text-lg font-semibold">{children}</h3>,
  h2: ({ children }) => <h4 className="mt-4 mb-2 text-sm font-semibold">{children}</h4>,
  h3: ({ children }) => <h4 className="mt-4 mb-2 text-sm font-semibold">{children}</h4>,
  p: ({ children }) => <p className="mb-2 text-sm leading-6 whitespace-pre-wrap">{children}</p>,
  ul: ({ children }) => <ul className="mb-2 list-disc space-y-2 pl-5 text-sm leading-6">{children}</ul>,
  strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
  code: ({ children }) => <code className="rounded-sm bg-muted px-1 py-0.5 font-mono text-xs">{children}</code>,
};

export function LensAgentFix({ title, fix }: { title: string; fix: AgentFix }) {
  return (
    <div className="space-y-4 border-y py-4">
      <div>
        <h3 className="text-lg font-semibold">How to fix your agent</h3>
        <p className="mt-1 text-xs text-muted-foreground">
          Pick a fix and paste its prompt into the coding agent that owns this agent.
        </p>
      </div>
      <div className="rounded-lg bg-muted/40 p-4">
        <ReactMarkdown components={markdown}>{fixSummary(fix)}</ReactMarkdown>
      </div>
      {fix.options.map((option, index) => (
        <details key={option.title} className="group rounded-lg border p-4" open={index === 0}>
          <summary className="cursor-pointer text-base font-semibold">
            Option {index + 1}: {option.title}
          </summary>
          <div className="mt-3">
            <ReactMarkdown components={markdown}>{option.change}</ReactMarkdown>
          </div>
          <Button
            className="mt-3"
            variant="outline"
            aria-label={`Copy option ${index + 1} prompt for Claude Code or Codex`}
            onClick={() => copyToClipboard(fixPrompt(title, fix, option), "Prompt copied")}
          >
            <Copy className="size-4" />
            Copy prompt for
            <img src={anthropicLogo.src} alt="" aria-hidden className="size-4" />
            Claude Code or
            <img src={openaiLogo.src} alt="" aria-hidden className="size-4" />
            Codex
          </Button>
        </details>
      ))}
    </div>
  );
}
