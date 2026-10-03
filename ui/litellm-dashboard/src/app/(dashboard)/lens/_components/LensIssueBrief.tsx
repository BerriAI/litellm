import { useState } from "react";
import { Check } from "lucide-react";
import { copyToClipboard } from "@/utils/dataUtils";
import anthropicLogo from "../../../../../public/assets/logos/anthropic.svg";
import openaiLogo from "../../../../../public/assets/logos/openai_small.svg";
import { briefMarkdown, type IssueBrief } from "./lensData";

const AGENTS = [
  { name: "Claude Code", logo: anthropicLogo.src },
  { name: "Codex", logo: openaiLogo.src },
] as const;

const COPIED_RESET_MS = 1500;

export function LensIssueBrief({ title, brief }: { title: string; brief: IssueBrief }) {
  const [copied, setCopied] = useState<string | null>(null);
  const markdown = briefMarkdown(title, brief);
  const copy = async (agent: string) => {
    if (await copyToClipboard(markdown, `Copied for ${agent}`)) {
      setCopied(agent);
      window.setTimeout(() => setCopied(null), COPIED_RESET_MS);
    }
  };
  return (
    <div className="overflow-hidden rounded-lg border border-border bg-muted/30">
      <div className="flex h-10 items-center gap-1 border-b border-border px-3">
        <span className="font-mono text-xs text-muted-foreground">issue-brief.md</span>
        <span className="ml-auto mr-1 text-xs text-muted-foreground">Copy for</span>
        {AGENTS.map((agent) => (
          <button
            key={agent.name}
            type="button"
            onClick={() => void copy(agent.name)}
            aria-label={`Copy for ${agent.name}`}
            className="inline-flex h-7 items-center gap-1.5 rounded-md border border-border bg-background px-2 text-xs font-medium hover:bg-muted"
          >
            {copied === agent.name ? (
              <Check className="size-3.5 text-emerald-600" />
            ) : (
              <img src={agent.logo} alt="" aria-hidden className="size-3.5" />
            )}
            {agent.name}
          </button>
        ))}
      </div>
      <pre className="m-0 max-h-[28rem] overflow-auto px-4 py-3 font-mono text-xs leading-5 whitespace-pre-wrap break-words text-foreground">
        <code>{markdown}</code>
      </pre>
    </div>
  );
}
