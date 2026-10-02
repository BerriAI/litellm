import anthropicLogo from "../../../../public/assets/logos/anthropic.svg";
import { Logo } from "@/components/molecules/logo/Logo";
import { cn } from "@/lib/cva.config";

import type { TraceSummary } from "./traceTypes";

export interface TraceFramework {
  readonly id: string;
  readonly label: string;
  readonly logo: string;
}

const FRAMEWORKS: readonly TraceFramework[] = [
  { id: "claude-agent-sdk", label: "Claude Agent SDK", logo: anthropicLogo.src },
  { id: "claude-code", label: "Claude Code", logo: anthropicLogo.src },
];

/**
 * The SDK that produced the trace. A Claude Agent SDK trace also carries "claude-code" (its root span has no
 * SDK marker), so registry order decides: the more specific framework wins.
 */
export const traceFramework = (summary: Pick<TraceSummary, "frameworks">): TraceFramework | null =>
  FRAMEWORKS.find((framework) => summary.frameworks?.includes(framework.id)) ?? null;

export function FrameworkLogo({ framework, className }: { framework: TraceFramework; className?: string }) {
  return (
    <span aria-hidden className="contents">
      <Logo src={framework.logo} label={framework.label} className={cn("size-3.5 shrink-0", className)} />
    </span>
  );
}
