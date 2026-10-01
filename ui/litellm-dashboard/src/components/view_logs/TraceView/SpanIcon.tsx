"use client";

import { Bot, BrainCircuit, Link2, Network, UserRound, Wrench } from "lucide-react";

import { Logo } from "@/components/molecules/logo/Logo";
import { cn } from "@/lib/cva.config";

import { useSpanProvider } from "./spanProvider";
import type { SpanType } from "./traceTypes";

const SIZE = { sm: "size-[18px] rounded-[4px]", md: "size-5 rounded-[5px]", lg: "size-[22px] rounded-md" } as const;
const GLYPH = { sm: "size-[11px]", md: "size-3", lg: "size-[13px]" } as const;

export type IconSize = keyof typeof SIZE;

interface SpanIconProps {
  type: SpanType;
  model?: string | null;
  error?: boolean;
  size?: IconSize;
}

const TYPE_GLYPH: Record<SpanType, typeof Bot> = {
  agent: Bot,
  llm: BrainCircuit,
  tool: Wrench,
  chain: Link2,
  framework: Network,
};

const tileTone = (type: SpanType, error: boolean): string => {
  if (error) return "bg-destructive text-white";
  if (type === "tool") return "bg-success text-success-foreground";
  if (type === "llm") return "border border-border bg-card text-muted-foreground";
  if (type === "framework") return "bg-muted-foreground text-background";
  return "bg-info text-info-foreground";
};

/** Solid square type tile; LLM spans show their provider's logo when we can resolve it. */
export function SpanIcon({ type, model = null, error = false, size = "md" }: SpanIconProps) {
  const provider = useSpanProvider(type === "llm" ? model : null);
  const Glyph = TYPE_GLYPH[type];
  return (
    <span
      className={cn("grid shrink-0 place-items-center", SIZE[size], tileTone(type, error))}
      data-testid="span-icon"
      data-provider={provider ?? undefined}
    >
      {provider && !error ? <Logo provider={provider} className={GLYPH[size]} /> : <Glyph className={GLYPH[size]} />}
    </span>
  );
}

/** Square tile for chat message roles: user / system / AI (provider logo) / tool. */
export function RoleTile({ role, model, error = false }: { role: string; model: string | null; error?: boolean }) {
  if (role === "assistant") return <SpanIcon type="llm" model={model} size="sm" />;
  if (role === "tool") return <SpanIcon type="tool" error={error} size="sm" />;
  const tone = role === "system" ? "bg-muted-foreground text-background" : "bg-trace-human text-white";
  return (
    <span className={cn("grid size-4 shrink-0 place-items-center rounded-[2px] p-0.5 text-[10px] font-semibold", tone)}>
      {role === "system" ? "S" : <UserRound className="size-3" />}
    </span>
  );
}
