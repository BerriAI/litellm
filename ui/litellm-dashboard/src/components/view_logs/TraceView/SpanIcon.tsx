"use client";

import { Bot, BrainCircuit, Link2, Network, Wrench } from "lucide-react";

import { Logo } from "@/components/molecules/logo/Logo";
import { cn } from "@/lib/cva.config";

import { useSpanProvider } from "./spanProvider";
import type { SpanType } from "./traceTypes";

const SIZE = {
  sm: "size-4 rounded-[2px]",
  card: "size-[18px] rounded-[3px]",
  md: "size-5 rounded-full",
  lg: "size-5 rounded-full",
} as const;
const GLYPH = { sm: "size-3", card: "size-3", md: "size-3", lg: "size-3" } as const;
const LOGO_TILE = "bg-white ring-1 ring-trace-line ring-inset dark:bg-white";

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
  if (type === "tool") return "bg-trace-tool text-trace-glyph";
  if (type === "llm") return "bg-trace-llm text-trace-glyph";
  if (type === "framework") return "bg-trace-key text-trace-glyph";
  return "bg-trace-chain text-trace-glyph";
};

/** Solid square type tile; LLM spans show their provider's logo, knocked out to the tile glyph color. */
export function SpanIcon({ type, model = null, error = false, size = "md" }: SpanIconProps) {
  const provider = useSpanProvider(type === "llm" ? model : null);
  const Glyph = TYPE_GLYPH[type];
  const showLogo = provider !== null && !error;
  return (
    <span
      className={cn("grid shrink-0 place-items-center", SIZE[size], showLogo ? LOGO_TILE : tileTone(type, error))}
      data-testid="span-icon"
      data-provider={provider ?? undefined}
    >
      {showLogo ? <Logo provider={provider} className={GLYPH[size]} /> : <Glyph className={GLYPH[size]} />}
    </span>
  );
}
