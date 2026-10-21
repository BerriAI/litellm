"use client";

import {
  ArrowDownUp,
  ClipboardCheck,
  FileText,
  Layers,
  Link2,
  MessageSquareText,
  Network,
  Search,
  ShieldCheck,
  Split,
  Wrench,
} from "lucide-react";

import { Logo } from "@/components/molecules/logo/Logo";
import { cn } from "@/lib/cva.config";

import { useSpanProvider } from "./spanProvider";
import type { SpanType } from "../types";

export type SpanFamily = "orchestration" | "model" | "tool";

export const SPAN_FAMILY: Record<SpanType, SpanFamily> = {
  agent: "orchestration",
  chain: "orchestration",
  framework: "orchestration",
  decision: "orchestration",
  llm: "model",
  embedding: "model",
  reranker: "model",
  prompt: "model",
  tool: "tool",
  retriever: "tool",
  guardrail: "tool",
  evaluator: "tool",
};

const FAMILY_TILE: Record<SpanFamily, string> = {
  orchestration: "bg-trace-chain-soft text-trace-chain",
  model: "bg-trace-llm-soft text-trace-llm",
  tool: "bg-trace-tool-soft text-trace-tool",
};

export const FAMILY_BAR: Record<SpanFamily, string> = {
  orchestration: "bg-trace-chain",
  model: "bg-trace-llm",
  tool: "bg-trace-tool",
};

const SIZE = {
  sm: "size-4 rounded-sm",
  md: "size-5 rounded-md",
  lg: "size-6 rounded-md",
} as const;
const GLYPH = { sm: "size-3", md: "size-3.5", lg: "size-4" } as const;
const FAILED_TILE = "bg-destructive/10 text-destructive";
const LOGO_TILE = "bg-white ring-1 ring-border dark:bg-white";

export type IconSize = keyof typeof SIZE;

interface SpanIconProps {
  type: SpanType;
  model?: string | null;
  error?: boolean;
  size?: IconSize;
}

const TYPE_GLYPH: Record<SpanType, typeof Network> = {
  agent: Network,
  llm: MessageSquareText,
  tool: Wrench,
  chain: Link2,
  framework: Network,
  retriever: Search,
  embedding: Layers,
  reranker: ArrowDownUp,
  guardrail: ShieldCheck,
  evaluator: ClipboardCheck,
  prompt: FileText,
  decision: Split,
};

const tileTone = (type: SpanType, error: boolean, logo: boolean): string => {
  if (error) return FAILED_TILE;
  if (logo) return LOGO_TILE;
  return FAMILY_TILE[SPAN_FAMILY[type]];
};

export function SpanIcon({ type, model = null, error = false, size = "md" }: SpanIconProps) {
  const provider = useSpanProvider(type === "llm" ? model : null);
  const Glyph = TYPE_GLYPH[type];
  const showLogo = provider !== null && !error;
  return (
    <span
      className={cn("grid shrink-0 place-items-center", SIZE[size], tileTone(type, error, showLogo))}
      data-testid="span-icon"
      data-provider={provider ?? undefined}
    >
      {showLogo ? <Logo provider={provider} className={GLYPH[size]} /> : <Glyph className={GLYPH[size]} />}
    </span>
  );
}
