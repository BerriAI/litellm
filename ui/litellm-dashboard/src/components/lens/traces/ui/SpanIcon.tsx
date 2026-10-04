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

const SIZE = {
  sm: "size-4 rounded-sm",
  card: "size-[18px] rounded-sm",
  md: "size-5 rounded-full",
  lg: "size-5 rounded-full",
} as const;
const GLYPH = { sm: "size-3", card: "size-3", md: "size-3", lg: "size-3" } as const;
const LOGO_TILE = "bg-white dark:bg-white";

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

const tileTone = (error: boolean): string => (error ? "text-destructive" : "text-muted-foreground");

export function SpanIcon({ type, model = null, error = false, size = "md" }: SpanIconProps) {
  const provider = useSpanProvider(type === "llm" ? model : null);
  const Glyph = TYPE_GLYPH[type];
  const showLogo = provider !== null && !error;
  return (
    <span
      className={cn("grid shrink-0 place-items-center", SIZE[size], showLogo ? LOGO_TILE : tileTone(error))}
      data-testid="span-icon"
      data-provider={provider ?? undefined}
    >
      {showLogo ? <Logo provider={provider} className={GLYPH[size]} /> : <Glyph className={GLYPH[size]} />}
    </span>
  );
}
