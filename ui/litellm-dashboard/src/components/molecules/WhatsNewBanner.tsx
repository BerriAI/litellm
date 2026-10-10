import React from "react";
import { ArrowRight, Box, Sparkles, X, Zap, type LucideIcon } from "lucide-react";
import { z } from "zod";

import { storageKey, useStoredValue } from "@/lib/storage";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

export const HIDE_WHATS_NEW_BANNER_KEY = storageKey("local", "hideWhatsNewBanner", z.boolean(), false);

interface WhatsNewItem {
  readonly icon: LucideIcon;
  readonly title: string;
  readonly subtitle: string;
  readonly href: string;
  readonly publishedOn: string;
}

const LAUNCHES: readonly WhatsNewItem[] = [
  {
    icon: Box,
    title: "GPT-6.1 Sol",
    subtitle: "OpenAI",
    href: "https://docs.litellm.ai/blog/gpt_6_1_sol",
    publishedOn: "2026-09-29",
  },
  {
    icon: Box,
    title: "Claude Haiku 5.5",
    subtitle: "Anthropic",
    href: "https://docs.litellm.ai/blog/claude-haiku-5-5",
    publishedOn: "2026-10-07",
  },
  {
    icon: Box,
    title: "TypeSafe Jev",
    subtitle: "New provider",
    href: "https://docs.litellm.ai/blog/typesafe_jev",
    publishedOn: "2026-09-20",
  },
  {
    icon: Zap,
    title: "OpenAI ultrafast tier",
    subtitle: "service_tier: ultrafast",
    href: "https://docs.litellm.ai/docs/providers/openai/ultrafast",
    publishedOn: "2026-10-06",
  },
];

export const WHATS_NEW_ITEMS: readonly WhatsNewItem[] = [...LAUNCHES].sort((a, b) =>
  b.publishedOn.localeCompare(a.publishedOn),
);

const formatPublishedOn = (isoDate: string) =>
  new Date(`${isoDate}T00:00:00`).toLocaleDateString("en-US", { month: "short", day: "numeric" });

const WhatsNewBanner: React.FC = () => {
  const [dismissed, setDismissed] = useStoredValue(HIDE_WHATS_NEW_BANNER_KEY);

  if (dismissed) {
    return null;
  }

  return (
    <div className="mb-4 flex items-center gap-3">
      <div className="flex shrink-0 flex-col gap-0.5 pr-1">
        <span className="flex items-center gap-1.5 text-sm font-semibold">
          <Sparkles className="size-4" />
          What&apos;s new
        </span>
        <span className="text-xs text-muted-foreground">in LiteLLM</span>
      </div>
      <div className="flex min-w-0 flex-1 gap-2 overflow-x-auto [scrollbar-width:none]">
        {WHATS_NEW_ITEMS.map((item) => (
          <a key={item.href} href={item.href} target="_blank" rel="noopener noreferrer" className="min-w-0 flex-1">
            <Card className="min-w-[200px] cursor-pointer flex-row items-center gap-3 px-3 py-2 hover:ring-foreground/25">
              <div className="flex size-8 shrink-0 items-center justify-center rounded-md bg-muted">
                <item.icon className="size-4" />
              </div>
              <div className="min-w-0 flex-1">
                <div className="truncate text-sm font-medium leading-tight">{item.title}</div>
                <div className="truncate text-xs text-muted-foreground">
                  {item.subtitle} · {formatPublishedOn(item.publishedOn)}
                </div>
              </div>
              <ArrowRight className="size-3.5 shrink-0 text-muted-foreground" />
            </Card>
          </a>
        ))}
      </div>
      <Button
        type="button"
        variant="ghost"
        size="icon-sm"
        onClick={() => setDismissed(true)}
        className="shrink-0"
        aria-label="Dismiss what's new"
      >
        <X />
      </Button>
    </div>
  );
};

export default WhatsNewBanner;
