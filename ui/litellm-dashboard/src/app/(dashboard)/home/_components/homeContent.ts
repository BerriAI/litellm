import { Box, Scale, Zap, type LucideIcon } from "lucide-react";

export interface Launch {
  readonly icon: LucideIcon;
  readonly title: string;
  readonly description: string;
  readonly href: string;
  readonly publishedOn: string;
}

const LAUNCHES: readonly Launch[] = [
  {
    icon: Box,
    title: "Claude Haiku 5.5",
    description: "Anthropic's newest small model, with day 0 pricing on Anthropic, Bedrock and Vertex AI.",
    href: "https://docs.litellm.ai/blog/claude-haiku-5-5",
    publishedOn: "2026-10-07",
  },
  {
    icon: Scale,
    title: "Decision Models",
    description: "Call decision models at /v1/decisions and try them in the Decisions playground.",
    href: "https://docs.litellm.ai/docs/decisions",
    publishedOn: "2026-10-07",
  },
  {
    icon: Zap,
    title: "OpenAI Ultrafast",
    description: "service_tier: ultrafast on GPT-6 Astra and GPT-6.1 Sol, including /ultrafast in Codex.",
    href: "https://docs.litellm.ai/docs/providers/openai/ultrafast",
    publishedOn: "2026-10-06",
  },
];

export const WHATS_NEW_ITEMS: readonly Launch[] = [...LAUNCHES].sort((a, b) =>
  b.publishedOn.localeCompare(a.publishedOn),
);

export const formatPublishedOn = (isoDate: string): string =>
  new Date(`${isoDate}T00:00:00`).toLocaleDateString("en-US", { month: "short", day: "numeric" });
