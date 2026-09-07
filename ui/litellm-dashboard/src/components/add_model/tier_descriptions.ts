import type { ComplexityTierLabels, ComplexityTiers } from "./ComplexityRouterConfig";

export const TIER_DESCRIPTIONS: Record<
  keyof ComplexityTiers,
  { label: string; description: string; examples: string }
> = {
  SIMPLE: {
    label: "Simple",
    description: "Basic questions, greetings, simple factual queries",
    examples: '"Hello!", "What is Python?", "Thanks!"',
  },
  MEDIUM: {
    label: "Medium",
    description: "Standard queries requiring some reasoning or explanation",
    examples: '"Explain how REST APIs work", "Debug this error"',
  },
  COMPLEX: {
    label: "Complex",
    description: "Technical, multi-part requests requiring deep knowledge",
    examples: '"Design a microservices architecture", "Implement a rate limiter"',
  },
  REASONING: {
    label: "Reasoning",
    description: "Chain-of-thought, analysis, explicit reasoning requests",
    examples: '"Think step by step...", "Analyze the pros and cons..."',
  },
};

export const TIER_KEYS = Object.keys(TIER_DESCRIPTIONS) as Array<keyof ComplexityTiers>;

export const effectiveTierLabel = (tier: keyof ComplexityTiers, tierLabels: ComplexityTierLabels | undefined): string =>
  tierLabels?.[tier]?.trim() || TIER_DESCRIPTIONS[tier].label;
