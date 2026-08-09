/** Stable Tremor badge colors for AI Hub capability pills. */
export const CAPABILITY_BADGE_COLORS = [
  "blue",
  "purple",
  "orange",
  "green",
  "rose",
  "amber",
  "cyan",
  "indigo",
] as const;

export type CapabilityBadgeColor = (typeof CAPABILITY_BADGE_COLORS)[number];

/** Preferred colors for well-known capabilities (stable across rows). */
const KNOWN_CAPABILITY_COLORS: Record<string, CapabilityBadgeColor> = {
  supports_function_calling: "blue",
  supports_parallel_function_calling: "indigo",
  supports_vision: "purple",
  supports_prompt_caching: "green",
  supports_response_schema: "cyan",
  supports_system_messages: "amber",
  supports_tool_choice: "blue",
  supports_assistant_prefill: "orange",
  supports_audio_input: "rose",
  supports_audio_output: "rose",
  supports_pdf_input: "orange",
  supports_reasoning: "amber",
  supports_web_search: "cyan",
};

function hashCapability(key: string): number {
  let hash = 0;
  for (let i = 0; i < key.length; i++) {
    hash = (hash * 31 + key.charCodeAt(i)) >>> 0;
  }
  return hash;
}

/** Same capability key always maps to the same color (not position-based). */
export function getCapabilityBadgeColor(capabilityKey: string): CapabilityBadgeColor {
  const known = KNOWN_CAPABILITY_COLORS[capabilityKey];
  if (known) {
    return known;
  }
  return CAPABILITY_BADGE_COLORS[hashCapability(capabilityKey) % CAPABILITY_BADGE_COLORS.length];
}

/** Tailwind classes for shadcn/ui Badge when Tremor color props are unavailable. */
export const CAPABILITY_BADGE_CLASS: Record<CapabilityBadgeColor, string> = {
  blue: "border-blue-200 bg-blue-50 text-blue-800",
  purple: "border-purple-200 bg-purple-50 text-purple-800",
  orange: "border-orange-200 bg-orange-50 text-orange-800",
  green: "border-green-200 bg-green-50 text-green-800",
  rose: "border-rose-200 bg-rose-50 text-rose-800",
  amber: "border-amber-200 bg-amber-50 text-amber-900",
  cyan: "border-cyan-200 bg-cyan-50 text-cyan-800",
  indigo: "border-indigo-200 bg-indigo-50 text-indigo-800",
};

export function getCapabilityBadgeClassName(capabilityKey: string): string {
  return CAPABILITY_BADGE_CLASS[getCapabilityBadgeColor(capabilityKey)];
}
