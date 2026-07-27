export const CAPABILITY_COLOR_PALETTE = [
  "blue",
  "green",
  "purple",
  "orange",
  "red",
  "yellow",
  "cyan",
  "indigo",
  "pink",
  "amber",
] as const;

export type CapabilityColor = (typeof CAPABILITY_COLOR_PALETTE)[number];

const SEMANTIC_CAPABILITY_COLORS: Record<string, CapabilityColor> = {
  function_calling: "blue",
  parallel_function_calling: "cyan",
  vision: "purple",
  reasoning: "orange",
  prompt_caching: "green",
  system_messages: "indigo",
  tool_choice: "pink",
  response_schema: "amber",
  audio_input: "yellow",
  audio_output: "yellow",
  pdf_input: "red",
  web_search: "green",
};

const CAPABILITY_BADGE_CLASSES: Record<CapabilityColor, string> = {
  blue: "border-transparent bg-blue-100 text-blue-800",
  green: "border-transparent bg-green-100 text-green-800",
  purple: "border-transparent bg-purple-100 text-purple-800",
  orange: "border-transparent bg-orange-100 text-orange-800",
  red: "border-transparent bg-red-100 text-red-800",
  yellow: "border-transparent bg-yellow-100 text-yellow-800",
  cyan: "border-transparent bg-cyan-100 text-cyan-800",
  indigo: "border-transparent bg-indigo-100 text-indigo-800",
  pink: "border-transparent bg-pink-100 text-pink-800",
  amber: "border-transparent bg-amber-100 text-amber-800",
};

const normalizeCapabilityKey = (capability: string): string =>
  capability
    .replace(/^supports_/, "")
    .trim()
    .toLowerCase()
    .replace(/[\s-]+/g, "_");

const hashCapabilityKey = (key: string): number => {
  let hash = 0;
  for (const ch of key) {
    hash = (hash * 31 + ch.charCodeAt(0)) >>> 0;
  }
  return hash;
};

export const getCapabilityColor = (capability: string): CapabilityColor => {
  const key = normalizeCapabilityKey(capability);
  const semantic = SEMANTIC_CAPABILITY_COLORS[key];
  if (semantic) {
    return semantic;
  }
  return CAPABILITY_COLOR_PALETTE[hashCapabilityKey(key) % CAPABILITY_COLOR_PALETTE.length];
};

export const getCapabilityBadgeClassName = (capability: string): string =>
  CAPABILITY_BADGE_CLASSES[getCapabilityColor(capability)];
