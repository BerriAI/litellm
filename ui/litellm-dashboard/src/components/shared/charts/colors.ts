export const CHART_COLOR_HEX = {
  slate: "#64748b",
  gray: "#6b7280",
  zinc: "#71717a",
  neutral: "#737373",
  stone: "#78716c",
  red: "#ef4444",
  orange: "#f97316",
  amber: "#f59e0b",
  yellow: "#eab308",
  lime: "#84cc16",
  green: "#22c55e",
  emerald: "#10b981",
  teal: "#14b8a6",
  cyan: "#06b6d4",
  sky: "#0ea5e9",
  blue: "#3b82f6",
  indigo: "#6366f1",
  violet: "#8b5cf6",
  purple: "#a855f7",
  fuchsia: "#d946ef",
  pink: "#ec4899",
  rose: "#f43f5e",
} as const;

export type ChartColor = keyof typeof CHART_COLOR_HEX;

export const DEFAULT_COLOR_CYCLE: readonly ChartColor[] = [
  "blue",
  "cyan",
  "sky",
  "indigo",
  "violet",
  "purple",
  "fuchsia",
  "slate",
  "gray",
  "zinc",
  "neutral",
  "stone",
  "red",
  "orange",
  "amber",
  "yellow",
  "lime",
  "green",
  "emerald",
  "teal",
  "pink",
  "rose",
];

export const chartColorValue = (color: ChartColor): string => `var(--color-${color}-500, ${CHART_COLOR_HEX[color]})`;

export const categoryFills = (count: number, colors?: readonly ChartColor[]): readonly string[] => {
  const cycle = colors && colors.length > 0 ? colors : DEFAULT_COLOR_CYCLE;
  return Array.from({ length: count }, (_, i) => chartColorValue(cycle[i % cycle.length]));
};
