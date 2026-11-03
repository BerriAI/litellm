import type { TelemetryGroup } from "@/app/(dashboard)/hooks/telemetry/useTelemetrySettings";

export interface GroupCopy {
  readonly title: string;
  readonly slices?: string;
  readonly counts: string;
  readonly helps: string;
}

export const PROXY_GROUPS: readonly TelemetryGroup[] = [
  "heartbeat",
  "request_success",
  "token_info",
  "request_taxonomy",
  "event_details",
  "instance_configuration",
];

export const UI_GROUPS: readonly TelemetryGroup[] = ["page_navigation"];

export const GROUP_COPY: Readonly<Record<TelemetryGroup, GroupCopy>> = {
  heartbeat: {
    title: "Heartbeat",
    counts: "Random instance id, LiteLLM version, report window, which groups are on",
    helps: "Which versions are running",
  },
  request_success: {
    title: "Request success",
    slices: "endpoint, LiteLLM status, provider status, stream, handled by Rust, LiteLLM cache hit",
    counts: "requests, provider attempts, latency to headers and to first byte",
    helps: "Error-rate and latency regressions, Rust vs Python",
  },
  token_info: {
    title: "Token info",
    slices: "provider prompt-cache hit",
    counts: "input, output and cache-read token sums",
    helps: "Broken prompt caching or token counting",
  },
  request_taxonomy: {
    title: "Request taxonomy",
    slices: "provider, salted deployment hash",
    counts: "one row per provider attempt with its status",
    helps: "Which provider is failing, retries and fallbacks",
  },
  event_details: {
    title: "Event details",
    counts: "message block counts and types, allowlisted header names (no values), provider time to first token",
    helps: "Which request shapes and clients fail",
  },
  instance_configuration: {
    title: "Instance configuration",
    counts: "names of allowlisted config keys that are set (no values)",
    helps: "Which features an upgrade must keep working",
  },
  page_navigation: {
    title: "Page navigation",
    counts: "Admin UI page views and tab switches by route name (no ids or typed text)",
    helps: "Which pages and tabs get used",
  },
};

export type Requires = ReadonlyMap<TelemetryGroup, TelemetryGroup | null>;

const dependents = (group: TelemetryGroup, requires: Requires): readonly TelemetryGroup[] =>
  [...requires.entries()]
    .filter(([, parent]) => parent === group)
    .flatMap(([child]) => [child, ...dependents(child, requires)]);

export const depthOf = (group: TelemetryGroup, requires: Requires): number => {
  const parent = requires.get(group) ?? null;
  return parent === null ? 0 : 1 + depthOf(parent, requires);
};

export const relativeDepths = (
  groups: readonly TelemetryGroup[],
  requires: Requires,
): ReadonlyMap<TelemetryGroup, number> => {
  const depths = groups.map((group) => [group, depthOf(group, requires)] as const);
  const shallowest = Math.min(...depths.map(([, depth]) => depth));
  return new Map(depths.map(([group, depth]) => [group, depth - shallowest]));
};

const withAncestors = (group: TelemetryGroup, requires: Requires): readonly TelemetryGroup[] => {
  const parent = requires.get(group) ?? null;
  return parent === null ? [group] : [group, ...withAncestors(parent, requires)];
};

export const toggleGroup = (
  group: TelemetryGroup,
  on: boolean,
  enabled: ReadonlySet<TelemetryGroup>,
  requires: Requires,
): ReadonlySet<TelemetryGroup> => {
  if (on) return new Set([...enabled, ...withAncestors(group, requires)]);
  const removed = new Set([group, ...dependents(group, requires)]);
  return new Set([...enabled].filter((g) => !removed.has(g)));
};
