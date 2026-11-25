import type { ParseKeys } from "i18next";
import type { TelemetryGroup } from "@/app/(dashboard)/hooks/telemetry/useTelemetrySettings";

export interface GroupCopy {
  readonly title: ParseKeys;
  readonly slices?: ParseKeys;
  readonly counts: ParseKeys;
  readonly helps: ParseKeys;
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
    title: "telemetry.groups.heartbeat.title",
    counts: "telemetry.groups.heartbeat.counts",
    helps: "telemetry.groups.heartbeat.helps",
  },
  request_success: {
    title: "telemetry.groups.requestSuccess.title",
    slices: "telemetry.groups.requestSuccess.slices",
    counts: "telemetry.groups.requestSuccess.counts",
    helps: "telemetry.groups.requestSuccess.helps",
  },
  token_info: {
    title: "telemetry.groups.tokenInfo.title",
    slices: "telemetry.groups.tokenInfo.slices",
    counts: "telemetry.groups.tokenInfo.counts",
    helps: "telemetry.groups.tokenInfo.helps",
  },
  request_taxonomy: {
    title: "telemetry.groups.requestTaxonomy.title",
    slices: "telemetry.groups.requestTaxonomy.slices",
    counts: "telemetry.groups.requestTaxonomy.counts",
    helps: "telemetry.groups.requestTaxonomy.helps",
  },
  event_details: {
    title: "telemetry.groups.eventDetails.title",
    counts: "telemetry.groups.eventDetails.counts",
    helps: "telemetry.groups.eventDetails.helps",
  },
  instance_configuration: {
    title: "telemetry.groups.instanceConfiguration.title",
    counts: "telemetry.groups.instanceConfiguration.counts",
    helps: "telemetry.groups.instanceConfiguration.helps",
  },
  page_navigation: {
    title: "telemetry.groups.pageNavigation.title",
    counts: "telemetry.groups.pageNavigation.counts",
    helps: "telemetry.groups.pageNavigation.helps",
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
