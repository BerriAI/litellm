import { parseAsNativeArrayOf, parseAsString, type inferParserType } from "nuqs";

import type { EntityType } from "@/components/EntityUsageExport/types";
import { relativeTimeOptions } from "@/components/shared/advanced_date_picker";
import type { DateRangePickerValue } from "@/components/shared/date_picker_types";
import { uiHref } from "@/utils/uiHref";
import { USAGE_OPTIONS, type UsageOption } from "./components/UsageViewSelect/UsageViewSelect";

export const DEFAULT_USAGE_VIEW: UsageOption = "global";
export const USAGE_TABS = ["cost", "models", "keys", "mcp", "endpoints"] as const;
export type UsageTab = (typeof USAGE_TABS)[number];
export const DEFAULT_USAGE_TAB: UsageTab = "cost";

const entityIds = parseAsNativeArrayOf(parseAsString);

export const usageUrlParsers = {
  view: parseAsString,
  tab: parseAsString,
  range: parseAsString,
  from: parseAsString,
  to: parseAsString,
  user: parseAsString,
  team: entityIds,
  org: entityIds,
  customer: entityIds,
  tag: entityIds,
  agent: entityIds,
};

export type UsageUrlParams = inferParserType<typeof usageUrlParsers>;
export type UsageUrlPatch = { readonly [K in keyof UsageUrlParams]?: UsageUrlParams[K] | null };

type MultiEntityType = Exclude<EntityType, "user">;
type EntityKey = "user" | (typeof MULTI_ENTITY_KEYS)[MultiEntityType];

const MULTI_ENTITY_KEYS = {
  organization: "org",
  team: "team",
  customer: "customer",
  tag: "tag",
  agent: "agent",
} as const satisfies Record<MultiEntityType, keyof UsageUrlParams>;

const VIEW_ENTITY_KEYS: Readonly<Record<UsageOption, readonly EntityKey[]>> = {
  global: ["user"],
  "my-usage": [],
  organization: ["org"],
  team: ["team"],
  customer: ["customer"],
  tag: ["tag"],
  agent: ["agent"],
  user: ["user"],
  "user-agent-activity": [],
};

const TABBED_VIEWS: readonly UsageOption[] = ["global", "my-usage"];

const RESET_TO_DEFAULT_VIEW: UsageUrlPatch = {
  view: null,
  tab: null,
  user: null,
  team: null,
  org: null,
  customer: null,
  tag: null,
  agent: null,
};

const isUsageOption = (value: string | null): value is UsageOption => USAGE_OPTIONS.some((option) => option === value);

const isUsageTab = (value: string | null): value is UsageTab => USAGE_TABS.some((tab) => tab === value);

const hasEntityValue = (params: UsageUrlParams, key: EntityKey): boolean =>
  key === "user" ? params.user !== null : params[key].length > 0;

export const usageViewFromParams = (params: UsageUrlParams): UsageOption =>
  isUsageOption(params.view) ? params.view : DEFAULT_USAGE_VIEW;

export const usageTabFromParams = (params: UsageUrlParams): UsageTab =>
  isUsageTab(params.tab) ? params.tab : DEFAULT_USAGE_TAB;

export const usageViewPatch = (view: UsageOption): UsageUrlPatch => ({
  ...RESET_TO_DEFAULT_VIEW,
  view: view === DEFAULT_USAGE_VIEW ? null : view,
});

export const usageTabPatch = (tab: string): UsageUrlPatch => ({
  tab: isUsageTab(tab) && tab !== DEFAULT_USAGE_TAB ? tab : null,
});

export const selectedEntitiesFromParams = (params: UsageUrlParams, entityType: EntityType): readonly string[] => {
  if (entityType === "user") return params.user === null ? [] : [params.user];
  return params[MULTI_ENTITY_KEYS[entityType]];
};

export const entitySelectionPatch = (entityType: EntityType, ids: readonly string[]): UsageUrlPatch => {
  if (entityType === "user") return { user: ids[0] ?? null };
  return { [MULTI_ENTITY_KEYS[entityType]]: ids.length > 0 ? [...ids] : null };
};

const parseIsoDate = (value: string): Date | undefined => {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? undefined : date;
};

const customRangeFromParams = (from: string, to: string): DateRangePickerValue | undefined => {
  const fromDate = parseIsoDate(from);
  const toDate = parseIsoDate(to);
  if (!fromDate || !toDate || fromDate > toDate) return undefined;
  return { from: fromDate, to: toDate };
};

type DateParams = Pick<UsageUrlParams, "range" | "from" | "to">;

export const dateRangeFromParams = (params: DateParams): DateRangePickerValue | undefined => {
  if (params.range !== null) {
    return relativeTimeOptions.find((option) => option.shortLabel === params.range)?.getValue();
  }
  if (params.from === null || params.to === null) return undefined;
  return customRangeFromParams(params.from, params.to);
};

export const dateRangePatch = (value: DateRangePickerValue, presetShortLabel: string | null): UsageUrlPatch => {
  if (presetShortLabel !== null && relativeTimeOptions.some((option) => option.shortLabel === presetShortLabel)) {
    return { range: presetShortLabel, from: null, to: null };
  }
  if (!value.from || !value.to) return { range: null, from: null, to: null };
  return { range: null, from: value.from.toISOString(), to: value.to.toISOString() };
};

const invalidDatePatch = (params: DateParams): UsageUrlPatch => {
  const hasCustomRange = params.from !== null || params.to !== null;
  if (params.range !== null) {
    const rangeIsPreset = relativeTimeOptions.some((option) => option.shortLabel === params.range);
    return {
      ...(rangeIsPreset ? {} : { range: null }),
      ...(hasCustomRange ? { from: null, to: null } : {}),
    };
  }
  if (!hasCustomRange || dateRangeFromParams(params)) return {};
  return { from: null, to: null };
};

export interface UsageUrlAccess {
  readonly allowedViews: readonly UsageOption[];
  readonly isAdmin: boolean;
}

export interface UsageUrlCleanup {
  readonly patch: UsageUrlPatch;
  readonly deniedAccess: boolean;
}

export const cleanUsageUrl = (params: UsageUrlParams, access: UsageUrlAccess): UsageUrlCleanup | null => {
  const view = usageViewFromParams(params);
  const deniedView = !access.allowedViews.includes(view);
  const deniedUserFilter = view === "global" && params.user !== null && !access.isAdmin;
  if (deniedView || deniedUserFilter) {
    return { patch: { ...RESET_TO_DEFAULT_VIEW, ...invalidDatePatch(params) }, deniedAccess: true };
  }

  const viewKeys = VIEW_ENTITY_KEYS[view];
  const strayEntityKeys = (["user", "team", "org", "customer", "tag", "agent"] as const).filter(
    (key) => !viewKeys.includes(key) && hasEntityValue(params, key),
  );
  const invalidTab = params.tab !== null && (!TABBED_VIEWS.includes(view) || !isUsageTab(params.tab));
  const patch: UsageUrlPatch = {
    ...(params.view !== null && !isUsageOption(params.view) ? { view: null } : {}),
    ...(invalidTab ? { tab: null } : {}),
    ...Object.fromEntries(strayEntityKeys.map((key) => [key, null])),
    ...invalidDatePatch(params),
  };
  return Object.keys(patch).length > 0 ? { patch, deniedAccess: false } : null;
};

export const usageHrefForUser = (userId: string): string => uiHref(`usage?${new URLSearchParams({ user: userId })}`);

export const userRecordHref = (userId: string): string => uiHref(`users?${new URLSearchParams({ user: userId })}`);
