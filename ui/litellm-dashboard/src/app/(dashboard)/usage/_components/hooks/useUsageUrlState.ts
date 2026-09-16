import type { DateRangePickerValue } from "@/components/shared/date_picker_types";
import moment from "moment";
import {
  createParser,
  parseAsArrayOf,
  parseAsNumberLiteral,
  parseAsString,
  parseAsStringLiteral,
  useQueryStates,
} from "nuqs";
import { useMemo, useState } from "react";
import type { ModelViewType } from "../components/ModelViewToggle";
import { USAGE_OPTIONS, type UsageOption } from "../components/UsageViewSelect/UsageViewSelect";

export const USAGE_TABS = ["cost", "models", "agents", "keys", "mcp", "endpoints"] as const;
export type UsageTab = (typeof USAGE_TABS)[number];
export type EntityUsageTab = Exclude<UsageTab, "mcp">;
export const GLOBAL_USAGE_TABS: readonly Exclude<UsageTab, "agents">[] = ["cost", "models", "keys", "mcp", "endpoints"];

export const USAGE_TOP_LIMITS = [5, 10, 25, 50] as const;
export type UsageTopLimit = (typeof USAGE_TOP_LIMITS)[number];
const toUsageTopLimit = (limit: number): UsageTopLimit | null =>
  USAGE_TOP_LIMITS.find((allowed) => allowed === limit) ?? null;

const MODEL_VIEWS: readonly ModelViewType[] = ["groups", "individual"];
const DAY_FORMAT = "YYYY-MM-DD";
const DEFAULT_RANGE_DAYS = 7;

const parseAsLocalDay = createParser<Date>({
  parse: (value) => {
    const day = moment(value, DAY_FORMAT, true);
    return day.isValid() ? day.toDate() : null;
  },
  serialize: (date) => moment(date).format(DAY_FORMAT),
  eq: (a, b) => a.getTime() === b.getTime(),
});

const USAGE_URL_PARSERS = {
  view: parseAsStringLiteral(USAGE_OPTIONS).withDefault("global"),
  from: parseAsLocalDay,
  to: parseAsLocalDay,
  user: parseAsString,
  model_view: parseAsStringLiteral(MODEL_VIEWS).withDefault("groups"),
  top_keys: parseAsNumberLiteral(USAGE_TOP_LIMITS).withDefault(5),
  top_models: parseAsNumberLiteral(USAGE_TOP_LIMITS).withDefault(5),
  top_agents: parseAsNumberLiteral(USAGE_TOP_LIMITS).withDefault(5),
  filter: parseAsArrayOf(parseAsString).withDefault([]),
};

const defaultUsageRange = (): { from: Date; to: Date } => ({
  from: moment().subtract(DEFAULT_RANGE_DAYS, "days").toDate(),
  to: new Date(),
});

export interface UsageUrlState {
  view: UsageOption;
  dateValue: DateRangePickerValue;
  user: string | null;
  modelView: ModelViewType;
  topKeys: UsageTopLimit;
  topModels: UsageTopLimit;
  topAgents: UsageTopLimit;
  filter: string[];
  setView: (view: UsageOption) => void;
  setDateValue: (value: DateRangePickerValue) => void;
  setUser: (user: string | null) => void;
  setModelView: (view: ModelViewType) => void;
  setTopKeys: (limit: number) => void;
  setTopModels: (limit: number) => void;
  setTopAgents: (limit: number) => void;
  setFilter: (filter: string[]) => void;
}

export function useUsageUrlState(): UsageUrlState {
  const [state, setState] = useQueryStates(USAGE_URL_PARSERS);
  const [fallbackRange] = useState(defaultUsageRange);

  const dateValue = useMemo<DateRangePickerValue>(
    () => ({
      from: state.from ? moment(state.from).startOf("day").toDate() : fallbackRange.from,
      to: state.to ? moment(state.to).endOf("day").toDate() : fallbackRange.to,
    }),
    [state.from, state.to, fallbackRange],
  );

  const actions = useMemo(
    () => ({
      setView: (view: UsageOption) =>
        void setState((previous) => (previous.view === view ? { view } : { view, filter: null })),
      setDateValue: (value: DateRangePickerValue) => void setState({ from: value.from ?? null, to: value.to ?? null }),
      setUser: (user: string | null) => void setState({ user }),
      setModelView: (view: ModelViewType) => void setState({ model_view: view }),
      setTopKeys: (limit: number) => void setState({ top_keys: toUsageTopLimit(limit) }),
      setTopModels: (limit: number) => void setState({ top_models: toUsageTopLimit(limit) }),
      setTopAgents: (limit: number) => void setState({ top_agents: toUsageTopLimit(limit) }),
      setFilter: (filter: string[]) => void setState({ filter }),
    }),
    [setState],
  );

  return {
    view: state.view,
    dateValue,
    user: state.user,
    modelView: state.model_view,
    topKeys: state.top_keys,
    topModels: state.top_models,
    topAgents: state.top_agents,
    filter: state.filter,
    ...actions,
  };
}
