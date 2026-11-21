import type { SortingState } from "@tanstack/react-table";
import {
  customerBudgetLimit,
  customerBudgetPeriod,
  customerBudgetStatus,
  customerBudgetUsage,
} from "../../customers/_components/customerBudgetUsage";
import type { BudgetGroups } from "../../customers/_components/customerBudgetGroupsPlan";
import type { CustomerUsageTotal } from "../../customers/_components/customerUsageTotals";
import type { CustomerUsageEntity } from "../../customers/_components/customerUsageApi";
import { appUsageEntity } from "./appBudgetPresentation";
import type { ApplicationRow } from "./applications";

export const APPLICATION_COLUMNS = [
  { id: "application", label: "apps.list.columns.application", defaultDescending: false },
  { id: "group", label: "customerPortal.list.columns.tier", defaultDescending: false },
  { id: "spend", label: "customerPortal.list.columns.monthlySpend", defaultDescending: true },
  { id: "budget", label: "customerPortal.hierarchy.maxBudget", defaultDescending: false },
  { id: "share", label: "customerPortal.list.columns.overallSpendShare", defaultDescending: true },
  { id: "tokens", label: "customerPortal.list.columns.tokens", defaultDescending: true },
  { id: "status", label: "customerPortal.list.columns.status", defaultDescending: true },
] as const;

export const APPLICATION_TABLE = {
  sortFields: APPLICATION_COLUMNS.map(({ id }) => id),
  defaultSort: { id: "application", desc: false },
  defaultPageSize: 25,
  filterColumns: [],
};

const STATUS_ORDER = {
  "customerPortal.analytics.withinBudget": 0,
  "customerPortal.analytics.reachedLimit": 1,
  "customerPortal.analytics.noLimit": null,
  "customerPortal.analytics.budgetUsageUnavailable": null,
} as const;

function applicationSortValues(
  entity: CustomerUsageEntity,
  total: CustomerUsageTotal | null | undefined,
  overallSpend: number | null,
  customLabel: string | null,
) {
  const budgets = customerBudgetUsage(entity);
  const monthly = budgets.find((budget) => customerBudgetPeriod(budget.budget_duration) === "monthly");
  const custom = budgets.length > 0 || entity.rateLimits ? customLabel : null;
  const share = total && overallSpend ? total.spend / overallSpend : 0;
  return {
    application: entity.label,
    group: customLabel === null ? null : entity.budgetGroup ?? custom,
    spend: monthly ? monthly.spend : total?.spend ?? null,
    budget: customerBudgetLimit(entity),
    share: total && overallSpend !== null ? share : null,
    tokens: total?.tokens ?? null,
    status: STATUS_ORDER[customerBudgetStatus(budgets)],
  };
}

export function applicationTableRows(
  rows: readonly ApplicationRow[],
  groups: BudgetGroups["default_budgets"] | undefined,
  usage: ReadonlyMap<string | null, CustomerUsageTotal> | undefined,
  customLabel: string,
) {
  const names = new Map(groups?.map((group) => [group.budget_id, group.name]));
  const overallSpend = usage?.get(null)?.spend ?? null;
  return rows.map(({ resource, app }) => {
    const entity = {
      ...appUsageEntity(resource.resource_id, resource.alias, {
        spend: app?.spend ?? 0,
        effective_budget: app?.budget,
        budget_usage: app?.budget_usage,
      }),
      budgetGroup: groups ? names.get(app?.budget_id ?? "") : "—",
    };
    const total = app ? usage?.get(app.app_id) : null;
    return {
      resource,
      app,
      entity,
      total,
      values: applicationSortValues(entity, total, overallSpend, groups ? customLabel : null),
    };
  });
}

type ApplicationTableRow = ReturnType<typeof applicationTableRows>[number];

const compareValues = (left: string | number | null, right: string | number | null, descending: boolean) => {
  if (left === null && right === null) return 0;
  if (left === null) return 1;
  if (right === null) return -1;
  const comparison =
    typeof left === "string" && typeof right === "string" ? left.localeCompare(right) : Number(left) - Number(right);
  return comparison * (descending ? -1 : 1);
};

export function sortApplications(rows: readonly ApplicationTableRow[], search: string, sorting: SortingState) {
  const query = search.trim().toLocaleLowerCase();
  const active = sorting[0] ?? APPLICATION_TABLE.defaultSort;
  const column = APPLICATION_COLUMNS.find(({ id }) => id === active.id) ?? APPLICATION_COLUMNS[0];
  return rows
    .filter(({ resource }) =>
      [resource.alias, resource.resource_id].some((value) => value.toLocaleLowerCase().includes(query)),
    )
    .sort(
      (left, right) =>
        compareValues(left.values[column.id], right.values[column.id], active.desc) ||
        left.resource.alias.localeCompare(right.resource.alias) ||
        left.resource.resource_id.localeCompare(right.resource.resource_id),
    );
}
