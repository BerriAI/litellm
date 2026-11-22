import { createSerializer } from "nuqs";
import { uiHref } from "@/utils/uiHref";
import { customerAppParsers } from "../../customers/_components/CustomerHierarchyNavigation";
import type { App, AppResource } from "./appForms";
import type { AppGroup } from "./appGroupApi";
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

export type ApplicationSpendPeriod = "monthly" | "selected";

function applicationSortValues(
  entity: CustomerUsageEntity,
  total: CustomerUsageTotal | null | undefined,
  {
    overallSpend,
    customLabel,
    spendPeriod,
  }: { overallSpend: number | null; customLabel: string | null; spendPeriod: ApplicationSpendPeriod },
) {
  const budgets = customerBudgetUsage(entity);
  const monthly = budgets.find((budget) => customerBudgetPeriod(budget.budget_duration) === "monthly");
  const custom = budgets.length > 0 || entity.rateLimits ? customLabel : null;
  const share = total && overallSpend ? total.spend / overallSpend : 0;
  return {
    application: entity.label,
    group: customLabel === null ? null : entity.budgetGroup ?? custom,
    spend: spendPeriod === "monthly" && monthly ? monthly.spend : total?.spend ?? null,
    budget: customerBudgetLimit(entity),
    share: total && overallSpend !== null ? share : null,
    tokens: total?.tokens ?? null,
    status: STATUS_ORDER[customerBudgetStatus(budgets)],
  };
}

const usageHref = createSerializer(customerAppParsers);

function budgetUsageRow(
  source: {
    id: string;
    label: string;
    record: Pick<App, "spend" | "budget" | "budget_id" | "budget_usage"> | null;
    usageId: string | null;
  },
  groups: BudgetGroups["default_budgets"] | undefined,
  usage: ReadonlyMap<string | null, CustomerUsageTotal> | undefined,
  { customLabel, spendPeriod = "monthly" }: { customLabel: string; spendPeriod?: ApplicationSpendPeriod },
) {
  const { id, label, record, usageId } = source;
  const entity = {
    ...appUsageEntity(id, label, {
      spend: record?.spend ?? 0,
      effective_budget: record?.budget,
      budget_usage: record?.budget_usage,
    }),
    budgetGroup: groups ? groups.find((group) => group.budget_id === record?.budget_id)?.name : "—",
  };
  const total = usageId ? usage?.get(usageId) : null;
  return {
    entity,
    total,
    values: applicationSortValues(entity, total, {
      overallSpend: usage?.get(null)?.spend ?? null,
      customLabel: groups ? customLabel : null,
      spendPeriod,
    }),
  };
}

export interface ApplicationUsageTableRow extends ReturnType<typeof budgetUsageRow> {
  href?: string;
  description?: string | null;
  resourceType?: AppResource["resource_type"];
}

export function applicationTableRows(
  rows: readonly ApplicationRow[],
  groups: BudgetGroups["default_budgets"] | undefined,
  usage: ReadonlyMap<string | null, CustomerUsageTotal> | undefined,
  options: { customLabel: string; spendPeriod?: ApplicationSpendPeriod },
) {
  return rows.map(({ resource, app }) => {
    const source = { id: resource.resource_id, label: resource.alias, record: app, usageId: app?.app_id ?? null };
    return {
      resource,
      app,
      ...budgetUsageRow(source, groups, usage, options),
      resourceType: resource.resource_type,
      href: app ? usageHref(uiHref("apps/usage"), { app_id: app.app_id }) : undefined,
    };
  });
}

export function appGroupTableRows(
  rows: readonly AppGroup[],
  groups: BudgetGroups["default_budgets"] | undefined,
  usage: ReadonlyMap<string | null, CustomerUsageTotal> | undefined,
  customLabel: string,
): ApplicationUsageTableRow[] {
  return rows.map((group) => {
    const source = { id: group.app_group_id, label: group.name, record: group, usageId: group.app_group_id };
    return {
      ...budgetUsageRow(source, groups, usage, { customLabel }),
      description: group.description,
    };
  });
}

const compareValues = (left: string | number | null, right: string | number | null, descending: boolean) => {
  if (left === null && right === null) return 0;
  if (left === null) return 1;
  if (right === null) return -1;
  const comparison =
    typeof left === "string" && typeof right === "string" ? left.localeCompare(right) : Number(left) - Number(right);
  return comparison * (descending ? -1 : 1);
};

export function sortApplications<T extends ApplicationUsageTableRow>(
  rows: readonly T[],
  search: string,
  sorting: SortingState,
) {
  const query = search.trim().toLocaleLowerCase();
  const active = sorting[0] ?? APPLICATION_TABLE.defaultSort;
  const column = APPLICATION_COLUMNS.find(({ id }) => id === active.id) ?? APPLICATION_COLUMNS[0];
  return rows
    .filter(({ entity, description }) =>
      [entity.label, entity.id, description ?? ""].some((value) => value.toLocaleLowerCase().includes(query)),
    )
    .sort(
      (left, right) =>
        compareValues(left.values[column.id], right.values[column.id], active.desc) ||
        left.entity.label.localeCompare(right.entity.label) ||
        left.entity.id.localeCompare(right.entity.id),
    );
}
