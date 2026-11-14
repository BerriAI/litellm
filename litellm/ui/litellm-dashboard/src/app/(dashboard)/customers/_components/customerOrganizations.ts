import type { SortingState } from "@tanstack/react-table";
import { apiClient } from "@/components/networking";
import type { DailyActivityAggregatedResponse } from "@/components/UsagePage/dailyActivityApi";
import type { components } from "@/lib/http/schema";
import i18n from "@/i18n/i18n";
import { customerBudgetPeriod, customerBudgetStatus, customerBudgetUsage } from "./customerBudgetUsage";
import { mergeCustomerActivityPages } from "./customerUsageData";
import { customerUsageTotals } from "./customerUsageTotals";

export type CustomerOrganization = components["schemas"]["Organization"];
type OrganizationList = components["schemas"]["ListResponse_Organization_"];
type Activity = ReturnType<typeof mergeCustomerActivityPages>;

export const CUSTOMER_ORGANIZATION_TABLE = {
  sortFields: ["created_at", "customer", "spend", "share", "status"],
  defaultSort: { id: "created_at", desc: true },
  defaultPageSize: 25,
  filterColumns: [],
};

const PREFIX = "/management/v1/customer_organizations";
const PAGE_SIZE = 100;
const MAX_PAGES = 1_000;
const MAX_CONCURRENT = 4;

export async function customerOrganizationsCall(accessToken: string, signal?: AbortSignal) {
  const rows: CustomerOrganization[] = [];
  for (let page = 1; page <= MAX_PAGES; page++) {
    const result = await apiClient.get<OrganizationList>(PREFIX, {
      accessToken,
      signal,
      query: { page, page_size: PAGE_SIZE },
    });
    rows.push(...result.data);
    if (page >= result.meta.total_pages)
      return [...new Map(rows.map((organization) => [organization.customer_org_id, organization])).values()];
  }
  throw new Error(i18n.t("customerPortal.analytics.incompletePages"));
}

export async function customerOrganizationsUsageCall({
  accessToken,
  ids,
  startDate,
  endDate,
  signal,
}: {
  accessToken: string;
  ids: readonly string[];
  startDate: string;
  endDate: string;
  signal?: AbortSignal;
}): Promise<Activity> {
  if (ids.some((id) => !id)) throw new Error(i18n.t("customerPortal.analytics.invalidIdentifier"));
  const uniqueIds = [...new Set(ids)];
  const needsExactFilter = (id: string) => id.includes(",") || id.trim() !== id;
  const batchIds = uniqueIds.filter((id) => !needsExactFilter(id));
  const filters = [
    ...Array.from({ length: Math.ceil(batchIds.length / PAGE_SIZE) }, (_, index) => ({
      customer_org_ids: batchIds.slice(index * PAGE_SIZE, (index + 1) * PAGE_SIZE).join(","),
    })),
    ...uniqueIds.filter(needsExactFilter).map((id) => ({ customer_org_id: id })),
  ];
  const responses: DailyActivityAggregatedResponse[] = [];
  for (let start = 0; start < filters.length; start += MAX_CONCURRENT) {
    const batch = await Promise.all(
      filters.slice(start, start + MAX_CONCURRENT).map((filter) =>
        apiClient.get<DailyActivityAggregatedResponse>(`${PREFIX}/daily/activity/aggregated`, {
          accessToken,
          signal,
          query: { start_date: startDate, end_date: endDate, ...filter },
        }),
      ),
    );
    if (batch.some((response) => !response.metadata || response.metadata.has_more))
      throw new Error(i18n.t("customerPortal.analytics.incompletePages"));
    responses.push(...batch);
  }
  return mergeCustomerActivityPages(responses);
}

export const customerOrganizationRows = (organizations: readonly CustomerOrganization[], monthlyUsage: Activity) => {
  const totals = customerUsageTotals(monthlyUsage.results);
  return organizations.map((organization) => {
    const monthlySpend = totals[organization.customer_org_id]?.spend ?? 0;
    const budgetEntity = {
      id: organization.customer_org_id,
      label: organization.alias || organization.customer_org_id,
      spend: organization.spend,
      budgetLimit: organization.budget?.max_budget ?? null,
      budgetDuration: organization.budget?.budget_duration ?? null,
      budgetLimits: organization.budget?.budget_limits ?? [],
      budgetResetAt: organization.budget_reset_at ?? null,
      budgetUsage: organization.budget_usage,
    };
    const budgets = customerBudgetUsage(budgetEntity);
    const monthlyBudgets = budgets.filter((budget) => customerBudgetPeriod(budget.budget_duration) === "monthly");
    return {
      organization,
      monthlySpend,
      monthlyBudgets,
      displaySpend: monthlyBudgets.length > 0 ? monthlyBudgets[0].spend : monthlySpend,
      share: monthlyUsage.metadata.total_spend > 0 ? monthlySpend / monthlyUsage.metadata.total_spend : 0,
      status: customerBudgetStatus(budgets),
    };
  });
};

export type CustomerOrganizationRow = ReturnType<typeof customerOrganizationRows>[number];

const STATUS_ORDER = {
  "customerPortal.analytics.noLimit": 0,
  "customerPortal.analytics.withinBudget": 1,
  "customerPortal.analytics.budgetUsageUnavailable": 2,
  "customerPortal.analytics.reachedLimit": 3,
} as const;

const sortValue = (row: CustomerOrganizationRow, field: string): string | number | null => {
  switch (field) {
    case "customer":
      return row.organization.alias || row.organization.customer_org_id;
    case "spend":
      return row.displaySpend;
    case "share":
      return row.share;
    case "status":
      return STATUS_ORDER[row.status];
    default:
      return Date.parse(row.organization.created_at);
  }
};

export function sortCustomerOrganizations(
  rows: readonly CustomerOrganizationRow[],
  search: string,
  sorting: SortingState,
): CustomerOrganizationRow[] {
  const query = search.trim().toLocaleLowerCase();
  const active = sorting[0] ?? CUSTOMER_ORGANIZATION_TABLE.defaultSort;
  return rows
    .filter(({ organization }) =>
      [organization.customer_org_id, organization.alias ?? ""].some((value) =>
        value.toLocaleLowerCase().includes(query),
      ),
    )
    .sort((left, right) => {
      const a = sortValue(left, active.id);
      const b = sortValue(right, active.id);
      if (a === null && b === null) return 0;
      if (a === null) return 1;
      if (b === null) return -1;
      const comparison = typeof a === "string" && typeof b === "string" ? a.localeCompare(b) : Number(a) - Number(b);
      return (
        comparison * (active.desc ? -1 : 1) ||
        left.organization.customer_org_id.localeCompare(right.organization.customer_org_id)
      );
    });
}
