import { useTranslation } from "react-i18next";
import type { SortingState } from "@tanstack/react-table";
import { createSerializer, parseAsString } from "nuqs";
import { TableCell, TableRow } from "@/components/ui/table";
import { uiHref } from "@/utils/uiHref";
import { $api } from "@/lib/http/api";
import { customerOrganizationBudgetSource, type CustomerOrganizationRow } from "./customerOrganizations";
import { CustomerSpendCells, CustomerSpendIdentity, CustomerSpendTier } from "./CustomerSpendCells";
import CustomerTokenTotal from "./CustomerTokenTotal";
import { EMPTY_CUSTOMER_USAGE, type CustomerUsageTotals } from "./customerUsageTotals";
import { CustomerUsageEditButton, CustomerUsageTable, CustomerUsageTableHeaders } from "./CustomerUsageTable";

const usageHref = createSerializer({ org: parseAsString, from: parseAsString, to: parseAsString });
const COLUMNS = [
  { id: "customer", label: "customerPortal.list.columns.customer", defaultDescending: false },
  { id: "group", label: "customerPortal.list.columns.tier" },
  { id: "spend", label: "customerPortal.list.columns.monthlySpend", defaultDescending: true },
  { id: "budget", label: "customerPortal.hierarchy.maxBudget", defaultDescending: false },
  { id: "share", label: "customerPortal.list.columns.overallSpendShare", defaultDescending: true },
  { id: "tokens", label: "customerPortal.list.columns.tokens" },
  { id: "status", label: "customerPortal.list.columns.status", defaultDescending: true },
  { id: "actions", label: "apps.actions" },
] as const;

export default function CustomerOrganizationsTable({
  rows,
  sorting,
  onSort,
  loading,
  searching,
  startDate,
  endDate,
  usageTotals,
  canManage,
  onEdit,
}: {
  rows: readonly CustomerOrganizationRow[];
  sorting: SortingState;
  onSort: (sorting: SortingState) => void;
  loading: boolean;
  searching: boolean;
  startDate: string;
  endDate: string;
  usageTotals: CustomerUsageTotals | null;
  canManage: boolean;
  onEdit: (organizationId: string) => void;
}) {
  const { t } = useTranslation();
  const groups = $api.useQuery("get", "/management/v1/customer_organization_budget_groups", {});
  const emptyMessage =
    rows.length === 0 ? t(searching ? "customerPortal.list.noMatches" : "customerPortal.list.empty") : undefined;
  return (
    <CustomerUsageTable
      label={t("customerPortal.list.title")}
      columnCount={COLUMNS.length}
      headers={<CustomerUsageTableHeaders columns={COLUMNS} sorting={sorting} onSort={onSort} />}
      message={loading ? t("customerPortal.list.loading") : emptyMessage}
    >
      {!loading &&
        rows.map(({ organization, entity, monthlySpend, totalMonthlySpend }) => {
          const budgetSource = customerOrganizationBudgetSource(organization, groups.data?.default_budgets);
          const customLabel =
            budgetSource?.kind === "custom" ? t("customerPortal.organizationGroups.customBadge") : null;
          const budgetGroup =
            budgetSource?.kind === "group"
              ? budgetSource.name ?? t("customerPortal.organizationGroups.groupBadge")
              : customLabel;
          const usage = usageTotals ? usageTotals.get(organization.customer_org_id) ?? EMPTY_CUSTOMER_USAGE : null;
          return (
            <TableRow key={organization.customer_org_id}>
              <TableCell className="px-4 py-5">
                <CustomerSpendIdentity
                  entity={entity}
                  usageLayout
                  nameClassName="max-w-40 hover:underline"
                  idClassName="max-w-40"
                  href={usageHref(uiHref("customers/usage"), {
                    org: organization.customer_org_id,
                    from: startDate,
                    to: endDate,
                  })}
                />
              </TableCell>
              <CustomerSpendTier entity={{ ...entity, budgetGroup: budgetGroup ?? "-" }} />
              <CustomerSpendCells
                entity={entity}
                monthlySpend={monthlySpend}
                yearlySpend={null}
                tokens={usage?.tokens ?? null}
                tokenContent={<CustomerTokenTotal usage={usage} />}
                organizationSpend={totalMonthlySpend > 0 ? totalMonthlySpend : 0}
                shareLabel={t("customerPortal.list.columns.overallSpendShare")}
                usageLayout
                showYearly={false}
                spendClassName="max-w-lg"
              />
              <TableCell className="px-4 text-right">
                <CustomerUsageEditButton
                  type="button"
                  disabled={!canManage}
                  onClick={() => onEdit(organization.customer_org_id)}
                >
                  {t("customerPortal.common.edit")}
                </CustomerUsageEditButton>
              </TableCell>
            </TableRow>
          );
        })}
    </CustomerUsageTable>
  );
}
