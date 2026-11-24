import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardAction } from "@/components/ui/card";
import { TableCell, TableHead, TableRow } from "@/components/ui/table";
import { ErrorState } from "@/components/shared/ErrorState";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { customerOrganizationsCall } from "../../customers/_components/customerOrganizations";
import { CustomerUsageTable, CustomerUsageEditButton } from "../../customers/_components/CustomerUsageTable";
import useAuthorized from "../../hooks/useAuthorized";
import AppBudgetLimits from "./AppBudgetLimits";
import type { GroupBudgetTarget } from "./AppGroupBudgetDialog";
import { loadAppGroupOrganizations, type AppGroup } from "./appGroupApi";

export default function AppGroupCustomers({
  group,
  onEdit,
}: {
  group: AppGroup;
  onEdit: (target: GroupBudgetTarget) => void;
}) {
  const { t } = useTranslation();
  const { accessToken, userId } = useAuthorized();
  const associations = useQuery({
    queryKey: ["app-group-organizations", group.app_group_id],
    queryFn: () => loadAppGroupOrganizations(group.app_group_id),
  });
  const organizations = useQuery({
    queryKey: ["customer-organizations", userId],
    queryFn: ({ signal }) => customerOrganizationsCall(accessToken!, signal),
    enabled: Boolean(accessToken),
  });
  const emptyMessage = associations.data?.length === 0 ? t("apps.groups.noCustomers") : undefined;
  const labels = new Map(organizations.data?.map((org) => [org.customer_org_id, org.alias || org.customer_org_id]));
  return (
    <Card className="overflow-hidden pb-0">
      <CardHeader>
        <CardTitle>{t("apps.groups.customers")}</CardTitle>
        <CardAction>
          <Button variant="outline" size="sm" onClick={() => onEdit({ kind: "organization", group })}>
            <Plus /> {t("apps.groups.setCustomerBudget")}
          </Button>
        </CardAction>
      </CardHeader>
      {associations.isError ? (
        <ErrorState
          message={extractProxyErrorMessage(associations.error)}
          onRetry={() => void associations.refetch()}
        />
      ) : (
        <CustomerUsageTable
          label={t("apps.groups.customers")}
          columnCount={3}
          message={associations.isPending ? t("apps.loading") : emptyMessage}
          headers={
            <>
              <TableHead className="px-4">{t("apps.groups.customer")}</TableHead>
              <TableHead className="px-4">{t("apps.groups.sharedBudget")}</TableHead>
              <TableHead className="px-4 text-right">{t("apps.actions")}</TableHead>
            </>
          }
        >
          {associations.data?.map((organization) => (
            <TableRow key={organization.customer_org_id}>
              <TableCell className="px-4 py-5">
                {labels.get(organization.customer_org_id) ?? organization.customer_org_id}
              </TableCell>
              <TableCell className="px-4">
                <AppBudgetLimits
                  budget={organization.budget}
                  usage={organization.budget_usage}
                  cumulativeSpend={organization.spend}
                />
              </TableCell>
              <TableCell className="px-4 text-right">
                <CustomerUsageEditButton onClick={() => onEdit({ kind: "organization", group, organization })}>
                  {t("apps.groups.editBudget")}
                </CustomerUsageEditButton>
              </TableCell>
            </TableRow>
          ))}
        </CustomerUsageTable>
      )}
    </Card>
  );
}
