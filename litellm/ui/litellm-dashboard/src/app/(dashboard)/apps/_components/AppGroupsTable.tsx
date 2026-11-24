import { useMemo } from "react";
import { useApplicationUsage } from "./useApplicationUsage";
import { useTranslation } from "react-i18next";
import { Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ErrorState } from "@/components/shared/ErrorState";
import { extractProxyErrorMessage } from "@/lib/http/client";
import ApplicationUsageTable from "./ApplicationUsageTable";
import { appGroupTableRows } from "./applicationTableData";
import { useAppGroupBudgetGroups } from "./useAppBudgetGroups";
import type { AppGroup } from "./appGroupApi";

export default function AppGroupsTable({
  groups,
  loading,
  onOpen,
  onCreate,
  onEdit,
}: {
  groups: readonly AppGroup[];
  loading: boolean;
  onOpen: (id: string) => void;
  onCreate: () => void;
  onEdit: (group: AppGroup) => void;
}) {
  const { t } = useTranslation();
  const catalog = useAppGroupBudgetGroups();
  const ids = useMemo(() => groups.map((group) => group.app_group_id).sort(), [groups]);
  const usage = useApplicationUsage(ids, true, "appGroupId");
  const rows = useMemo(
    () =>
      appGroupTableRows(
        groups,
        catalog.data?.default_budgets,
        usage.data,
        t("customerPortal.organizationGroups.customBadge"),
      ),
    [groups, catalog.data, usage.data, t],
  );
  return (
    <Card className="gap-0 overflow-hidden py-0">
      {usage.isError && (
        <ErrorState message={extractProxyErrorMessage(usage.error)} onRetry={() => void usage.refetch()} />
      )}
      <ApplicationUsageTable
        rows={rows}
        loading={loading}
        kind="groups"
        overallSpend={usage.data?.get(null)?.spend ?? null}
        onOpen={onOpen}
        onEdit={(id) => {
          const group = groups.find((row) => row.app_group_id === id);
          if (group) onEdit(group);
        }}
        actions={
          <Button className="ml-auto" onClick={onCreate}>
            <Plus /> {t("apps.groups.create")}
          </Button>
        }
      />
    </Card>
  );
}
