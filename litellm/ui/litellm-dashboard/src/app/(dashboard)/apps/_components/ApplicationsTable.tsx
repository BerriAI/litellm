import { useMemo } from "react";
import { useApplicationUsage } from "./useApplicationUsage";
import { useTranslation } from "react-i18next";
import { ErrorState } from "@/components/shared/ErrorState";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { useOverallAppBudgetGroups } from "./useAppBudgetGroups";
import { type ApplicationRow } from "./applications";
import { applicationTableRows } from "./applicationTableData";
import ApplicationUsageTable from "./ApplicationUsageTable";

export default function ApplicationsTable({
  rows,
  loading,
  onOpen,
  keyPrefix,
}: {
  rows: readonly ApplicationRow[];
  loading: boolean;
  onOpen?: (resourceId: string) => void;
  keyPrefix?: string;
}) {
  const { t } = useTranslation();
  const groups = useOverallAppBudgetGroups(Boolean(onOpen));
  const appIds = useMemo(() => rows.flatMap(({ app }) => (app ? [app.app_id] : [])).sort(), [rows]);
  const usage = useApplicationUsage(appIds, Boolean(onOpen));
  const tableRows = useMemo(
    () =>
      applicationTableRows(
        rows,
        onOpen && groups.isSuccess ? groups.data.default_budgets : undefined,
        usage.data,
        t("customerPortal.organizationGroups.customBadge"),
      ),
    [rows, onOpen, groups.isSuccess, groups.data, usage.data, t],
  );
  return (
    <>
      {usage.isError && (
        <ErrorState message={extractProxyErrorMessage(usage.error)} onRetry={() => void usage.refetch()} />
      )}
      <ApplicationUsageTable
        rows={tableRows}
        loading={loading}
        onEdit={onOpen}
        keyPrefix={keyPrefix}
        overallSpend={usage.data?.get(null)?.spend ?? null}
      />
    </>
  );
}
