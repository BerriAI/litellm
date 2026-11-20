import { useTranslation } from "react-i18next";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { cn } from "@/lib/cva.config";
import AppBudgetLimits from "./AppBudgetLimits";
import { APP_RESOURCE_LABELS } from "./appForms";
import { applicationInitials, type ApplicationRow } from "./applications";

const COLUMNS = [
  { id: "application", label: "apps.list.columns.application" },
  { id: "spend", label: "apps.list.columns.spend" },
  { id: "surcharge", label: "apps.list.columns.surcharge" },
  { id: "customers", label: "apps.list.columns.newCustomers" },
  { id: "status", label: "customerPortal.list.columns.status" },
] as const;

const badge = "inline-flex whitespace-nowrap rounded-full bg-muted px-2 py-0.5 text-xs";

export default function ApplicationsTable({
  rows,
  loading,
  searching,
  onOpen,
}: {
  rows: readonly ApplicationRow[];
  loading: boolean;
  searching: boolean;
  onOpen?: (resourceId: string) => void;
}) {
  const { t } = useTranslation();
  return (
    <Table aria-label={t("apps.title")}>
      <TableHeader>
        <TableRow>
          {COLUMNS.map(({ id, label }) => (
            <TableHead
              key={id}
              className={cn("px-4 text-sm", id === "spend" && "min-w-60", id === "status" && "text-right")}
            >
              {t(label)}
            </TableHead>
          ))}
        </TableRow>
      </TableHeader>
      <TableBody>
        {loading && (
          <TableRow>
            <TableCell colSpan={COLUMNS.length} className="py-8 text-center text-muted-foreground">
              {t("apps.list.loading")}
            </TableCell>
          </TableRow>
        )}
        {!loading && rows.length === 0 && (
          <TableRow>
            <TableCell colSpan={COLUMNS.length} className="py-8 text-center text-muted-foreground">
              {t(searching ? "apps.list.noMatches" : "apps.noResources")}
            </TableCell>
          </TableRow>
        )}
        {!loading &&
          rows.map(({ resource, app }) => (
            <TableRow key={resource.resource_id}>
              <TableCell className="px-4 py-5">
                <div className="flex items-center gap-3">
                  <span
                    aria-hidden="true"
                    className="flex size-9 shrink-0 items-center justify-center rounded-full bg-muted text-xs font-medium"
                  >
                    {applicationInitials(resource.alias)}
                  </span>
                  <div className="min-w-0">
                    {onOpen ? (
                      <button
                        type="button"
                        className="block max-w-52 truncate text-left font-medium hover:text-primary hover:underline"
                        title={resource.alias}
                        onClick={() => onOpen(resource.resource_id)}
                      >
                        {resource.alias}
                      </button>
                    ) : (
                      <p className="max-w-52 truncate font-medium" title={resource.alias}>
                        {resource.alias}
                      </p>
                    )}
                    <p
                      className="max-w-52 truncate font-mono text-[11px] text-muted-foreground"
                      title={resource.resource_id}
                    >
                      {resource.resource_id}
                    </p>
                    <p className="mt-1 text-xs text-muted-foreground">
                      {t(APP_RESOURCE_LABELS[resource.resource_type])}
                    </p>
                  </div>
                </div>
              </TableCell>
              <TableCell className="px-4">
                <AppBudgetLimits budget={app?.budget} usage={app?.budget_usage} cumulativeSpend={app?.spend ?? 0} />
              </TableCell>
              <TableCell className="px-4 tabular-nums">
                {app?.customer_margin === null || app?.customer_margin === undefined
                  ? t("apps.list.gatewaySurcharge")
                  : t("format.number", { val: app.customer_margin, style: "percent", maximumFractionDigits: 2 })}
              </TableCell>
              <TableCell className="px-4">
                <span
                  className={cn(badge, app?.auto_create_customers === false && "bg-destructive/10 text-destructive")}
                >
                  {t(
                    app?.auto_create_customers === false ? "apps.list.customersBlocked" : "apps.list.customersAllowed",
                  )}
                </span>
              </TableCell>
              <TableCell className="px-4 text-right">
                <span className={badge}>{t(app ? "apps.list.configured" : "apps.list.notConfigured")}</span>
              </TableCell>
            </TableRow>
          ))}
      </TableBody>
    </Table>
  );
}
