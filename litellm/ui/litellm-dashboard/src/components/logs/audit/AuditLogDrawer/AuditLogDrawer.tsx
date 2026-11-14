import { copyToClipboard } from "@/utils/dataUtils";
import { Check, Copy } from "lucide-react";
import { useState, useCallback } from "react";
import { useTranslation } from "react-i18next";
import moment from "moment";
import { AuditLogEntry, AUDIT_TABLE_NAME_DISPLAY } from "../AuditLogsTableColumns";
import DefaultProxyAdminTag from "../../../common_components/DefaultProxyAdminTag";
import CopyButton from "@/components/shared/CopyButton";
import { StatusBadge, type StatusTone } from "@/components/shared/table_cells/status_badge";
import { Button } from "@/components/ui/button";
import { Sheet, SheetTitle } from "@/components/ui/sheet";
import { ResizableSheetContent } from "@/components/ui/ResizableSheetContent";

interface AuditLogDrawerProps {
  open: boolean;
  onClose: () => void;
  log: AuditLogEntry | null;
}

const ACTION_TONE: Record<string, StatusTone> = {
  created: "success",
  updated: "info",
  deleted: "error",
  rotated: "warning",
  kill_switch_fired: "error",
};

function CopyableJsonBlock({ label, value }: { label: string; value: Record<string, any> }) {
  const { t } = useTranslation();
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    if (await copyToClipboard(JSON.stringify(value, null, 2))) {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  }, [value]);

  return (
    <div className="overflow-hidden rounded-sm border border-border bg-card">
      <div className="flex items-center justify-between border-b border-border bg-muted px-3 py-2">
        <span className="text-xs font-semibold text-muted-foreground">{label}</span>
        <Button
          variant="ghost"
          size="icon-xs"
          onClick={handleCopy}
          title={t("logs.audit.drawer.copyJson")}
          aria-label={t("logs.audit.drawer.copyJson")}
        >
          {copied ? <Check className="text-success" /> : <Copy />}
        </Button>
      </div>
      <pre className="m-0 max-h-96 overflow-auto bg-card p-3 font-mono text-xs break-all whitespace-pre-wrap">
        {JSON.stringify(value, null, 2)}
      </pre>
    </div>
  );
}

function MetadataRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-start gap-2 py-1.5">
      <span className="w-36 shrink-0 text-xs text-muted-foreground">{label}</span>
      <span className="text-xs break-all text-foreground">{value}</span>
    </div>
  );
}

function DiffSection({ log }: { log: AuditLogEntry }) {
  const { t } = useTranslation();
  const { action, table_name, before_value, updated_values } = log;
  const isKeyTable = table_name === "LiteLLM_VerificationToken";
  const isUpdateAction = action === "updated" || action === "rotated";

  let displayBefore = before_value;
  let displayAfter = updated_values;

  if (isUpdateAction && before_value && updated_values) {
    const changedBefore: Record<string, any> = {};
    const changedAfter: Record<string, any> = {};
    const allKeys = new Set([...Object.keys(before_value), ...Object.keys(updated_values)]);

    allKeys.forEach((key) => {
      const bStr = JSON.stringify(before_value[key]);
      const aStr = JSON.stringify(updated_values[key]);
      if (bStr !== aStr) {
        if (key in before_value) changedBefore[key] = before_value[key];
        if (key in updated_values) changedAfter[key] = updated_values[key];
      }
    });

    // Fields only in before (removed)
    Object.keys(before_value).forEach((key) => {
      if (!(key in updated_values) && !(key in changedBefore)) {
        changedBefore[key] = before_value[key];
        changedAfter[key] = undefined;
      }
    });

    // Fields only in after (added)
    Object.keys(updated_values).forEach((key) => {
      if (!(key in before_value) && !(key in changedAfter)) {
        changedAfter[key] = updated_values[key];
        changedBefore[key] = undefined;
      }
    });

    displayBefore =
      Object.keys(changedBefore).length > 0 ? changedBefore : { note: t("logs.audit.drawer.noDifferingFields") };
    displayAfter =
      Object.keys(changedAfter).length > 0 ? changedAfter : { note: t("logs.audit.drawer.noDifferingFields") };
  }

  const renderValue = (label: string, value: Record<string, any> | null | undefined) => {
    if (!value || Object.keys(value).length === 0) {
      return (
        <div className="overflow-hidden rounded-sm border border-border bg-card">
          <div className="flex items-center border-b border-border bg-muted px-3 py-2">
            <span className="text-xs font-semibold text-muted-foreground">{label}</span>
          </div>
          <p className="m-0 px-3 py-3 text-xs text-muted-foreground italic">{t("logs.audit.drawer.notAvailable")}</p>
        </div>
      );
    }

    // For key table updates, show only meaningful fields as plain text
    if (isKeyTable && isUpdateAction) {
      const knownKeyFields = ["token", "spend", "max_budget"];
      const hasOnlyKnown = Object.keys(value).every((k) => knownKeyFields.includes(k));
      if (hasOnlyKnown && !("note" in value)) {
        return (
          <div className="overflow-hidden rounded-sm border border-border bg-card">
            <div className="flex items-center border-b border-border bg-muted px-3 py-2">
              <span className="text-xs font-semibold text-muted-foreground">{label}</span>
            </div>
            <div className="space-y-1 px-3 py-3 text-xs">
              {value.token !== undefined && (
                <p>
                  <span className="text-muted-foreground">{t("logs.audit.drawer.token")}</span>{" "}
                  {value.token ?? t("logs.audit.drawer.notAvailable")}
                </p>
              )}
              {value.spend !== undefined && (
                <p>
                  <span className="text-muted-foreground">{t("logs.audit.drawer.spend")}</span> $
                  {Number(value.spend).toFixed(6)}
                </p>
              )}
              {value.max_budget !== undefined && (
                <p>
                  <span className="text-muted-foreground">{t("logs.audit.drawer.maxBudget")}</span> $
                  {Number(value.max_budget).toFixed(6)}
                </p>
              )}
            </div>
          </div>
        );
      }
    }

    return <CopyableJsonBlock label={label} value={value} />;
  };

  return (
    <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-2">
      {renderValue(t("logs.audit.drawer.before"), displayBefore)}
      {renderValue(t("logs.audit.drawer.after"), displayAfter)}
    </div>
  );
}

export function AuditLogDrawer({ open, onClose, log }: AuditLogDrawerProps) {
  const { t } = useTranslation();
  if (!log) return null;

  const tableDisplay = Object.hasOwn(AUDIT_TABLE_NAME_DISPLAY, log.table_name)
    ? t(AUDIT_TABLE_NAME_DISPLAY[log.table_name])
    : log.table_name;

  return (
    <Sheet open={open} onOpenChange={(nextOpen) => !nextOpen && onClose()}>
      <ResizableSheetContent storageKey="litellm:auditLogDrawerWidth" className="gap-0 p-0">
        <SheetTitle className="sr-only">{t("logs.audit.drawer.title")}</SheetTitle>

        <div className="flex shrink-0 items-center gap-3 border-b border-border bg-card px-6 py-4">
          <StatusBadge tone={ACTION_TONE[log.action] ?? "neutral"} label={log.action} />
          <span className="text-sm text-muted-foreground">
            {moment.utc(log.updated_at).local().format("MMM D, YYYY HH:mm:ss")}
          </span>
        </div>

        <div className="px-6 py-5">
          <div className="mb-5 rounded-lg border border-border bg-muted p-4">
            <p className="mb-2 text-xs font-semibold tracking-wide text-foreground uppercase">
              {t("logs.audit.drawer.details")}
            </p>
            <MetadataRow label={t("logs.audit.drawer.table")} value={tableDisplay} />
            <MetadataRow
              label={t("logs.audit.drawer.objectId")}
              value={
                <span className="inline-flex items-center gap-1 font-mono text-xs">
                  {log.object_id}
                  <CopyButton value={log.object_id} label={t("logs.audit.drawer.copyObjectId")} />
                </span>
              }
            />
            <MetadataRow
              label={t("logs.audit.drawer.changedBy")}
              value={<DefaultProxyAdminTag userId={log.changed_by} />}
            />
            <MetadataRow
              label={t("logs.audit.drawer.apiKeyHash")}
              value={
                log.changed_by_api_key ? (
                  <span className="inline-flex items-center gap-1 font-mono text-xs break-all">
                    {log.changed_by_api_key}
                    <CopyButton value={log.changed_by_api_key} label={t("logs.audit.drawer.copyApiKeyHash")} />
                  </span>
                ) : (
                  "—"
                )
              }
            />
          </div>

          <DiffSection log={log} />
        </div>
      </ResizableSheetContent>
    </Sheet>
  );
}
